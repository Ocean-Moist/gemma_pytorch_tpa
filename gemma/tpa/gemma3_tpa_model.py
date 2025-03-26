###############################################################################
# gemma3_tpa_model.py
###############################################################################
"""
Inference-only Gemma model implementation with Tensor Product Attention (TPA).
"""

import gc
import json
import os
from typing import List, Sequence, Tuple, Union, Mapping, Any

import torch
from torch import nn

from .. import model as gemma_model
from .. import tokenizer


class RMSNorm(nn.Module):
    """RMS Normalization module."""
    def __init__(
            self,
            dim: int,
            eps: float = 1e-6,
            add_unit_offset: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        if self.add_unit_offset:
            output = output * (1 + self.weight.float())
        else:
            output = output * self.weight.float()
        return output.type_as(x)


import torch
from torch import nn
import torch.nn.functional as F

from .. import config as gemma_config

class TPAAttention(nn.Module):
    """
    Tensor Product Attention (TPA) with factor-only approach:
      - We never form full Q, K, V. Instead, we store only rank-limited factors:
          A_q, B_q,  A_k, B_k,  A_v, B_v.
      - The final attention scores and outputs are computed via efficient einsum
        without big materialized Q or K arrays.

    Note:
      - This version handles GQA if num_kv_heads < num_heads.
      - Also integrates rotary embeddings on the B_q and B_k factors.
      - Caches the A_k, B_k, A_v, B_v per token in a “factorized KV cache” to reduce memory.
    """

    def __init__(self, config: gemma_config.GemmaConfig, attn_type: gemma_config.AttentionType):
        super().__init__()
        self.config = config
        self.attn_type = attn_type

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)

        self.hidden_size = config.hidden_size
        if hasattr(config, 'head_dim'):
            self.head_dim = config.head_dim
        else:
            # fallback if no explicit head_dim:
            self.head_dim = self.hidden_size // self.num_heads

        # If q/k/v have special ranks, store them:
        self.q_rank = getattr(config, 'q_rank', 6)
        self.k_rank = getattr(config, 'k_rank', 2)
        self.v_rank = getattr(config, 'v_rank', 2)

        # optional query/key normalization:
        self.query_norm = (
            RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if getattr(config, 'use_qk_norm', False)
            else None
        )
        self.key_norm = (
            RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if getattr(config, 'use_qk_norm', False)
            else None
        )

        # This is the usual Transformer scaling: 1/sqrt(head_dim)
        if config.query_pre_attn_scalar is not None:
            self.scaling = (config.query_pre_attn_scalar) ** -0.5
        else:
            self.scaling = self.head_dim ** -0.5

        # A-projection: hidden_size -> (num_heads * rank)
        self.W_A_q = nn.Linear(self.hidden_size, self.num_heads * self.q_rank, bias=False)
        self.W_A_k = nn.Linear(self.hidden_size, self.num_kv_heads * self.k_rank, bias=False)
        self.W_A_v = nn.Linear(self.hidden_size, self.num_kv_heads * self.v_rank, bias=False)

        # B-projection: hidden_size -> (rank * head_dim)
        self.W_B_q = nn.Linear(self.hidden_size, self.q_rank * self.head_dim, bias=False)
        self.W_B_k = nn.Linear(self.hidden_size, self.k_rank * self.head_dim, bias=False)
        self.W_B_v = nn.Linear(self.hidden_size, self.v_rank * self.head_dim, bias=False)

        # final output projection
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # optional logit capping
        self.attn_logit_softcapping = getattr(config, 'attn_logit_softcapping', None)
        # local sliding window
        self.sliding_window_size = getattr(config, 'sliding_window_size', None)

        # Factor-only caches:  (batch_size, seq_len, #kv_heads, rank or head_dim)
        self.cache_kA = None
        self.cache_kB = None
        self.cache_vA = None
        self.cache_vB = None

    def _init_kv_cache(self, batch_size, max_seq_len):
        """
        Initialize the factorized KV caches:
          cache_kA, cache_kB, cache_vA, cache_vB.
        They each store rank-based factors rather than the full K/V arrays.
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        self.cache_kA = torch.zeros((batch_size, max_seq_len, self.num_kv_heads, self.k_rank),
                                    device=device, dtype=dtype)
        self.cache_kB = torch.zeros((batch_size, max_seq_len, self.k_rank, self.head_dim),
                                    device=device, dtype=dtype)
        self.cache_vA = torch.zeros((batch_size, max_seq_len, self.num_kv_heads, self.v_rank),
                                    device=device, dtype=dtype)
        self.cache_vB = torch.zeros((batch_size, max_seq_len, self.v_rank, self.head_dim),
                                    device=device, dtype=dtype)

    def forward(
            self,
            hidden_states: torch.Tensor,      # [batch_size, seq_len, hidden_size]
            freqs_cis: torch.Tensor,          # [seq_len, head_dim/2] in complex form, for rotary
            kv_write_indices: torch.Tensor,   # [seq_len_in_this_step] positions to write
            kv_cache,                         # not used directly but we read shape
            mask: torch.Tensor,               # [batch, 1, seq_len, seq_len] or similar
            local_mask: torch.Tensor = None,  # optional local sliding mask
    ):
        """
        Factor-only TPA forward:
          1) Compute A_q, A_k, A_v and B_q, B_k, B_v for current tokens.
          2) Store factor-KV in self.cache_kA etc.
          3) Reconstruct attn scores from factor expansions, apply softmax/mask.
          4) Reconstruct output from factor expansions of V.
          5) Return final [batch, seq_len, hidden_size].
        """
        bsz, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # factor-projection A:
        A_q = self.W_A_q(hidden_states)  # shape [b, seq, num_heads*q_rank]
        A_k = self.W_A_k(hidden_states)  # shape [b, seq, num_kv_heads*k_rank]
        A_v = self.W_A_v(hidden_states)  # shape [b, seq, num_kv_heads*v_rank]

        # factor-projection B:
        B_q = self.W_B_q(hidden_states)  # shape [b, seq, q_rank*head_dim]
        B_k = self.W_B_k(hidden_states)  # shape [b, seq, k_rank*head_dim]
        B_v = self.W_B_v(hidden_states)  # shape [b, seq, v_rank*head_dim]

        # reshape into (b, seq, #heads, rank) or (b, seq, rank, head_dim)
        A_q = A_q.view(bsz, seq_len, self.num_heads, self.q_rank)
        A_k = A_k.view(bsz, seq_len, self.num_kv_heads, self.k_rank)
        A_v = A_v.view(bsz, seq_len, self.num_kv_heads, self.v_rank)

        B_q = B_q.view(bsz, seq_len, self.q_rank, self.head_dim)
        B_k = B_k.view(bsz, seq_len, self.k_rank, self.head_dim)
        B_v = B_v.view(bsz, seq_len, self.v_rank, self.head_dim)

        # Apply rotary embedding to B_q and B_k
        B_q = self._apply_rotary_emb_to_B(B_q, freqs_cis)
        B_k = self._apply_rotary_emb_to_B(B_k, freqs_cis)

        # if needed, init the factor-kv cache
        if self.cache_kA is None or bsz > self.cache_kA.shape[0]:
            # kv_cache[0].shape is [batch_size, max_seq_len, #heads, head_dim], so kv_cache[0].shape[1]
            max_seq_len = kv_cache[0].shape[1]
            self._init_kv_cache(bsz, max_seq_len)

        # move the cache to correct device/dtype
        self.cache_kA = self.cache_kA.to(device=device, dtype=A_k.dtype)
        self.cache_kB = self.cache_kB.to(device=device, dtype=B_k.dtype)
        self.cache_vA = self.cache_vA.to(device=device, dtype=A_v.dtype)
        self.cache_vB = self.cache_vB.to(device=device, dtype=B_v.dtype)

        # index_copy into caches
        self.cache_kA[:bsz].index_copy_(1, kv_write_indices, A_k)
        self.cache_vA[:bsz].index_copy_(1, kv_write_indices, A_v)
        self.cache_kB[:bsz].index_copy_(1, kv_write_indices, B_k)
        self.cache_vB[:bsz].index_copy_(1, kv_write_indices, B_v)

        # read out the KV slices up to current position
        cache_len = kv_write_indices[-1] + 1 if kv_write_indices.numel() > 0 else 0
        A_k = self.cache_kA[:bsz, :cache_len]  # [b, kv_seq, #kv_heads, k_rank]
        A_v = self.cache_vA[:bsz, :cache_len]  # [b, kv_seq, #kv_heads, v_rank]
        B_k = self.cache_kB[:bsz, :cache_len]  # [b, kv_seq, k_rank, head_dim]
        B_v = self.cache_vB[:bsz, :cache_len]  # [b, kv_seq, v_rank, head_dim]

        kv_seq_len = A_k.shape[1]

        # If GQA => expand A_k, A_v from num_kv_heads to num_heads
        if self.num_kv_heads < self.num_heads:
            heads_per_kv = self.num_heads // self.num_kv_heads
            A_k = A_k.repeat_interleave(heads_per_kv, dim=2)
            A_v = A_v.repeat_interleave(heads_per_kv, dim=2)

        # Optionally apply RMSNorm to B_q, B_k if config says use_qk_norm
        if self.query_norm is not None and self.key_norm is not None:
            for r in range(self.q_rank):
                B_q[:, :, r, :] = self.query_norm(B_q[:, :, r, :])
            for r in range(self.k_rank):
                B_k[:, :, r, :] = self.key_norm(B_k[:, :, r, :])

        # B_dots: [b, q_len, kv_len, q_rank, k_rank]
        B_dots = torch.einsum('bqrd,bksd->bqkrs', B_q, B_k)
        B_dots = B_dots * self.scaling

        # attn_weights: [b, #heads, q_len, kv_len]
        attn_weights = torch.einsum(
            'bqhr,bkhs,bqkrs->bhqk',
            A_q, A_k, B_dots
        )

        if self.attn_logit_softcapping is not None:
            attn_weights = attn_weights / self.attn_logit_softcapping
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * self.attn_logit_softcapping

        if (
                self.attn_type == gemma_config.AttentionType.LOCAL_SLIDING
                and self.sliding_window_size is not None
                and local_mask is not None
        ):
            mask = local_mask

        if mask is not None:
            attn_weights = attn_weights + mask

        attn_probs = F.softmax(attn_weights.float(), dim=-1).type_as(hidden_states)

        # attn_output: [b, #heads, q_len, head_dim]
        attn_output = torch.einsum(
            'bhqk,bkhu,bkud->bhqd',
            attn_probs,
            A_v,
            B_v
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, seq_len, self.num_heads * self.head_dim)
        output = self.o_proj(attn_output)

        if torch.isnan(output).any():
            print("WARNING: NaN in TPA output, replacing with zero.")
            output = torch.nan_to_num(output, nan=0.0)

        return output

    def _apply_rotary_emb_to_B(self, B: torch.Tensor, freqs_cis: torch.Tensor):
        """
        Factor-only RoPE: apply rotary to each rank slice of B.
        B shape: [b, seq, rank, head_dim]
        freqs_cis shape: [seq, head_dim//2].
        """
        bsz, seq_len, rank, hd = B.shape
        if hd % 2 != 0:
            raise ValueError("Head dim must be even to apply rotary embeddings")

        B_out = torch.zeros_like(B)
        for t in range(seq_len):
            freqs = freqs_cis[t]
            for r in range(rank):
                b_slice = B[:, t, r, :]
                b_complex = torch.view_as_complex(b_slice.float().reshape(bsz, -1, 2))
                b_rotated = b_complex * freqs
                b_rotated = b_rotated.view(bsz, -1, 2)
                b_rotated = torch.view_as_real(b_rotated)
                b_rotated = b_rotated.reshape(bsz, hd)
                B_out[:, t, r, :] = b_rotated.type_as(B)
        return B_out


class TPADecoderLayer(nn.Module):
    """Gemma decoder layer using TPA attention."""

    def __init__(
            self,
            config: gemma_config.GemmaConfig,
            attn_type: gemma_config.AttentionType,
    ):
        super().__init__()
        self.attn_type = attn_type
        self.self_attn = TPAAttention(
            config=config,
            attn_type=self.attn_type,
        )
        self.mlp = gemma_model.GemmaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant=config.quant,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if getattr(config, 'use_pre_ffw_norm', False)
            else None
        )
        self.post_feedforward_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if getattr(config, 'use_post_ffw_norm', False)
            else None
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            freqs_cis: torch.Tensor,
            kv_write_indices: torch.Tensor,
            kv_cache: Tuple[torch.Tensor, torch.Tensor],
            mask: torch.Tensor,
            local_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
            mask=mask,
            local_mask=local_mask,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        if self.pre_feedforward_layernorm is not None:
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.post_feedforward_layernorm is not None:
            hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class TPAModel(nn.Module):
    """Gemma model with TPA attention."""

    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            if config.architecture in (
                    gemma_config.Architecture.GEMMA_2,
                    gemma_config.Architecture.GEMMA_3,
            ):
                attn_type = (
                    config.attn_types[i % len(config.attn_types)]
                    if config.attn_types is not None
                    else gemma_config.AttentionType.GLOBAL
                )
                self.layers.append(TPADecoderLayer(config, attn_type))
            else:
                raise ValueError(f'Unsupported architecture: {config.architecture}')
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            freqs_cis: Mapping[gemma_config.AttentionType, torch.Tensor],
            kv_write_indices: torch.Tensor,
            kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
            mask: torch.Tensor,
            local_mask: torch.Tensor,
    ) -> torch.Tensor:
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis.get(layer.attn_type),
                kv_write_indices=kv_write_indices,
                kv_cache=kv_caches[i],
                mask=mask,
                local_mask=local_mask,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class GemmaForCausalLMwithTPA(nn.Module):
    """Gemma model for causal language modeling with TPA attention."""
    def __init__(
            self,
            config: gemma_config.GemmaConfig,
    ):
        super().__init__()
        self.dtype = config.get_dtype()
        self.config = config
        max_seq_len = config.max_position_embeddings
        head_dim = config.head_dim
        vocab_size = config.vocab_size
        self.tokenizer = tokenizer.Tokenizer(config.tokenizer)
        self.text_token_embedder = gemma_model.Embedding(vocab_size, config.hidden_size, config.quant)

        # Use our TPA model instead of standard GemmaModel
        self.model = TPAModel(config)

        self.sampler = gemma_model.Sampler(vocab_size, config)

        if config.rope_wave_length is None:
            raise ValueError('rope_wave_length must be provided for Gemma3.')
        rope_lengths = config.rope_wave_length
        defaults = {
            gemma_config.AttentionType.LOCAL_SLIDING: 10_000,
            gemma_config.AttentionType.GLOBAL: 10_000,
        }
        self._register_freqs_cis('local_freqs_cis', head_dim, max_seq_len,
                                 theta=rope_lengths.get(
                                     gemma_config.AttentionType.LOCAL_SLIDING,
                                     defaults[gemma_config.AttentionType.LOCAL_SLIDING]
                                 ))
        self._register_freqs_cis('global_freqs_cis', head_dim, max_seq_len,
                                 theta=rope_lengths.get(
                                     gemma_config.AttentionType.GLOBAL,
                                     defaults[gemma_config.AttentionType.GLOBAL]
                                 ),
                                 rope_scaling_factor=config.rope_scaling_factor)

    def _register_freqs_cis(
            self, name: str, head_dim: int, max_seq_len: int, theta: int = 10_000, rope_scaling_factor: int = 1
    ):
        self.register_buffer(
            name,
            gemma_model.precompute_freqs_cis(head_dim, max_seq_len * 2,
                                             theta=theta, rope_scaling_factor=rope_scaling_factor)
        )

    @torch.no_grad()
    def forward(self,
                input_token_ids: torch.Tensor,
                input_positions: torch.Tensor = None,
                kv_caches: List[Tuple[torch.Tensor, torch.Tensor]] = None,
                mask: torch.Tensor = None,
                output_positions: torch.Tensor = None,
                temperatures: Union[torch.Tensor, None] = None,
                top_ps: torch.Tensor = None,
                top_ks: torch.Tensor = None,
                local_mask: torch.Tensor = None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:

        # Debug printing
        # print(f"DEBUG TPA FORWARD: input_token_ids shape: {input_token_ids.shape}, device: {input_token_ids.device}")
        # print(f"DEBUG TPA FORWARD: input_positions: {input_positions}, device: {input_positions.device}")

        # Ensure freqs_cis buffers on same device
        if self.local_freqs_cis.device != input_positions.device:
            self.local_freqs_cis = self.local_freqs_cis.to(input_positions.device)
            self.global_freqs_cis = self.global_freqs_cis.to(input_positions.device)

        freqs_cis = {}
        freqs_cis[gemma_config.AttentionType.LOCAL_SLIDING] = self.local_freqs_cis.index_select(0, input_positions)
        freqs_cis[gemma_config.AttentionType.GLOBAL] = self.global_freqs_cis.index_select(0, input_positions)

        # Move text_token_embedder if needed
        if hasattr(self.text_token_embedder, 'weight') and self.text_token_embedder.weight.device != input_token_ids.device:
            self.text_token_embedder = self.text_token_embedder.to(input_token_ids.device)

        hidden_states = self.text_token_embedder(input_token_ids)

        normalizer = torch.tensor(self.config.hidden_size ** 0.5,
                                  dtype=hidden_states.dtype,
                                  device=hidden_states.device)
        hidden_states = hidden_states * normalizer

        # We will pass kv_write_indices as input_positions:
        kv_write_indices = input_positions

        # ----------------------------------------------------------------
        # FIX: slice or index the last dimension of mask and local_mask
        # so that their shape matches attn_weights [b, heads, q_len, kv_len].
        # kv_len is kv_write_indices[-1]+1 if we have new tokens
        # or 0 if empty. Then the last dimension of mask becomes kv_len.
        if mask is not None:
            # Typically we already do something like:
            # mask = mask.index_select(2, input_positions)
            # but we must also slice the last dim to kv_len:
            if kv_write_indices.numel() > 0:
                kv_len = kv_write_indices[-1].item() + 1
            else:
                kv_len = 0
            # shape: [batch, 1, q_len, max_seq_len] -> [batch, 1, q_len, kv_len]
            mask = mask[..., :kv_len]

        if local_mask is not None:
            if kv_write_indices.numel() > 0:
                kv_len = kv_write_indices[-1].item() + 1
            else:
                kv_len = 0
            local_mask = local_mask[..., :kv_len]
        # ----------------------------------------------------------------

        # Also ensure entire TPA model is on the correct device
        input_device = input_token_ids.device
        if next(self.model.parameters()).device != input_device:
            self.model = self.model.to(input_device)
            self.sampler = self.sampler.to(input_device)

        hidden_states = self.model(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_caches=kv_caches,
            mask=mask,
            local_mask=local_mask,
        )

        embedder_weight = self.text_token_embedder.weight
        if self.config.quant:
            embedder_weight = embedder_weight * self.text_token_embedder.weight_scaler.unsqueeze(-1)

        next_tokens, logits = self.sampler(
            embedding=embedder_weight,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
        )
        return next_tokens, logits

    def create_attention_mask(self, input_ids, sequence_length):
        """
        Creates a causal attention mask (standard for auto-regressive models).
        """
        batch_size = input_ids.shape[0]
        # Create causal mask
        mask = torch.tril(torch.ones((batch_size, 1, sequence_length, sequence_length),
                                     dtype=torch.bool, device=input_ids.device))
        # Possibly create local mask for sliding window
        if self.config.sliding_window_size is not None:
            local_mask = torch.logical_and(
                mask,
                torch.triu(
                    torch.ones((1, 1, sequence_length, sequence_length), dtype=torch.bool, device=input_ids.device),
                    diagonal=-(self.config.sliding_window_size - 1)
                )
            )
        else:
            local_mask = None
        return mask, local_mask

    def generate(
            self,
            prompts: Union[str, Sequence[str]],
            device: Any = None,
            max_tokens: int = 100,
            temperature: Union[float, None] = 1.0,
            top_p: float = 0.95,
            top_k: int = 64,
    ) -> Union[str, Sequence[str]]:
        # If a single prompt is provided, treat it as a batch of 1.
        if device is None:
            device = next(self.parameters()).device

        if isinstance(prompts, str):
            prompts = [prompts]
            single_prompt = True
        else:
            single_prompt = False

        batch_size = len(prompts)
        prompt_tokens = [self.tokenizer.encode(prompt) for prompt in prompts]
        min_prompt_len = min(len(p) for p in prompt_tokens)
        max_prompt_len = max(len(p) for p in prompt_tokens)
        total_seq_len = max_prompt_len + max_tokens
        if total_seq_len > self.config.max_position_embeddings:
            total_seq_len = self.config.max_position_embeddings
            max_tokens = total_seq_len - max_prompt_len

        boolean_mask, local_boolean_mask = self.create_attention_mask(
            torch.zeros((batch_size, total_seq_len), dtype=torch.long, device=device),
            total_seq_len
        )
        min_dtype = torch.finfo(self.dtype).min
        mask_tensor = torch.where(boolean_mask, 0, torch.tensor(min_dtype, dtype=torch.float32, device=device))
        if local_boolean_mask is not None:
            local_mask_tensor = torch.where(local_boolean_mask, 0, torch.tensor(min_dtype, dtype=torch.float32, device=device))
        else:
            local_mask_tensor = None

        kv_caches = []
        for _ in range(self.config.num_hidden_layers):
            size = (batch_size, total_seq_len, self.config.num_attention_heads, self.head_dim)
            dtype = self.config.get_dtype()
            k_cache = torch.zeros(size=size, dtype=dtype, device=device)
            v_cache = torch.zeros(size=size, dtype=dtype, device=device)
            kv_caches.append((k_cache, v_cache))

        input_token_ids_tensor = torch.full((batch_size, min_prompt_len), self.tokenizer.pad_id,
                                            dtype=torch.int64, device=device)
        token_ids_tensor = torch.full((batch_size, total_seq_len), self.tokenizer.pad_id,
                                      dtype=torch.int64, device=device)

        for i, p in enumerate(prompt_tokens):
            token_ids_tensor[i, :len(p)] = torch.tensor(p, dtype=torch.long, device=device)
            input_token_ids_tensor[i, :min_prompt_len] = token_ids_tensor[i, :min_prompt_len]

        input_positions_tensor = torch.arange(0, min_prompt_len, dtype=torch.int64, device=device)
        prompt_mask_tensor = token_ids_tensor != self.tokenizer.pad_id
        curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
        curr_local_mask_tensor = local_mask_tensor.index_select(2, input_positions_tensor) if local_mask_tensor is not None else None
        output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(device)
        temperatures_tensor = None if temperature is None else torch.FloatTensor([temperature] * batch_size).to(device)
        top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
        top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)
        output_index = torch.tensor(min_prompt_len, dtype=torch.int64, device=device)

        for i in range(max_tokens):
            next_token_ids, _ = self(
                input_token_ids=input_token_ids_tensor,
                input_positions=input_positions_tensor,
                kv_caches=kv_caches,
                mask=curr_mask_tensor,
                output_positions=output_positions_tensor,
                temperatures=temperatures_tensor,
                top_ps=top_ps_tensor,
                top_ks=top_ks_tensor,
                local_mask=curr_local_mask_tensor,
            )

            curr_prompt_mask = prompt_mask_tensor.index_select(1, output_index).squeeze(dim=1)
            curr_token_ids = token_ids_tensor.index_select(1, output_index).squeeze(dim=1)
            output_token_ids = torch.where(curr_prompt_mask, curr_token_ids, next_token_ids).unsqueeze(dim=1)
            token_ids_tensor.index_copy_(1, output_index, output_token_ids)

            if (output_token_ids == self.tokenizer.eos_id).all():
                break

            input_token_ids_tensor = output_token_ids
            input_positions_tensor = output_index.unsqueeze(dim=-1)
            curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
            curr_local_mask_tensor = local_mask_tensor.index_select(2, input_positions_tensor) if local_mask_tensor is not None else None
            output_positions_tensor = torch.tensor(0, dtype=torch.int64, device=device)
            output_index = output_index + 1

        token_ids = token_ids_tensor.tolist()
        results = []
        for i, tokens in enumerate(token_ids):
            prompt_len = len(prompt_tokens[i])
            output = tokens[prompt_len:prompt_len + max_tokens]
            if self.tokenizer.eos_id in output:
                eos_index = output.index(self.tokenizer.eos_id)
                output = output[:eos_index]
            results.append(self.tokenizer.decode(output))

        return results[0] if single_prompt else results
