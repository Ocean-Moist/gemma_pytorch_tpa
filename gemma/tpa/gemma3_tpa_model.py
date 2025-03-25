"""A simpler TPA-based Gemma3 model for causal language modeling (text-only).

This file removes the prior accidental vision/multimodal logic and follows
the same style as the standard `model.py` in Gemma3, but uses Tensor Product
Attention (TPA) instead of standard MHA/GQA.

Classes:
  - RMSNorm
  - TPAAttention
  - Gemma3TPADecoderLayer
  - Gemma3TPAModel
  - Gemma3ForCausalLMwithTPA
"""

from typing import Any, List, Sequence, Tuple, Union, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .. import config as gemma_config
from .. import model as gemma_model
from .. import tokenizer


# We assume the user has a config module similar to gemma_config
# that contains definitions for:
#  - Architecture
#  - AttentionType
#  - GemmaConfig
# or you may import them from gemma.config if needed:
# from .. import config as gemma_config
# For demonstration, we assume gemma_config is accessible
# and can be used similarly to the standard Gemma code.


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

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x might be FP16 or BF16; do norms in float for stability
        output = self._norm(x.float())
        if self.add_unit_offset:
            output = output * (1 + self.weight.float())
        else:
            output = output * self.weight.float()
        return output.type_as(x)


class TPAAttention(nn.Module):
    """Tensor Product Attention module for Gemma 3 (text-only).

    This replaces the standard self-attention block with TPA,
    reducing KV-cache memory usage via factorized K, V.
    """

    def __init__(self, config: gemma_config.GemmaConfig, attn_type: gemma_config.AttentionType):
        super().__init__()
        self.config = config
        self.attn_type = attn_type

        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        # Dimension per head
        self.head_dim = config.head_dim

        # TPA ranks
        self.q_rank = getattr(config, 'q_rank', 6)
        self.k_rank = getattr(config, 'k_rank', 2)
        self.v_rank = getattr(config, 'v_rank', 2)

        # Scaling factor for attention scores
        if config.query_pre_attn_scalar is not None:
            self.scaling = config.query_pre_attn_scalar ** -0.5
        else:
            self.scaling = self.head_dim ** -0.5

        # Projection matrices for TPA factorization
        # W_A: hidden_size -> (num_heads * rank)
        self.W_A_q = nn.Linear(self.hidden_size, self.num_heads * self.q_rank, bias=False)
        self.W_A_k = nn.Linear(self.hidden_size, self.num_heads * self.k_rank, bias=False)
        self.W_A_v = nn.Linear(self.hidden_size, self.num_heads * self.v_rank, bias=False)

        # W_B: hidden_size -> (rank * head_dim)
        self.W_B_q = nn.Linear(self.hidden_size, self.q_rank * self.head_dim, bias=False)
        self.W_B_k = nn.Linear(self.hidden_size, self.k_rank * self.head_dim, bias=False)
        self.W_B_v = nn.Linear(self.hidden_size, self.v_rank * self.head_dim, bias=False)

        # Output projection
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Optional Q/K norm
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

        # Softcapping for attention logits
        self.attn_logit_softcapping = getattr(config, 'attn_logit_softcapping', None)
        # Sliding window for local attention
        self.sliding_window_size = getattr(config, 'sliding_window_size', None)

        # TPA factorization flags
        self.use_factorized_weights = False

        # Initialize KV cache for TPA
        self.cache_kA = None
        self.cache_kB = None
        self.cache_vA = None
        self.cache_vB = None

    def _init_kv_cache(self, batch_size: int, max_seq_len: int):
        """Initialize the KV cache for TPA attention."""
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # A components: [batch_size, max_seq_len, num_heads, rank]
        # B components: [batch_size, max_seq_len, rank, head_dim]
        try:
            self.cache_kA = torch.zeros((batch_size, max_seq_len, self.num_heads, self.k_rank),
                                        device=device, dtype=dtype)
            self.cache_vA = torch.zeros((batch_size, max_seq_len, self.num_heads, self.v_rank),
                                        device=device, dtype=dtype)

            self.cache_kB = torch.zeros((batch_size, max_seq_len, self.k_rank, self.head_dim),
                                        device=device, dtype=dtype)
            self.cache_vB = torch.zeros((batch_size, max_seq_len, self.v_rank, self.head_dim),
                                        device=device, dtype=dtype)

        except RuntimeError as e:
            # If out-of-memory, try smaller fallback dimensions or handle gracefully.
            print(f"Warning: Error creating KV cache: {e}")
            # This is a naive fallback for demonstration.
            # Production code might handle differently.
            fallback_seq_len = max_seq_len // 2 if max_seq_len > 128 else 64
            self.cache_kA = torch.zeros((batch_size, fallback_seq_len, self.num_heads, self.k_rank),
                                        device=device, dtype=dtype)
            self.cache_vA = torch.zeros((batch_size, fallback_seq_len, self.num_heads, self.v_rank),
                                        device=device, dtype=dtype)

            self.cache_kB = torch.zeros((batch_size, fallback_seq_len, self.k_rank, self.head_dim),
                                        device=device, dtype=dtype)
            self.cache_vB = torch.zeros((batch_size, fallback_seq_len, self.v_rank, self.head_dim),
                                        device=device, dtype=dtype)

    def forward(
            self,
            hidden_states: torch.Tensor,  # [batch_size, seq_len, hidden_size]
            freqs_cis: torch.Tensor,
            kv_write_indices: torch.Tensor,
            kv_cache: Tuple[torch.Tensor, torch.Tensor],
            mask: torch.Tensor,
            local_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # A projections
        A_q = self.W_A_q(hidden_states).view(batch_size, seq_len, self.num_heads, self.q_rank)
        A_k = self.W_A_k(hidden_states).view(batch_size, seq_len, self.num_heads, self.k_rank)
        A_v = self.W_A_v(hidden_states).view(batch_size, seq_len, self.num_heads, self.v_rank)

        # B projections
        B_q = self.W_B_q(hidden_states).reshape(batch_size, seq_len, self.q_rank, self.head_dim)
        B_k = self.W_B_k(hidden_states).reshape(batch_size, seq_len, self.k_rank, self.head_dim)
        B_v = self.W_B_v(hidden_states).reshape(batch_size, seq_len, self.v_rank, self.head_dim)

        # Apply RoPE to B_q and B_k
        def apply_rotary_emb_to_B(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
            """TPA-specific RoPE application for B-projections."""
            # x: [batch, seq_len, rank, head_dim]
            bsz, slen, rnk, hdim = x.shape
            if hdim % 2 != 0:
                raise ValueError("Head dimension must be even for RoPE")

            # We'll rotate each rank dimension separately
            # This is not the only approach, but a straightforward one.
            out = torch.zeros_like(x)
            for r in range(rnk):
                # x_r shape: [bsz, slen, head_dim]
                x_r = x[:, :, r, :]

                # standard complex representation for RoPE
                # half the head_dim for real-imag pairs
                x_c = torch.view_as_complex(x_r.float().reshape(*x_r.shape[:-1], -1, 2))
                # multiply by freqs
                x_rot = x_c * freqs  # broadcast: [bsz, slen, hdim//2]
                # convert back
                x_real = torch.view_as_real(x_rot)
                # flatten
                x_out = x_real.reshape(bsz, slen, hdim)
                out[:, :, r, :] = x_out.type_as(x)
            return out

        B_q = apply_rotary_emb_to_B(B_q, freqs_cis)
        B_k = apply_rotary_emb_to_B(B_k, freqs_cis)

        # Initialize KV cache if needed
        if self.cache_kA is None or batch_size > self.cache_kA.shape[0]:
            self._init_kv_cache(batch_size, kv_cache[0].shape[1])

        # Move to the correct device if not matching
        device_match = A_k.device
        self.cache_kA = self.cache_kA.to(device_match)
        self.cache_vA = self.cache_vA.to(device_match)
        self.cache_kB = self.cache_kB.to(device_match)
        self.cache_vB = self.cache_vB.to(device_match)

        # Index copy the new tokens into the KV cache
        self.cache_kA[:batch_size].index_copy_(1, kv_write_indices, A_k)
        self.cache_vA[:batch_size].index_copy_(1, kv_write_indices, A_v)
        self.cache_kB[:batch_size].index_copy_(1, kv_write_indices, B_k)
        self.cache_vB[:batch_size].index_copy_(1, kv_write_indices, B_v)

        # Now slice out the relevant portion up to the current step
        cache_len = int(kv_write_indices[-1]) + 1 if kv_write_indices.numel() > 0 else 0
        A_k = self.cache_kA[:batch_size, :cache_len]
        A_v = self.cache_vA[:batch_size, :cache_len]
        B_k = self.cache_kB[:batch_size, :cache_len]
        B_v = self.cache_vB[:batch_size, :cache_len]

        # Factorized Q, K, V
        # Q: [bsz, seq_len, num_heads, head_dim] by TPA
        #    Q = (A_q x B_q) / q_rank
        # K: [bsz, cache_len, num_heads, head_dim]
        # V: [bsz, cache_len, num_heads, head_dim]

        # Flatten batch*seq for matrix multiplications
        A_q_flat = A_q.reshape(batch_size * seq_len, self.num_heads, self.q_rank)
        B_q_flat = B_q.reshape(batch_size * seq_len, self.q_rank, self.head_dim)
        q = torch.bmm(A_q_flat, B_q_flat).div(self.q_rank)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)

        A_k_flat = A_k.reshape(batch_size * cache_len, self.num_heads, self.k_rank)
        B_k_flat = B_k.reshape(batch_size * cache_len, self.k_rank, self.head_dim)
        k = torch.bmm(A_k_flat, B_k_flat).div(self.k_rank)
        k = k.view(batch_size, cache_len, self.num_heads, self.head_dim)

        A_v_flat = A_v.reshape(batch_size * cache_len, self.num_heads, self.v_rank)
        B_v_flat = B_v.reshape(batch_size * cache_len, self.v_rank, self.head_dim)
        v = torch.bmm(A_v_flat, B_v_flat).div(self.v_rank)
        v = v.view(batch_size, cache_len, self.num_heads, self.head_dim)

        # Optional Q/K norm
        if self.query_norm is not None and self.key_norm is not None:
            q = self.query_norm(q)
            k = self.key_norm(k)

        # Permute to [batch_size, num_heads, seq_len, head_dim] for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply scaling
        q = q * self.scaling

        # attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        # Softcapping if configured
        if self.attn_logit_softcapping is not None:
            attn_weights = attn_weights / self.attn_logit_softcapping
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * self.attn_logit_softcapping

        # Local sliding window if needed
        if self.attn_type == gemma_config.AttentionType.LOCAL_SLIDING and self.sliding_window_size is not None:
            if local_mask is not None:
                mask = local_mask

        # Add the mask
        attn_weights = attn_weights + mask

        # softmax
        attn_probs = F.softmax(attn_weights.float(), dim=-1).type_as(q)

        # apply to v
        attn_output = torch.matmul(attn_probs, v)
        # [batch, num_heads, seq, head_dim] -> [batch, seq, num_heads * head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        # final linear
        output = self.o_proj(attn_output)
        return output


class Gemma3TPADecoderLayer(nn.Module):
    """A single decoder layer (Gemma3 style) with TPA-based self-attention."""

    def __init__(
            self,
            config: gemma_config.GemmaConfig,
            attn_type: gemma_config.AttentionType,
    ):
        super().__init__()
        self.attn_type = attn_type
        self.self_attn = TPAAttention(config=config, attn_type=self.attn_type)
        self.mlp = gemma_model.GemmaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant=config.quant,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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
        # Self-attention
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

        # MLP
        residual = hidden_states
        if self.pre_feedforward_layernorm is not None:
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.post_feedforward_layernorm is not None:
            hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Gemma3TPAModel(nn.Module):
    """Backbone Gemma3 model with TPA-based decoder layers (text-only)."""

    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.layers = nn.ModuleList()

        # Construct each decoder layer
        for i in range(config.num_hidden_layers):
            if config.architecture in (gemma_config.Architecture.GEMMA_2, gemma_config.Architecture.GEMMA_3):
                attn_type = (
                    config.attn_types[i % len(config.attn_types)]
                    if config.attn_types is not None
                    else gemma_config.AttentionType.GLOBAL
                )
                self.layers.append(Gemma3TPADecoderLayer(config, attn_type))
            else:
                raise ValueError(f"Unsupported architecture: {config.architecture}")

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


class Gemma3ForCausalLMwithTPA(nn.Module):
    """A Gemma3 model for causal language modeling using TPA self-attention.

    This version follows the standard text-only model.py style,
    removing any multimodal/vision references.
    """

    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.dtype = config.get_dtype()
        assert config.architecture == gemma_config.Architecture.GEMMA_3, (
            "Gemma3ForCausalLMwithTPA requires architecture=GEMMA_3"
        )
        self.config = config

        max_seq_len = config.max_position_embeddings
        head_dim = config.head_dim
        vocab_size = config.vocab_size

        # Tokenizer reference (if needed externally)
        self.tokenizer = tokenizer.Tokenizer(config.tokenizer)
        self.head_dim = config.head_dim

        # Text embedder
        self.text_token_embedder = gemma_model.Embedding(vocab_size, config.hidden_size, config.quant)

        # TPA-based backbone
        self.model = Gemma3TPAModel(config)

        # Sampler (logits to next token)
        self.sampler = gemma_model.Sampler(vocab_size, config)

        # RoPE setup
        if config.rope_wave_length is None:
            raise ValueError("rope_wave_length must be provided for Gemma3 with TPA.")
        rope_lengths = config.rope_wave_length
        defaults = {
            gemma_config.AttentionType.LOCAL_SLIDING: 10_000,
            gemma_config.AttentionType.GLOBAL: 10_000,
        }
        # register buffers for local/global freq
        self._register_freqs_cis(
            'local_freqs_cis',
            head_dim,
            max_seq_len,
            theta=rope_lengths.get(
                gemma_config.AttentionType.LOCAL_SLIDING, defaults[gemma_config.AttentionType.LOCAL_SLIDING]
            ),
        )
        self._register_freqs_cis(
            'global_freqs_cis',
            head_dim,
            max_seq_len,
            theta=rope_lengths.get(
                gemma_config.AttentionType.GLOBAL, defaults[gemma_config.AttentionType.GLOBAL]
            ),
            rope_scaling_factor=config.rope_scaling_factor,
        )

    def _register_freqs_cis(
            self,
            name: str,
            head_dim: int,
            max_seq_len: int,
            theta: int = 10_000,
            rope_scaling_factor: int = 1
    ):
        self.register_buffer(
            name,
            gemma_model.precompute_freqs_cis(
                head_dim, max_seq_len * 2, theta=theta, rope_scaling_factor=rope_scaling_factor
            )
        )

    @torch.no_grad()
    def forward(
            self,
            input_token_ids: torch.Tensor,  # [B, L]
            input_positions: torch.Tensor,  # [L] or [B, L]
            kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],  # for each layer
            mask: torch.Tensor,
            output_positions: torch.Tensor,
            temperatures: Union[torch.Tensor, None],
            top_ps: torch.Tensor,
            top_ks: torch.Tensor,
            local_mask: torch.Tensor | None = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for single-step or batched inference.

        Args:
            input_token_ids: [batch_size, seq_len]
            input_positions: The positions (indices) in the sequence dimension
            kv_caches: key-value caches for each layer
            mask: the full attention mask
            output_positions: positions for which we compute next-token sampling
            temperatures, top_ps, top_ks: sampling parameters
            local_mask: optional local sliding window mask
            **kwargs: for future expansions
        """
        # separate local & global freq_cis
        freqs_cis = {
            gemma_config.AttentionType.LOCAL_SLIDING: self.local_freqs_cis.index_select(0, input_positions),
            gemma_config.AttentionType.GLOBAL: self.global_freqs_cis.index_select(0, input_positions),
        }

        # Embed tokens
        hidden_states = self.text_token_embedder(input_token_ids)
        # scale by sqrt(hidden_size)
        normalizer = (self.config.hidden_size ** 0.5)
        hidden_states = hidden_states * normalizer

        # call TPA model
        hidden_states = self.model(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=input_positions,
            kv_caches=kv_caches,
            mask=mask,
            local_mask=local_mask,
        )

        # Sampler for next tokens
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

    def create_attention_mask(self, input_ids: torch.Tensor, sequence_length: int):
        """Create the standard causal attention mask, with optional local sliding."""
        batch_size = input_ids.shape[0]
        causal_mask = torch.tril(
            torch.ones((batch_size, 1, sequence_length, sequence_length),
                       dtype=torch.bool,
                       device=input_ids.device)
        )

        # (If the user had local sliding logic, we might replicate that here.)
        # For now, we just return the standard causal mask plus an optional local_mask
        local_mask = None
        return causal_mask, local_mask

    def generate(
            self,
            prompts: Sequence[str],
            device: Any = None,
            max_tokens: int = 100,
            temperature: Union[float, None] = 1.0,
            top_p: float = 0.95,
            top_k: int = 64,
    ) -> Sequence[str]:
        """Generate text for given string prompts, using TPA-based Gemma3 model."""
        # If a single string is given, wrap it in a list
        if isinstance(prompts, str):
            prompts = [prompts]

        if device is None:
            device = next(self.parameters()).device

        # Prepare input tokens via tokenizer
        tokenized = [self.tokenizer.encode(p) for p in prompts]
        batch_size = len(tokenized)
        max_prompt_len = max(len(t) for t in tokenized)
        total_seq_len = max_prompt_len + max_tokens

        # Build KV caches
        kv_caches = []
        for _ in range(self.config.num_hidden_layers):
            # TPA cache is smaller if we do factorization
            # we store shape [batch_size, total_seq_len, ...], but the code uses GQA style
            # We'll store [batch_size, total_seq_len, self.num_attention_heads, self.head_dim].
            # TPA will handle factorization internally.
            dtype = self.config.get_dtype()
            k_cache = torch.zeros((batch_size, total_seq_len, self.config.num_attention_heads, self.head_dim),
                                  dtype=dtype, device=device)
            v_cache = torch.zeros_like(k_cache)
            kv_caches.append((k_cache, v_cache))

        # Prepare input token tensor
        input_ids_tensor = torch.full((batch_size, total_seq_len), self.tokenizer.pad_id,
                                      dtype=torch.int64, device=device)
        # Fill in the prompts
        for i, tokens in enumerate(tokenized):
            input_ids_tensor[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.int64, device=device)

        # We do standard causal mask
        mask_tensor = torch.full((1, 1, total_seq_len, total_seq_len),
                                 -1e10, dtype=torch.float, device=device)
        mask_tensor = torch.triu(mask_tensor, diagonal=1)

        # For local sliding if used, we'd build local_mask. We'll skip that here.

        # We'll do an auto-regressive loop
        curr_idx = max_prompt_len
        results = [None] * batch_size

        # We'll define sampling config
        if temperature is not None:
            temperature_tensor = torch.tensor([temperature] * batch_size, dtype=torch.float, device=device)
        else:
            temperature_tensor = None

        top_ps = torch.tensor([top_p] * batch_size, dtype=torch.float, device=device)
        top_ks = torch.tensor([top_k] * batch_size, dtype=torch.long, device=device)

        # We'll do simple loop up to max_tokens
        for _ in range(max_tokens):
            # Build a small input_positions for the new token
            positions_tensor = torch.arange(curr_idx, dtype=torch.int64, device=device)

            output_positions_tensor = torch.tensor([curr_idx - 1], dtype=torch.int64, device=device)

            # Forward pass
            next_tokens, _ = self(
                input_token_ids=input_ids_tensor[:, :curr_idx],  # shape [B, curr_idx]
                input_positions=positions_tensor,
                kv_caches=kv_caches,
                mask=mask_tensor[:, :, :curr_idx, :curr_idx],
                output_positions=output_positions_tensor,
                temperatures=temperature_tensor,
                top_ps=top_ps,
                top_ks=top_ks,
            )

            # Place next token
            input_ids_tensor[:, curr_idx] = next_tokens

            # Check for any EOS
            # We'll do it naive: if any next_tokens == eos_id, we can finalize
            # but let's just keep it simple
            curr_idx += 1
            if curr_idx >= total_seq_len:
                break

        # Now decode each sample
        all_tokens = input_ids_tensor.tolist()
        for i, seq in enumerate(all_tokens):
            # cut at the first pad or up to total_seq_len
            # or until eos
            out_seq = seq[:]
            if self.tokenizer.eos_id in out_seq:
                eos_pos = out_seq.index(self.tokenizer.eos_id)
                out_seq = out_seq[:eos_pos]
            # also remove leading pad if any
            if self.tokenizer.pad_id in out_seq:
                # This is naive. If there's a pad in the middle, etc.
                pad_pos = out_seq.index(self.tokenizer.pad_id)
                out_seq = out_seq[:pad_pos]
            # cut off the prompt
            prompt_len = len(tokenized[i])
            generated_seq = out_seq[prompt_len:]
            text = self.tokenizer.decode(generated_seq)
            results[i] = text

        if len(results) == 1:
            return results[0]
        return results

    def load_weights(self, model_path: str):
        """Load TPA model weights from checkpoint (similar to standard Gemma)."""
        import os
        import gc

        if os.path.isfile(model_path):
            state_dict = torch.load(model_path, map_location="cpu")
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            self.load_state_dict(state_dict, strict=False)
        else:
            index_path = os.path.join(model_path, "pytorch_model.bin.index.json")
            import json
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            shard_files = list(set(index["weight_map"].values()))
            for shard_file in shard_files:
                shard_path = os.path.join(model_path, shard_file)
                sd = torch.load(shard_path, map_location="cpu")
                if "model_state_dict" in sd:
                    sd = sd["model_state_dict"]
                self.load_state_dict(sd, strict=False)
                del sd
                gc.collect()
