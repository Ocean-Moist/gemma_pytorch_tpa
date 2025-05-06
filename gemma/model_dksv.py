"""Inference-only Gemma model implementation with DK-SVD support."""

import json
import gc
import os
import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, List, Optional, Sequence, Tuple, Union, Mapping

from gemma import config as gemma_config
from gemma import tokenizer


class Sampler(nn.Module):

    def __init__(self, vocab_size: int, config: gemma_config.GemmaConfig):
        super().__init__()
        self.vocab_size = vocab_size
        self.config = config

    @torch.no_grad()
    def forward(
            self,
            embedding: torch.Tensor,
            hidden_states: torch.Tensor,
            output_positions: torch.Tensor,
            temperatures: Union[torch.Tensor, None],
            top_ps: torch.Tensor,
            top_ks: torch.Tensor,
            embedding_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Select the last element for each sequence.
        # (batch_size, input_len, hidden_size) -> (batch_size, hidden_size)
        hidden_states = hidden_states.index_select(
            1, output_positions).squeeze(dim=1)
        logits = torch.matmul(hidden_states, embedding.t())
        if embedding_bias is not None:
            logits += embedding_bias
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        if temperatures is None:
            return torch.argmax(logits, dim=-1).squeeze(dim=-1), logits

        # Apply temperature scaling.
        logits.div_(temperatures.unsqueeze(dim=1))

        # Calculate probabilities with softmax.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

        # Apply top-p, top-k.
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        top_ps_mask = (probs_sum - probs_sort) > top_ps.unsqueeze(dim=1)
        probs_sort = torch.where(top_ps_mask, 0, probs_sort)

        top_ks_mask = torch.arange(probs_idx.shape[-1],
                                   device=probs_idx.device)
        top_ks_mask = top_ks_mask.expand(probs_idx.shape[0], -1)
        top_ks_mask = top_ks_mask >= top_ks.unsqueeze(dim=1)
        probs_sort = torch.where(top_ks_mask, 0, probs_sort)

        # Re-normalization.
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        probs = torch.gather(probs_sort,
                             dim=-1,
                             index=torch.argsort(probs_idx, dim=-1))

        next_token_ids = torch.multinomial(probs,
                                           num_samples=1,
                                           replacement=True).squeeze(dim=-1)
        return next_token_ids, logits


def precompute_freqs_cis(dim: int,
                         end: int,
                         theta: float = 10000.0,
                         rope_scaling_factor:int = 1) -> torch.Tensor:
    """Precomputes the frequency cis."""
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    freqs = freqs/rope_scaling_factor
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies the rotary embedding to the query and key tensors."""
    # x shape: (batch_size, seq_len, num_heads, head_dim)
    # freqs_cis shape: (seq_len, head_dim/2)
    x_ = torch.view_as_complex(
        torch.stack(torch.chunk(x.transpose(1, 2).float(), 2, dim=-1),
                    dim=-1))
    # x_ shape: (batch_size, num_heads, seq_len, head_dim/2)
    # freqs_cis needs to be broadcastable to x_
    # freqs_cis: (seq_len, head_dim/2) -> (1, 1, seq_len, head_dim/2)
    freqs_cis_reshaped = freqs_cis.unsqueeze(0).unsqueeze(0)

    x_out = torch.view_as_real(x_ * freqs_cis_reshaped).type_as(x) # Ensure freqs_cis matches x_ device & dtype if complex
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2],
                          -1).transpose(1, 2)
    # x_out shape: (batch_size, seq_len, num_heads, head_dim)
    return x_out


class Linear(nn.Module):

    def __init__(self, in_features: int, out_features: int, quant: bool):
        super().__init__()
        if quant:
            self.weight = nn.Parameter(
                torch.empty((out_features, in_features), dtype=torch.int8),
                requires_grad=False,
            )
            self.weight_scaler = nn.Parameter(torch.Tensor(out_features))
        else:
            self.weight = nn.Parameter(
                torch.empty((out_features, in_features)),
                requires_grad=False,
            )
        self.quant = quant

    def forward(self, x):
        weight = self.weight
        if self.quant:
            weight = weight * self.weight_scaler.unsqueeze(-1)
        output = F.linear(x, weight)
        return output


class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, quant: bool):
        super().__init__()
        if quant:
            self.weight = nn.Parameter(
                torch.empty((num_embeddings, embedding_dim), dtype=torch.int8),
                requires_grad=False,
            )
            self.weight_scaler = nn.Parameter(torch.Tensor(num_embeddings))
        else:
            self.weight = nn.Parameter(
                torch.empty((num_embeddings, embedding_dim)),
                requires_grad=False,
            )
        self.quant = quant

    def forward(self, x):
        weight = self.weight
        if self.quant:
            weight = weight * self.weight_scaler.unsqueeze(-1)
        output = F.embedding(x, weight)
        return output


class RMSNorm(torch.nn.Module):

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


class GemmaMLP(nn.Module):

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            quant: bool,
    ):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, quant)
        self.up_proj = Linear(hidden_size, intermediate_size, quant)
        self.down_proj = Linear(intermediate_size, hidden_size, quant)

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = F.gelu(gate, approximate="tanh")
        up = self.up_proj(x)
        fuse = gate * up
        outputs = self.down_proj(fuse)
        return outputs


class GemmaAttention(nn.Module):

    def __init__(
            self,
            config: gemma_config.GemmaConfig,
            attn_type: gemma_config.AttentionType,
    ):
        super().__init__()
        self.config = config # Store config for DK-SVD checks
        self.attn_type = attn_type
        self.sliding_window_size = config.sliding_window_size
        self.attn_logit_softcapping = config.attn_logit_softcapping

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim # d_k and d_v original

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.use_dksvd = getattr(config, 'use_dksvd', False)
        self.dksvd_rank = getattr(config, 'dksvd_rank', None)

        if self.use_dksvd:
            if self.dksvd_rank is None or self.dksvd_rank <=0:
                raise ValueError("dksvd_rank must be a positive integer when use_dksvd is True.")

            # DK-SVD projections
            # For each GQA group, Q and K project to r_g (dksvd_rank)
            # Total output dimension for Q_dksvd is num_kv_heads * dksvd_rank
            self.q_proj_dksvd = Linear(
                self.hidden_size,
                self.num_kv_heads * self.dksvd_rank,
                quant=config.quant
            )
            self.k_proj_dksvd = Linear(
                self.hidden_size,
                self.num_kv_heads * self.dksvd_rank,
                quant=config.quant
            )
            # V projection remains similar, outputting num_kv_heads * head_dim (original head_dim for V)
            self.v_proj_dksvd = Linear(
                self.hidden_size,
                self.num_kv_heads * self.head_dim, # V uses original head_dim
                quant=config.quant
            )
            # Output projection takes concatenated outputs from KV groups.
            # Each group's output is head_dim (original V head_dim).
            self.o_proj_dksvd = Linear(
                self.num_kv_heads * self.head_dim,
                self.hidden_size,
                quant=config.quant
            )

            self.scaling = self.dksvd_rank**-0.5 # Scale by r_g for DK-SVD
            if config.use_qk_norm:
                self.query_norm = RMSNorm(self.dksvd_rank, eps=config.rms_norm_eps)
                self.key_norm = RMSNorm(self.dksvd_rank, eps=config.rms_norm_eps)
            else:
                self.query_norm = None
                self.key_norm = None

        else: # Original GQA path
            self.q_size = self.num_heads * self.head_dim
            self.kv_size = self.num_kv_heads * self.head_dim

            if config.query_pre_attn_scalar is not None:
                self.scaling = config.query_pre_attn_scalar**-0.5
            else:
                self.scaling = self.head_dim**-0.5

            self.qkv_proj = Linear(
                self.hidden_size,
                (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
                quant=config.quant)
            self.o_proj = Linear(
                self.num_heads * self.head_dim, self.hidden_size, quant=config.quant
            )
            if config.use_qk_norm:
                self.query_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
                self.key_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            else:
                self.query_norm = None
                self.key_norm = None


    def forward(
            self,
            hidden_states: torch.Tensor,
            freqs_cis: torch.Tensor, # RoPE for appropriate dimension (head_dim or dksvd_rank)
            kv_write_indices: torch.Tensor,
            kv_cache: Tuple[torch.Tensor, torch.Tensor], # K cache dim might change with DK-SVD
            mask: torch.Tensor,
            local_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        hidden_states_shape = hidden_states.shape
        assert len(hidden_states_shape) == 3
        batch_size, input_len, _ = hidden_states_shape

        k_cache, v_cache = kv_cache

        if self.use_dksvd:
            # DK-SVD Path
            # Projections
            # xq_proj: (batch_size, input_len, num_kv_heads * dksvd_rank)
            xq_proj = self.q_proj_dksvd(hidden_states)
            # xk_proj: (batch_size, input_len, num_kv_heads * dksvd_rank)
            xk_proj = self.k_proj_dksvd(hidden_states)
            # xv_proj: (batch_size, input_len, num_kv_heads * head_dim) (V uses original head_dim)
            xv_proj = self.v_proj_dksvd(hidden_states)

            # Reshape for RoPE and per-group attention
            # xq: (batch_size, input_len, num_kv_heads, dksvd_rank)
            xq = xq_proj.view(batch_size, input_len, self.num_kv_heads, self.dksvd_rank)
            # xk: (batch_size, input_len, num_kv_heads, dksvd_rank)
            xk = xk_proj.view(batch_size, input_len, self.num_kv_heads, self.dksvd_rank)
            # xv: (batch_size, input_len, num_kv_heads, head_dim)
            xv = xv_proj.view(batch_size, input_len, self.num_kv_heads, self.head_dim)

            if self.query_norm is not None and self.key_norm is not None:
                xq = self.query_norm(xq)
                xk = self.key_norm(xk)

            xq = apply_rotary_emb(xq, freqs_cis=freqs_cis)
            xk = apply_rotary_emb(xk, freqs_cis=freqs_cis)

            # k_cache shape: (batch_size, max_seq_len, num_kv_heads, dksvd_rank)
            # v_cache shape: (batch_size, max_seq_len, num_kv_heads, head_dim)
            k_cache.index_copy_(1, kv_write_indices, xk)
            v_cache.index_copy_(1, kv_write_indices, xv)

            key = k_cache
            value = v_cache
            # No repeat_interleave needed as Q is already shaped for num_kv_heads

            # q: (batch_size, num_kv_heads, input_len, dksvd_rank)
            q_att = xq.transpose(1, 2)
            # k: (batch_size, num_kv_heads, max_seq_len, dksvd_rank)
            k_att = key.transpose(1, 2)
            # v: (batch_size, num_kv_heads, max_seq_len, head_dim)
            v_att = value.transpose(1, 2)

            q_att.mul_(self.scaling)
            # scores: (batch_size, num_kv_heads, input_len, max_seq_len)
            scores = torch.matmul(q_att, k_att.transpose(2, 3))

            if (
                    self.attn_type == gemma_config.AttentionType.LOCAL_SLIDING
                    and self.sliding_window_size is not None
                    and local_mask is not None
            ):
                mask = local_mask

            if self.attn_logit_softcapping is not None:
                scores = scores / self.attn_logit_softcapping
                scores = torch.tanh(scores)
                scores = scores * self.attn_logit_softcapping

            scores = scores + mask # Mask is (1, 1, input_len, max_seq_len) or similar broadcastable
            scores = F.softmax(scores.float(), dim=-1).type_as(q_att)

            # output_att: (batch_size, num_kv_heads, input_len, head_dim)
            output_att = torch.matmul(scores, v_att)
            # output: (batch_size, input_len, num_kv_heads * head_dim)
            output = (output_att.transpose(1, 2).contiguous().view(
                batch_size, input_len, -1))
            output = self.o_proj_dksvd(output)

        else: # Original GQA Path
            qkv = self.qkv_proj(hidden_states)
            xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size],
                                   dim=-1)

            xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
            xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
            xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)

            if self.query_norm is not None and self.key_norm is not None:
                xq = self.query_norm(xq)
                xk = self.key_norm(xk)

            xq = apply_rotary_emb(xq, freqs_cis=freqs_cis)
            xk = apply_rotary_emb(xk, freqs_cis=freqs_cis)

            k_cache.index_copy_(1, kv_write_indices, xk)
            v_cache.index_copy_(1, kv_write_indices, xv)

            key = k_cache
            value = v_cache
            if self.num_kv_heads != self.num_heads:
                key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=2)
                value = torch.repeat_interleave(value,
                                                self.num_queries_per_kv,
                                                dim=2)

            q = xq.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)

            q.mul_(self.scaling)
            scores = torch.matmul(q, k.transpose(2, 3))
            if (
                    self.attn_type == gemma_config.AttentionType.LOCAL_SLIDING
                    and self.sliding_window_size is not None
                    and local_mask is not None
            ):
                mask = local_mask

            if self.attn_logit_softcapping is not None:
                scores = scores / self.attn_logit_softcapping
                scores = torch.tanh(scores)
                scores = scores * self.attn_logit_softcapping

            scores = scores + mask
            scores = F.softmax(scores.float(), dim=-1).type_as(q)

            output = torch.matmul(scores, v)
            output = (output.transpose(1, 2).contiguous().view(
                batch_size, input_len, -1))
            output = self.o_proj(output)

        return output


class GemmaDecoderLayer(nn.Module): # For Gemma 1 architecture

    def __init__(
            self,
            config: gemma_config.GemmaConfig,
    ):
        super().__init__()
        self.attn_type = gemma_config.AttentionType.GLOBAL # Gemma1 default
        self.self_attn = GemmaAttention(
            config=config,
            attn_type=self.attn_type)
        self.mlp = GemmaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant=config.quant,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            freqs_cis: torch.Tensor,
            kv_write_indices: torch.Tensor,
            kv_cache: Tuple[torch.Tensor, torch.Tensor],
            mask: torch.Tensor,
            local_mask: torch.Tensor, # local_mask added for consistency
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
            mask=mask,
            local_mask=local_mask, # Pass local_mask
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Gemma2DecoderLayer(nn.Module): # For Gemma 2 and 3 architectures

    def __init__(
            self,
            config: gemma_config.GemmaConfig,
            attn_type: gemma_config.AttentionType,
    ):
        super().__init__()
        self.attn_type = attn_type
        self.self_attn = GemmaAttention( # GemmaAttention will handle DK-SVD internally
            config=config,
            attn_type=self.attn_type,
        )
        self.mlp = GemmaMLP(
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
            if config.use_pre_ffw_norm
            else None
        )
        self.post_feedforward_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_post_ffw_norm
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
        # Self Attention
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
        hidden_states = self.post_attention_layernorm(hidden_states) # This is present in Gemma2
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


class GemmaModel(nn.Module):

    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            if config.architecture == gemma_config.Architecture.GEMMA_1:
                self.layers.append(GemmaDecoderLayer(config))
            elif config.architecture in (
                    gemma_config.Architecture.GEMMA_2,
                    gemma_config.Architecture.GEMMA_3,
            ):
                attn_type = (
                    config.attn_types[i % len(config.attn_types)]
                    if config.attn_types is not None
                    else gemma_config.AttentionType.GLOBAL
                )
                self.layers.append(Gemma2DecoderLayer(config, attn_type))
            else:
                raise ValueError(f'Unknown architecture: {config.architecture}')
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            # freqs_cis now a map from attn_type to tensor, to support different RoPE for DK-SVD
            freqs_cis_map: Mapping[gemma_config.AttentionType, torch.Tensor],
            kv_write_indices: torch.Tensor,
            kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
            mask: torch.Tensor,
            local_mask: torch.Tensor,
    ) -> torch.Tensor:
        for i in range(len(self.layers)):
            layer = self.layers[i]
            # Select the appropriate freqs_cis based on layer's attn_type
            current_freqs_cis = freqs_cis_map.get(layer.attn_type)
            if current_freqs_cis is None:
                # Fallback or error if a specific RoPE table is missing
                default_attn_type_for_rope = gemma_config.AttentionType.GLOBAL # Or some other logic
                current_freqs_cis = freqs_cis_map.get(default_attn_type_for_rope)
                if current_freqs_cis is None and freqs_cis_map: # If any freqs_cis exists, use the first one
                    current_freqs_cis = next(iter(freqs_cis_map.values()))


            hidden_states = layer(
                hidden_states=hidden_states,
                freqs_cis=current_freqs_cis,
                kv_write_indices=kv_write_indices,
                kv_cache=kv_caches[i],
                mask=mask,
                local_mask=local_mask,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class GemmaForCausalLM(nn.Module):

    def __init__(
            self,
            config: gemma_config.GemmaConfig,
    ):
        super().__init__()
        self.config = config
        assert config.hidden_size % config.num_attention_heads == 0

        max_seq_len = config.max_position_embeddings
        # head_dim for RoPE depends on whether DK-SVD is used
        self.use_dksvd = getattr(config, 'use_dksvd', False)
        self.dksvd_rank = getattr(config, 'dksvd_rank', None)

        rope_dim = self.dksvd_rank if self.use_dksvd else config.head_dim
        if self.use_dksvd and self.dksvd_rank is None:
            raise ValueError("dksvd_rank must be set in config if use_dksvd is True.")

        vocab_size = config.vocab_size

        self.tokenizer = tokenizer.Tokenizer(config.tokenizer)
        self.embedder = Embedding(vocab_size, config.hidden_size, config.quant)
        self.model = GemmaModel(config) # GemmaModel will handle DK-SVD layers internally
        self.sampler = Sampler(vocab_size, config)

        # Pre-compute rotary embedding table(s).
        if config.architecture == gemma_config.Architecture.GEMMA_3:
            if config.rope_wave_length is None:
                raise ValueError('rope_wave_length must be provided for Gemma3.')

            rope_lengths = config.rope_wave_length
            defaults = { # Default thetas
                gemma_config.AttentionType.LOCAL_SLIDING: 10_000,
                gemma_config.AttentionType.GLOBAL: 10_000,
            }

            # Suffix for DK-SVD specific RoPE tables
            rope_suffix = "_dksvd" if self.use_dksvd else ""

            for attn_type, base_name_stem in [
                (gemma_config.AttentionType.LOCAL_SLIDING, 'local_freqs_cis'),
                (gemma_config.AttentionType.GLOBAL, 'global_freqs_cis'),
            ]:
                theta = rope_lengths.get(attn_type, defaults[attn_type])
                # Register RoPE tables with appropriate dimension (rope_dim)
                # Their names will be e.g. local_freqs_cis or local_freqs_cis_dksvd
                self._register_freqs_cis(f"{base_name_stem}{rope_suffix}", rope_dim, max_seq_len, theta=theta)
        else: # Gemma 1 architecture (or older Gemma 2)
            rope_suffix = "_dksvd" if self.use_dksvd else ""
            self._register_freqs_cis(f"freqs_cis{rope_suffix}", rope_dim, max_seq_len)


    def _register_freqs_cis(
            self, name: str, head_dim: int, max_seq_len: int, theta: float = 10_000.0
    ):
        # Note: head_dim here is the dimension for RoPE (original head_dim or dksvd_rank)
        self.register_buffer(
            name, precompute_freqs_cis(head_dim, max_seq_len * 2, theta=theta)
        )

    @torch.no_grad()
    def forward(
            self,
            input_token_ids: torch.Tensor,
            input_positions: torch.Tensor,
            # kv_write_indices is effectively input_positions for dense attention
            kv_caches: List[Tuple[torch.Tensor, torch.Tensor]], # K cache shape depends on DK-SVD
            mask: torch.Tensor,
            output_positions: torch.Tensor,
            temperatures: Union[torch.Tensor, None],
            top_ps: torch.Tensor,
            top_ks: torch.Tensor,
            local_mask: torch.Tensor | None = None,
            kv_write_indices: Optional[torch.Tensor] = None, # Added for consistency, Gemma uses input_positions
            **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # kv_write_indices is typically input_positions for this model implementation
        if kv_write_indices is None:
            kv_write_indices = input_positions

        # Select appropriate RoPE tables
        freqs_cis_map = {}
        rope_suffix = "_dksvd" if self.use_dksvd else ""

        if self.config.architecture == gemma_config.Architecture.GEMMA_3:
            local_rope_buffer_name = f"local_freqs_cis{rope_suffix}"
            global_rope_buffer_name = f"global_freqs_cis{rope_suffix}"

            freqs_cis_map[gemma_config.AttentionType.LOCAL_SLIDING] = (
                getattr(self, local_rope_buffer_name).index_select(0, input_positions)
            )
            freqs_cis_map[gemma_config.AttentionType.GLOBAL] = (
                getattr(self, global_rope_buffer_name).index_select(0, input_positions)
            )
        else: # Gemma 1 / older Gemma 2
            # These architectures might only have one type of RoPE table or implicitly use GLOBAL.
            # This assumes a single 'freqs_cis' or 'freqs_cis_dksvd' buffer.
            single_rope_buffer_name = f"freqs_cis{rope_suffix}"
            selected_freqs_cis = getattr(self, single_rope_buffer_name).index_select(0, input_positions)
            freqs_cis_map[gemma_config.AttentionType.LOCAL_SLIDING] = selected_freqs_cis
            freqs_cis_map[gemma_config.AttentionType.GLOBAL] = selected_freqs_cis


        # [batch_size, input_len, hidden_size]
        hidden_states = self.embedder(input_token_ids)
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = hidden_states * normalizer

        hidden_states = self.model(
            hidden_states=hidden_states,
            freqs_cis_map=freqs_cis_map, # Pass the map
            kv_write_indices=kv_write_indices,
            kv_caches=kv_caches,
            mask=mask,
            local_mask=local_mask,
        )
        embedder_weight = self.embedder.weight
        if self.config.quant:
            embedder_weight = (
                    embedder_weight * self.embedder.weight_scaler.unsqueeze(-1))
        next_tokens, logits = self.sampler(
            embedding=embedder_weight,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
        )
        return next_tokens, logits

    def generate(
            self,
            prompts: Union[str, Sequence[str]],
            device: Any,
            output_len: int = 100,
            temperature: Union[float, None] = 1.0,
            top_p: float = 0.95,
            top_k: int = 64,
    ) -> Union[str, Sequence[str]]:
        is_str_prompt = isinstance(prompts, str)
        if is_str_prompt:
            prompts = [prompts]

        batch_size = len(prompts)
        prompt_tokens = [self.tokenizer.encode(prompt) for prompt in prompts]
        min_prompt_len = min(len(p) for p in prompt_tokens)
        max_prompt_len = max(len(p) for p in prompt_tokens)
        max_seq_len = max_prompt_len + output_len
        assert max_seq_len <= self.config.max_position_embeddings

        # build KV caches
        kv_caches = []
        k_head_dim_cache = self.dksvd_rank if self.use_dksvd else self.config.head_dim

        for _ in range(self.config.num_hidden_layers):
            # K cache head dimension changes for DK-SVD
            size_k_cache = (batch_size, max_seq_len, self.config.num_key_value_heads, k_head_dim_cache)
            # V cache head dimension remains original
            size_v_cache = (batch_size, max_seq_len, self.config.num_key_value_heads, self.config.head_dim)

            dtype = self.config.get_dtype()
            k_cache = torch.zeros(size=size_k_cache, dtype=dtype, device=device)
            v_cache = torch.zeros(size=size_v_cache, dtype=dtype, device=device)
            kv_caches.append((k_cache, v_cache))

        # prepare inputs (largely same as original)
        token_ids_tensor = torch.full((batch_size, max_seq_len),
                                      self.tokenizer.pad_id, dtype=torch.int64)
        input_token_ids_tensor = torch.full((batch_size, min_prompt_len),
                                            self.tokenizer.pad_id,
                                            dtype=torch.int64)
        for i, p in enumerate(prompt_tokens):
            token_ids_tensor[i, :len(p)] = torch.tensor(p)
            input_token_ids_tensor[i, :min_prompt_len] = torch.tensor(
                p[:min_prompt_len])
        token_ids_tensor = token_ids_tensor.to(device)
        input_token_ids_tensor = input_token_ids_tensor.to(device)
        prompt_mask_tensor = token_ids_tensor != self.tokenizer.pad_id
        input_positions_tensor = torch.arange(0, min_prompt_len,
                                              dtype=torch.int64).to(device)
        mask_tensor = torch.full((1, 1, max_seq_len, max_seq_len),
                                 -2.3819763e38).to(torch.float)
        mask_tensor = torch.triu(mask_tensor, diagonal=1).to(device)
        local_mask_tensor = mask_tensor + torch.tril(
            torch.full((1, 1, max_seq_len, max_seq_len), -2.3819763e38, device=device),
            diagonal=-self.config.sliding_window_size,
        ) if self.config.sliding_window_size else None # sliding_window_size might not be on config if not Gemma2+

        curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
        curr_local_mask_tensor = local_mask_tensor.index_select(
            2, input_positions_tensor
        ) if local_mask_tensor is not None else None
        output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(device)
        temperatures_tensor = None if not temperature else torch.FloatTensor(
            [temperature] * batch_size).to(device)
        top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
        top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)
        output_index = torch.tensor(min_prompt_len, dtype=torch.int64).to(device)

        # Prefill and decode loop
        for i in range(max_seq_len - min_prompt_len):
            next_token_ids, _ = self(
                input_token_ids=input_token_ids_tensor,
                input_positions=input_positions_tensor,
                # kv_write_indices implicitly input_positions_tensor by GemmaModel.forward
                kv_caches=kv_caches,
                mask=curr_mask_tensor,
                output_positions=output_positions_tensor,
                temperatures=temperatures_tensor,
                top_ps=top_ps_tensor,
                top_ks=top_ks_tensor,
                local_mask=curr_local_mask_tensor,
            )

            curr_prompt_mask = prompt_mask_tensor.index_select(
                1, output_index).squeeze(dim=1)
            curr_token_ids = token_ids_tensor.index_select(
                1, output_index).squeeze(dim=1)
            output_token_ids = torch.where(curr_prompt_mask, curr_token_ids,
                                           next_token_ids).unsqueeze(dim=1)
            token_ids_tensor.index_copy_(1, output_index, output_token_ids)

            input_token_ids_tensor = output_token_ids
            input_positions_tensor = output_index.unsqueeze(dim=-1)
            curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
            curr_local_mask_tensor = local_mask_tensor.index_select(
                2, input_positions_tensor
            ) if local_mask_tensor is not None else None
            output_positions_tensor = torch.tensor(0, dtype=torch.int64).to(device)
            output_index = output_index + 1

        # Detokenization.
        token_ids = token_ids_tensor.tolist()
        results = []
        for i, tokens in enumerate(token_ids):
            trimmed_output = tokens[len(prompt_tokens[i]):len(prompt_tokens[i])
                                                          + output_len]
            if self.tokenizer.eos_id in trimmed_output:
                eos_index = trimmed_output.index(self.tokenizer.eos_id)
                trimmed_output = trimmed_output[:eos_index]
            results.append(self.tokenizer.decode(trimmed_output))

        return results[0] if is_str_prompt else results

    def load_weights(self, model_path: str):
        # This load_weights method should be robust enough if the DK-SVD checkpoint
        # uses the new layer names (e.g., q_proj_dksvd) and the model is initialized
        # with config.use_dksvd=True (so it creates those layers).
        # strict=False might be necessary if there are mismatches or if non-DK-SVD
        # layers (like qkv_proj) are absent in the DK-SVD checkpoint.
        # The conversion script should ensure the saved state_dict only contains
        # the necessary weights for the active configuration (DK-SVD or original).

        strict_loading = True # Default to True, can be set to False if issues arise
        # A more robust way is to check config.use_dksvd and adjust expected keys,
        # but typically strict=False handles missing/unexpected keys gracefully if the
        # model structure itself is correct based on the config.

        if os.path.isfile(model_path):
            state_dict_container = torch.load(model_path, mmap=True, weights_only=True)
            # Check if checkpoint is new format with 'config'
            if 'model_state_dict' in state_dict_container and 'config' in state_dict_container:
                loaded_config_dict = state_dict_container['config']
                # Update self.config with loaded DK-SVD flags if they exist
                # This is crucial if the model is instantiated before config is fully known from checkpoint
                if 'use_dksvd' in loaded_config_dict:
                    self.config.use_dksvd = loaded_config_dict['use_dksvd']
                    setattr(self.config, 'use_dksvd', loaded_config_dict['use_dksvd']) # ensure direct attribute
                if 'dksvd_rank' in loaded_config_dict:
                    self.config.dksvd_rank = loaded_config_dict['dksvd_rank']
                    setattr(self.config, 'dksvd_rank', loaded_config_dict['dksvd_rank'])

                # Re-initialize parts of the model if config change requires it
                # This is complex; ideally, config is known at __init__
                # For now, assume __init__ has already set up layers correctly based on initial config.
                # The loaded state_dict should then match.

                self.load_state_dict(state_dict_container['model_state_dict'], strict=strict_loading)

            else: # Old checkpoint format (just state_dict)
                self.load_state_dict(state_dict_container, strict=strict_loading)

        else: # Sharded checkpoint
            index_path = os.path.join(model_path, 'pytorch_model.bin.index.json')
            if not os.path.exists(index_path):
                # Try to load HF SafeTensors model weights index
                index_path = os.path.join(model_path, 'model.safetensors.index.json')

            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)

            # Potentially update config from a config.json in the directory
            config_json_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_json_path):
                with open(config_json_path, "r") as f:
                    loaded_config_json = json.load(f)
                    # Example: if "gemma_config" has dksvd flags, update self.config
                    if loaded_config_json.get("architectures", [""])[0].startswith("Gemma") : # Basic check
                        if "use_dksvd" in loaded_config_json:
                            self.config.use_dksvd = loaded_config_json['use_dksvd']
                            setattr(self.config, 'use_dksvd', loaded_config_json['use_dksvd'])
                        if "dksvd_rank" in loaded_config_json:
                            self.config.dksvd_rank = loaded_config_json['dksvd_rank']
                            setattr(self.config, 'dksvd_rank', loaded_config_json['dksvd_rank'])


            shard_files = list(set(index["weight_map"].values()))
            for shard_file in shard_files:
                shard_path = os.path.join(model_path, shard_file)
                state_dict_shard = {}
                if shard_file.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    state_dict_shard = load_file(shard_path, device="cpu")
                else: # .bin file
                    state_dict_shard = torch.load(shard_path, map_location="cpu", weights_only=True)

                self.load_state_dict(state_dict_shard, strict=False) # strict=False for shards
                del state_dict_shard
                gc.collect()

        # After loading, ensure model is on the correct device (done by runner script)