# Energy‑DK‑SVD compressed Gemma model
# ---------------------------------------------------------------
# This file re‑implements the attention blocks of Gemma to consume
# the low‑rank Q/K factors produced by convert_weights.py.  Only the
# parts that differ from the original Gemma implementation are
# rewritten; everything else is imported from gemma.model so that the
# public API remains identical (forward / generate work the same).
#
# The key idea: each Grouped‑Query‑Attention (GQA) group keeps its own
# *r_g*-dimensional shared key projection plus *s* query projections of
# the same width.  Values remain at the original head dimension
# (config.head_dim).  During runtime we expand the shared K/V across
# the query heads with repeat_interleave so the rest of the code can
# stay unchanged.
# ---------------------------------------------------------------
from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from gemma import config as gemma_config
from gemma import tokenizer

# Re‑use utility layers from the reference implementation.
from gemma.model import (
    Linear,
    Embedding,
    RMSNorm,
    precompute_freqs_cis,
    apply_rotary_emb, GemmaMLP, Sampler,
)

# ------------------------------------------------------------------
# Attention block working with E‑DK‑SVD weights
# ------------------------------------------------------------------
class GemmaAttentionDKSVD(nn.Module):
    """Self‑attention with low‑rank Q/K projections produced by E‑DK‑SVD."""

    def __init__(self, config: gemma_config.GemmaConfig, attn_type: gemma_config.AttentionType):
        super().__init__()

        if not hasattr(config, "qk_rank") or config.qk_rank is None:
            raise ValueError("GemmaConfig must define qk_rank when using the DKSVD variant.")

        # Core sizes -----------------------------------------------------------------
        self.num_heads = config.num_attention_heads          # total query heads, N_h
        self.num_kv_heads = config.num_key_value_heads       # GQA groups, N_kv
        assert self.num_heads % self.num_kv_heads == 0, "N_h must be a multiple of N_kv"
        self.num_q_per_kv = self.num_heads // self.num_kv_heads  # s

        self.hidden_size = config.hidden_size                # model width, d
        self.qk_rank = config.qk_rank                        # compressed rank, r_g
        self.v_head_dim = config.head_dim                    # original value dim, d_v

        # Scaling for dot‑product attention: 1/sqrt(r_g)
        self.scaling = self.qk_rank ** -0.5 if config.query_pre_attn_scalar is None else config.query_pre_attn_scalar ** -0.5

        # ------------------------------------------------------------------
        # Per‑group projection matrices.
        # Names are chosen so that convert_weights.py can store the factors
        # exactly where we expect them (e.g. ``q_linears.0.3.weight``).
        # ------------------------------------------------------------------
        # Each KV head owns one *shared* K and V projection.
        self.k_linears = nn.ModuleList([
            Linear(self.hidden_size, self.qk_rank, quant=config.quant) for _ in range(self.num_kv_heads)
        ])
        self.v_linears = nn.ModuleList([
            Linear(self.hidden_size, self.v_head_dim, quant=config.quant) for _ in range(self.num_kv_heads)
        ])

        # Each KV head also owns *num_q_per_kv* query projections of width r_g.
        self.q_linears = nn.ModuleList([
            nn.ModuleList([
                Linear(self.hidden_size, self.qk_rank, quant=config.quant) for _ in range(self.num_q_per_kv)
            ]) for _ in range(self.num_kv_heads)
        ])

        # Output projection (unchanged from the reference)
        self.o_proj = Linear(self.num_heads * self.v_head_dim, self.hidden_size, quant=config.quant)

        # Norms – adapted to r_g rather than d_k
        self.query_norm = RMSNorm(self.qk_rank, eps=config.rms_norm_eps) if config.use_qk_norm else None
        self.key_norm = RMSNorm(self.qk_rank, eps=config.rms_norm_eps) if config.use_qk_norm else None

        # Misc options from config ----------------------------------------------------
        self.attn_type = attn_type
        self.sliding_window_size = config.sliding_window_size
        self.attn_logit_softcapping = config.attn_logit_softcapping

    # ----------------------------------------------------------------------
    # forward
    # ----------------------------------------------------------------------
    def forward(
            self,
            hidden_states: torch.Tensor,                      # (B, T, d)
            freqs_cis: torch.Tensor | None,                   # (T, r_g//2) complex or None
            kv_write_indices: torch.Tensor,                   # (T,) positions being written this step
            kv_cache: Tuple[torch.Tensor, torch.Tensor],      # (k_cache, v_cache)
            mask: torch.Tensor,                               # broadcastable attn mask
            local_mask: torch.Tensor | None = None,           # optional local sliding mask
    ) -> torch.Tensor:                                    # returns (B, T, d)
        batch_size, seq_len, _ = hidden_states.shape

        k_cache, v_cache = kv_cache                       # (B, L, N_kv, r_g) / (B, L, N_kv, d_v)

        # ------------------------------------------------------------------
        # Build projections for this step & update KV cache per group.
        # ------------------------------------------------------------------
        query_list: List[torch.Tensor] = []               # queries for *all* heads, to be stacked
        key_list: List[torch.Tensor] = []                 # keys per KV group (before expansion)
        value_list: List[torch.Tensor] = []               # values per KV group (before expansion)

        for g in range(self.num_kv_heads):
            # Shared KEY ---------------------------------------------------
            k_g = self.k_linears[g](hidden_states)        # (B, T, r_g)
            if self.key_norm is not None:
                k_g = self.key_norm(k_g)
            if freqs_cis is not None:
                k_g = apply_rotary_emb(k_g.view(batch_size, seq_len, 1, self.qk_rank), freqs_cis=freqs_cis).squeeze(2)

            # Shared VALUE -------------------------------------------------
            v_g = self.v_linears[g](hidden_states)        # (B, T, d_v)

            # Write to cache for the current positions
            k_cache[:, kv_write_indices, g, :] = k_g
            v_cache[:, kv_write_indices, g, :] = v_g

            # Gather full sequence from cache up to current max length
            key_list.append(k_cache[:, :k_cache.shape[1], g, :])      # (B, L, r_g)
            value_list.append(v_cache[:, :v_cache.shape[1], g, :])    # (B, L, d_v)

            # Queries for every head in this group ------------------------
            for i in range(self.num_q_per_kv):
                q_gi = self.q_linears[g][i](hidden_states)  # (B, T, r_g)
                if self.query_norm is not None:
                    q_gi = self.query_norm(q_gi)
                if freqs_cis is not None:
                    q_gi = apply_rotary_emb(q_gi.view(batch_size, seq_len, 1, self.qk_rank), freqs_cis=freqs_cis).squeeze(2)
                query_list.append(q_gi)

        # Stack along head dimension ---------------------------------------
        # After the loop we have: len(query_list) == N_h, len(key_list) == N_kv.
        xq = torch.stack(query_list, dim=2)          # (B, T, N_h, r_g)
        k  = torch.stack(key_list,   dim=2)          # (B, L, N_kv, r_g)
        v  = torch.stack(value_list, dim=2)          # (B, L, N_kv, d_v)

        # Expand keys/values so every query head has a matching entry ------
        k = torch.repeat_interleave(k, self.num_q_per_kv, dim=2)       # (B, L, N_h, r_g)
        v = torch.repeat_interleave(v, self.num_q_per_kv, dim=2)       # (B, L, N_h, d_v)

        # Prepare shapes for matmul: (B, N_h, T/L, dim)
        q = xq.transpose(1, 2)                                          # (B, N_h, T, r_g)
        k = k.transpose(1, 2)                                           # (B, N_h, L, r_g)
        v = v.transpose(1, 2)                                           # (B, N_h, L, d_v)

        # Attention scores -------------------------------------------------
        q = q * self.scaling
        attn_scores = torch.matmul(q, k.transpose(-2, -1))              # (B, N_h, T, L)

        # Local sliding window mask if configured
        if (
                self.attn_type == gemma_config.AttentionType.LOCAL_SLIDING
                and self.sliding_window_size is not None
                and local_mask is not None
        ):
            mask = local_mask

        if self.attn_logit_softcapping is not None:
            attn_scores = attn_scores / self.attn_logit_softcapping
            attn_scores = torch.tanh(attn_scores) * self.attn_logit_softcapping

        attn_scores = attn_scores + mask
        attn_probs = F.softmax(attn_scores.float(), dim=-1).type_as(q)

        # MatMul with values ----------------------------------------------
        attn_output = torch.matmul(attn_probs, v)                        # (B, N_h, T, d_v)

        # Reshape back to (B, T, d)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output

# ------------------------------------------------------------------
# Decoder layer wrapper
# ------------------------------------------------------------------
class Gemma2DecoderLayerDKSVD(nn.Module):
    """Gemma‑3 style decoder layer with DK‑SVD attention."""

    def __init__(self, config: gemma_config.GemmaConfig, attn_type: gemma_config.AttentionType):
        super().__init__()
        self.attn_type = attn_type
        self.self_attn = GemmaAttentionDKSVD(config=config, attn_type=attn_type)
        self.mlp = GemmaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant=config.quant,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps) if config.use_pre_ffw_norm else None
        )
        self.post_feedforward_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps) if config.use_post_ffw_norm else None
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            freqs_cis: torch.Tensor,
            kv_write_indices: torch.Tensor,
            kv_cache: Tuple[torch.Tensor, torch.Tensor],
            mask: torch.Tensor,
            local_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # Self‑attention --------------------------------------------------
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

        # Feed‑forward ----------------------------------------------------
        residual = hidden_states
        if self.pre_feedforward_layernorm is not None:
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.post_feedforward_layernorm is not None:
            hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

# ------------------------------------------------------------------
# Stacked decoder + full LM wrapper
# ------------------------------------------------------------------
class GemmaModelDKSVD(nn.Module):
    """Backbone transformer stack composed of DK‑SVD layers."""

    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            attn_type = (
                config.attn_types[i % len(config.attn_types)] if config.attn_types is not None else gemma_config.AttentionType.GLOBAL
            )
            self.layers.append(Gemma2DecoderLayerDKSVD(config, attn_type))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            freqs_cis: Mapping[gemma_config.AttentionType, torch.Tensor],
            kv_write_indices: torch.Tensor,
            kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
            mask: torch.Tensor,
            local_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
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

# ------------------------------------------------------------------
class GemmaForCausalLMDKSVD(nn.Module):
    """Gemma causal‑LM wrapper that consumes E‑DK‑SVD weights."""

    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.config = config
        if not hasattr(config, "qk_rank"):
            raise ValueError("GemmaConfig must include qk_rank for the DKSVD variant.")

        self.vocab_size = config.vocab_size
        self.tokenizer = tokenizer.Tokenizer(config.tokenizer)
        self.embedder = Embedding(config.vocab_size, config.hidden_size, config.quant)
        self.model = GemmaModelDKSVD(config)
        self.sampler = Sampler(config.vocab_size, config)

        # Pre‑compute RoPE tables at width r_g  --------------------------------------
        if config.architecture == gemma_config.Architecture.GEMMA_3:
            if config.rope_wave_length is None:
                raise ValueError("rope_wave_length must be provided for Gemma3.")

            rope_lengths = config.rope_wave_length
            defaults = {
                gemma_config.AttentionType.LOCAL_SLIDING: 10_000,
                gemma_config.AttentionType.GLOBAL: 10_000,
            }
            for attn_type, name in [
                (gemma_config.AttentionType.LOCAL_SLIDING, "local_freqs_cis"),
                (gemma_config.AttentionType.GLOBAL, "global_freqs_cis"),
            ]:
                theta = rope_lengths.get(attn_type, defaults[attn_type])
                self._register_freqs_cis(name, config.qk_rank, config.max_position_embeddings * 2, theta=theta)
        else:
            # GEMMA_1 / GEMMA_2 legacy path – single table
            self._register_freqs_cis("freqs_cis", config.qk_rank, config.max_position_embeddings * 2)

    # Helper -----------------------------------------------------------------
    def _register_freqs_cis(self, name: str, head_dim: int, max_seq_len: int, theta: int = 10_000):
        self.register_buffer(name, precompute_freqs_cis(head_dim, max_seq_len, theta=theta))

    # ----------------------------------------------------------------------
    @torch.no_grad()
    def forward(
            self,
            input_token_ids: torch.Tensor,              # (B, T_in)
            input_positions: torch.Tensor,              # (T_in,)
            kv_write_indices: torch.Tensor,             # (T_in,)
            kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
            mask: torch.Tensor,
            output_positions: torch.Tensor,             # indices of tokens whose logits we return
            temperatures: Optional[torch.Tensor],
            top_ps: torch.Tensor,
            top_ks: torch.Tensor,
            local_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # RoPE lookup tables ------------------------------------------------
        freqs_cis = {}
        if self.config.architecture == gemma_config.Architecture.GEMMA_3:
            freqs_cis[gemma_config.AttentionType.LOCAL_SLIDING] = self.local_freqs_cis.index_select(0, input_positions)
            freqs_cis[gemma_config.AttentionType.GLOBAL] = self.global_freqs_cis.index_select(0, input_positions)
        else:
            shared = self.freqs_cis.index_select(0, input_positions)
            freqs_cis[gemma_config.AttentionType.LOCAL_SLIDING] = shared
            freqs_cis[gemma_config.AttentionType.GLOBAL] = shared

        # Embedding & positional norm ------------------------------------------------
        hidden_states = self.embedder(input_token_ids)
        hidden_states = hidden_states * (self.config.hidden_size ** 0.5)

        hidden_states = self.model(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_caches=kv_caches,
            mask=mask,
            local_mask=local_mask,
        )

        # Weight tying --------------------------------------------------------------
        embedder_weight = self.embedder.weight
        if self.config.quant:
            embedder_weight = embedder_weight * self.embedder.weight_scaler.unsqueeze(-1)

        next_tokens, logits = self.sampler(
            embedding=embedder_weight,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
        )
        return next_tokens, logits

    # ----------------------------------------------------------------------
    # Minimal generate loop.  We simply adapt the cache shapes while re‑using
    # the reference implementation logic for brevity.
    # ----------------------------------------------------------------------
    def generate(
            self,
            prompts: Union[str, Sequence[str]],
            device: Any,
            output_len: int = 100,
            temperature: Union[float, None] = 1.0,
            top_p: float = 0.95,
            top_k: int = 64,
    ) -> Union[str, Sequence[str]]:
        # Borrow the reference Gemma.generate implementation but override the
        # cache dimensions to accommodate r_g.
        is_str_prompt = isinstance(prompts, str)
        if is_str_prompt:
            prompts = [prompts]

        batch_size = len(prompts)
        prompt_tokens = [self.tokenizer.encode(p) for p in prompts]
        min_prompt_len = min(len(p) for p in prompt_tokens)
        max_prompt_len = max(len(p) for p in prompt_tokens)
        max_seq_len = max_prompt_len + output_len
        assert max_seq_len <= self.config.max_position_embeddings

        # Build KV caches with r_g for keys, d_v for values ----------------
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        dtype = self.config.get_dtype()
        for _ in range(self.config.num_hidden_layers):
            k_cache = torch.zeros(
                (batch_size, max_seq_len, self.config.num_key_value_heads, self.config.qk_rank),
                dtype=dtype,
                device=device,
            )
            v_cache = torch.zeros(
                (batch_size, max_seq_len, self.config.num_key_value_heads, self.config.head_dim),
                dtype=dtype,
                device=device,
            )
            kv_caches.append((k_cache, v_cache))

        # Prepare static tensors ----------------------------------------------------
        pad_id = self.tokenizer.pad_id
        token_ids_tensor = torch.full((batch_size, max_seq_len), pad_id, dtype=torch.int64, device=device)
        input_token_ids_tensor = torch.full((batch_size, min_prompt_len), pad_id, dtype=torch.int64, device=device)
        for i, p in enumerate(prompt_tokens):
            token_ids_tensor[i, : len(p)] = torch.tensor(p, device=device)
            input_token_ids_tensor[i, : min_prompt_len] = torch.tensor(p[:min_prompt_len], device=device)

        prompt_mask_tensor = token_ids_tensor != pad_id
        input_positions_tensor = torch.arange(0, min_prompt_len, dtype=torch.int64, device=device)

        # Causal & local masks -------------------------------------------------------
        mask_tensor = torch.full((1, 1, max_seq_len, max_seq_len), float("-inf"), device=device)
        mask_tensor = torch.triu(mask_tensor, diagonal=1)
        local_mask_tensor = None
        if self.config.sliding_window_size:
            local_band = torch.tril(torch.full_like(mask_tensor, float("-inf")), diagonal=-self.config.sliding_window_size)
            local_mask_tensor = mask_tensor + local_band

        curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
        curr_local_mask_tensor = (
            local_mask_tensor.index_select(2, input_positions_tensor) if local_mask_tensor is not None else None
        )
        output_positions_tensor = torch.tensor([min_prompt_len - 1], dtype=torch.int64, device=device)
        temperatures_tensor = None if temperature is None else torch.full((batch_size,), float(temperature), device=device)
        top_ps_tensor = torch.full((batch_size,), float(top_p), device=device)
        top_ks_tensor = torch.full((batch_size,), int(top_k), dtype=torch.int64, device=device)
        output_index = torch.tensor(min_prompt_len, dtype=torch.int64, device=device)

        # Prefill and autoregressive loop -------------------------------------------
        for _ in range(max_seq_len - min_prompt_len):
            next_token_ids, _ = self(
                input_token_ids=input_token_ids_tensor,
                input_positions=input_positions_tensor,
                kv_write_indices=input_positions_tensor,
                kv_caches=kv_caches,
                mask=curr_mask_tensor,
                output_positions=output_positions_tensor,
                temperatures=temperatures_tensor,
                top_ps=top_ps_tensor,
                top_ks=top_ks_tensor,
                local_mask=curr_local_mask_tensor,
            )

            curr_prompt_mask = prompt_mask_tensor.index_select(1, output_index).squeeze(1)
            curr_token_ids = token_ids_tensor.index_select(1, output_index).squeeze(1)
            output_token_ids = torch.where(curr_prompt_mask, curr_token_ids, next_token_ids).unsqueeze(1)
            token_ids_tensor.index_copy_(1, output_index, output_token_ids)

            # Shift one step ------------------------------------------------------
            input_token_ids_tensor = output_token_ids
            input_positions_tensor = output_index.unsqueeze(-1)
            curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
            curr_local_mask_tensor = (
                local_mask_tensor.index_select(2, input_positions_tensor) if local_mask_tensor is not None else None
            )
            output_positions_tensor = torch.tensor([0], dtype=torch.int64, device=device)
            output_index = output_index + 1

        # Detokenise ---------------------------------------------------------------
        results: List[str] = []
        for i, tokens in enumerate(token_ids_tensor.tolist()):
            trimmed = tokens[len(prompt_tokens[i]) : len(prompt_tokens[i]) + output_len]
            if self.tokenizer.eos_id in trimmed:
                trimmed = trimmed[: trimmed.index(self.tokenizer.eos_id)]
            results.append(self.tokenizer.decode(trimmed))

        return results[0] if is_str_prompt else results
