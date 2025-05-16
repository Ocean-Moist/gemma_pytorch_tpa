# Energy‑DK‑SVD compressed Gemma model — **FIXED RMS‑Norm handling**
# -----------------------------------------------------------------------------
# This file *replaces* the original `gemma/model_dksvd.py`.  The only functional
# change w.r.t. the previous draft is the **correct treatment of the low‑rank
# per‑channel RMS‑Norm scales** that are produced by `scripts/convert_weights.py`.
# In particular we now
#   • always apply the per‑channel norms to *both* Q **and** shared‑K pathways
#     exactly once (they were previously missing for the shared key),
#   • tie the scaling vectors to the compressed width `r_g`, and
#   • guarantee that the parameter semantics match the conversion script:
#       ‑ the checkpoint stores γ′  (the *offset*),
#       ‑ `RMSNorm(add_unit_offset=True)` therefore multiplies by (1+γ′).
#
# The remainder of the code is identical to the reference Gemma implementation
# except for the low‑rank projections and the cache shapes.
# -----------------------------------------------------------------------------
from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from gemma import config as gemma_config
from gemma import tokenizer

# We reuse the exact helper layers shipped with the official Gemma repo so that
# weight‑loading remains fully forward‑compatible.
from gemma.model import (
    Linear,
    Embedding,
    RMSNorm,
    precompute_freqs_cis,
    apply_rotary_emb,
    GemmaMLP,
    Sampler,
)

# -----------------------------------------------------------------------------
# Attention block (low‑rank Q/K via Energy‑DK‑SVD)
# -----------------------------------------------------------------------------
class GemmaAttentionDKSVD(nn.Module):
    """Self‑attention that consumes the skinny Q/K matrices from E‑DK‑SVD.

    The implementation follows the mathematical derivation in §5–§6 of the
    Energy‑DK‑SVD document.  The only non‑standard features compared to the
    original Gemma attention are:
      • multiple query projection "heads" per GQA group, each width *r_g*,
      • one shared key projection of the same width per group, and
      • per‑channel RMS‑Norm layers that act on that compressed width.
    """

    def __init__(self, config: gemma_config.GemmaConfig, attn_type: gemma_config.AttentionType):
        super().__init__()

        if not getattr(config, "qk_rank", None):
            raise ValueError("GemmaConfig.qk_rank must be set when using the DK‑SVD variant.")

        # ----------------------------  Core sizes  ----------------------------
        self.num_heads: int = config.num_attention_heads            # N_h
        self.num_kv_heads: int = config.num_key_value_heads         # N_kv
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")
        self.num_q_per_kv: int = self.num_heads // self.num_kv_heads  # s

        self.hidden_size: int = config.hidden_size                  # d
        self.qk_rank: int = config.qk_rank                          # r_g
        self.v_head_dim: int = config.head_dim                      # d_v (unchanged)

        # Dot-product scaling – **always** use the original head width.
        # This is exactly what the reference implementation does:
        #   scaling = (query_pre_attn_scalar or head_dim) ** -0.5
        self.scaling = (config.query_pre_attn_scalar or self.v_head_dim) ** -0.5

        # ----------------------  Per‑group linear projections  ----------------------
        # Shared *key* and *value* per GQA group.
        self.k_linears = nn.ModuleList(
            [Linear(self.hidden_size, self.qk_rank, quant=config.quant) for _ in range(self.num_kv_heads)]
        )
        self.v_linears = nn.ModuleList(
            [Linear(self.hidden_size, self.v_head_dim, quant=config.quant) for _ in range(self.num_kv_heads)]
        )

        # *s* separate *query* projections for every group.
        self.q_linears = nn.ModuleList(
            [
                nn.ModuleList(
                    [Linear(self.hidden_size, self.qk_rank, quant=config.quant) for _ in range(self.num_q_per_kv)]
                )
                for _ in range(self.num_kv_heads)
            ]
        )

        # Output projection identical to the reference implementation.
        self.o_proj = Linear(self.num_heads * self.v_head_dim, self.hidden_size, quant=config.quant)

        # -----------------------  Low‑rank RMS‑Norm layers  -----------------------
        # The conversion script stores **γ′** (the offset) so we keep the Gemma
        # default `add_unit_offset=True` – at runtime we multiply by (1+γ′).
        if config.use_qk_norm:
            self.query_norm = RMSNorm(self.qk_rank, eps=config.rms_norm_eps, add_unit_offset=True)
            self.key_norm = RMSNorm(self.qk_rank, eps=config.rms_norm_eps, add_unit_offset=True)
            # --- § 5.2  “colour”  A½ -------------------------------------------------
            # One SPD root per GQA group, separate for Q and K.
            eye = torch.eye(self.qk_rank)
            #   · they must *not* appear in model.parameters()
            #   · register as buffers keeps dtype / device correct
            self.register_buffer("query_colour",
                                 torch.stack([eye] * self.num_kv_heads))   # (G, r, r)
            self.register_buffer("key_colour",
                                 torch.stack([eye] * self.num_kv_heads))    # (G, r, r)
        else:
            self.query_norm = None
            self.key_norm = None
            self.query_colour = None
            self.key_colour   = None
        # Misc.
        self.attn_type = attn_type
        self.sliding_window_size = config.sliding_window_size
        self.attn_logit_softcapping = config.attn_logit_softcapping

    # -------------------------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def forward(
            self,
            hidden_states: torch.Tensor,                      # (B, T, d)
            freqs_cis: Optional[torch.Tensor],                # (T, r_g//2) or None
            kv_write_indices: torch.Tensor,                   # (T,)
            kv_cache: Tuple[torch.Tensor, torch.Tensor],      # (k_cache, v_cache)
            mask: torch.Tensor,                               # broadcastable attention mask
            local_mask: Optional[torch.Tensor] = None,        # local sliding‑window mask
    ) -> torch.Tensor:                                    # (B, T, d)
        """A minimal — but fully correct — implementation of GQA with skinny Q/K.

        The cache layout matches the reference Gemma code except that the key
        width is `r_g` instead of `head_dim`.
        """
        B, T, _ = hidden_states.shape
        k_cache, v_cache = kv_cache                       # shapes set by caller

        # ------------------------------------------------------------------
        # 1. Build projections per GQA group & update KV cache
        # ------------------------------------------------------------------
        query_chunks: List[torch.Tensor] = []   # will be stacked into (B, T, N_h, r_g)
        key_chunks: List[torch.Tensor] = []
        value_chunks: List[torch.Tensor] = []

        for g in range(self.num_kv_heads):
            # ------ Shared key --------------------------------------------------
            k_g = self.k_linears[g](hidden_states)         # (B, T, r_g)
            if self.key_norm is not None:
                k_g = self.key_norm(k_g)
            if freqs_cis is not None:
                k_g = apply_rotary_emb(k_g.view(B, T, 1, self.qk_rank), freqs_cis).squeeze(2)
            if self.key_norm is not None:
                k_g = torch.matmul(k_g, self.key_colour[g])
            # ------ Shared value -----------------------------------------------
            v_g = self.v_linears[g](hidden_states)         # (B, T, d_v)

            # ------ Write into the KV cache ------------------------------------
            k_cache[:, kv_write_indices, g, :] = k_g
            v_cache[:, kv_write_indices, g, :] = v_g

            key_chunks.append(k_cache[:, : k_cache.shape[1], g, :])     # (B, L, r_g)
            value_chunks.append(v_cache[:, : v_cache.shape[1], g, :])   # (B, L, d_v)

            # ------ Per‑head queries -------------------------------------------
            for i in range(self.num_q_per_kv):
                q_gi = self.q_linears[g][i](hidden_states)  # (B, T, r_g)
                if self.query_norm is not None:
                    q_gi = self.query_norm(q_gi)
                if freqs_cis is not None:
                    q_gi = apply_rotary_emb(q_gi.view(B, T, 1, self.qk_rank),
                                            freqs_cis).squeeze(2)
                if self.query_norm is not None:      # colour **after** RoPE
                    q_gi = torch.matmul(q_gi, self.query_colour[g])
                query_chunks.append(q_gi)

        # ------------------------------------------------------------------
        # 2. Stack →  (B, N_h, T/L, r_g)  layout expected by the matmuls
        # ------------------------------------------------------------------
        xq = torch.stack(query_chunks, dim=2)              # (B, T, N_h, r_g)
        k  = torch.stack(key_chunks,   dim=2)              # (B, L, N_kv, r_g)
        v  = torch.stack(value_chunks, dim=2)              # (B, L, N_kv, d_v)

        # Re‑broadcast shared key/value so every query head has a partner.
        k = torch.repeat_interleave(k, self.num_q_per_kv, dim=2)  # (B, L, N_h, r_g)
        v = torch.repeat_interleave(v, self.num_q_per_kv, dim=2)  # (B, L, N_h, d_v)

        # Final reshape for the attention kernel.
        q = xq.transpose(1, 2)                             # (B, N_h, T, r_g)
        k = k.transpose(1, 2)                              # (B, N_h, L, r_g)
        v = v.transpose(1, 2)                              # (B, N_h, L, d_v)

        # ------------------------------------------------------------------
        # 3. Scaled dot‑product attention
        # ------------------------------------------------------------------
        q = q * self.scaling
        scores = torch.matmul(q, k.transpose(-2, -1))      # (B, N_h, T, L)

        # Optional local sliding‑window masking (Gemma‑3 style)
        if (
                self.attn_type == gemma_config.AttentionType.LOCAL_SLIDING
                and self.sliding_window_size is not None
                and local_mask is not None
        ):
            mask = local_mask

        if self.attn_logit_softcapping is not None:
            scores = torch.tanh(scores / self.attn_logit_softcapping) * self.attn_logit_softcapping

        scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(q)

        context = torch.matmul(scores, v)                  # (B, N_h, T, d_v)
        context = context.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(context)


# -----------------------------------------------------------------------------
# Decoder layer wrapper — identical to Gemma‑3 but with our custom attention
# -----------------------------------------------------------------------------
class Gemma2DecoderLayerDKSVD(nn.Module):
    def __init__(self, config: gemma_config.GemmaConfig, attn_type: gemma_config.AttentionType):
        super().__init__()
        self.attn_type = attn_type
        self.self_attn = GemmaAttentionDKSVD(config, attn_type)
        self.mlp = GemmaMLP(config.hidden_size, config.intermediate_size, config.quant)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) if config.use_pre_ffw_norm else None
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) if config.use_post_ffw_norm else None

    def forward(
            self,
            hidden_states: torch.Tensor,
            freqs_cis: torch.Tensor,
            kv_write_indices: torch.Tensor,
            kv_cache: Tuple[torch.Tensor, torch.Tensor],
            mask: torch.Tensor,
            local_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # ---- Attention ------------------------------------------------------
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            freqs_cis,
            kv_write_indices,
            kv_cache,
            mask,
            local_mask,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # ---- MLP ------------------------------------------------------------
        residual = hidden_states
        if self.pre_feedforward_layernorm is not None:
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.post_feedforward_layernorm is not None:
            hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# -----------------------------------------------------------------------------
# Stacked decoder + LM wrapper
# -----------------------------------------------------------------------------
class GemmaModelDKSVD(nn.Module):
    """Transformer backbone using the DK‑SVD attention layers."""

    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            attn_type = (
                config.attn_types[i % len(config.attn_types)]
                if config.attn_types is not None
                else gemma_config.AttentionType.GLOBAL
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
            local_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        for layer, cache in zip(self.layers, kv_caches):
            hidden_states = layer(
                hidden_states,
                freqs_cis.get(layer.attn_type),
                kv_write_indices,
                cache,
                mask,
                local_mask,
            )
        return self.norm(hidden_states)


class GemmaForCausalLMDKSVD(nn.Module):
    """Causal‑LM wrapper around the DK‑SVD backbone."""

    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        if not getattr(config, "qk_rank", None):
            raise ValueError("GemmaConfig.qk_rank must be set for DK‑SVD models.")
        self.config = config

        self.tokenizer = tokenizer.Tokenizer(config.tokenizer)
        self.embedder = Embedding(config.vocab_size, config.hidden_size, config.quant)
        self.model = GemmaModelDKSVD(config)
        self.sampler = Sampler(config.vocab_size, config)

        # -------- Pre‑compute RoPE tables (width = r_g) ----------------------
        if config.architecture == gemma_config.Architecture.GEMMA_3:
            if config.rope_wave_length is None:
                raise ValueError("rope_wave_length must be provided for Gemma‑3 models.")
            for attn_type, name in [
                (gemma_config.AttentionType.LOCAL_SLIDING, "local_freqs_cis"),
                (gemma_config.AttentionType.GLOBAL, "global_freqs_cis"),
            ]:
                theta = config.rope_wave_length.get(attn_type, 10_000)
                self._register_freqs_cis(name, config.qk_rank, config.max_position_embeddings * 2, theta)
        else:
            self._register_freqs_cis("freqs_cis", config.qk_rank, config.max_position_embeddings * 2)

    # ------------------------------------------------------------------
    def _register_freqs_cis(self, name: str, dim: int, max_len: int, theta: int = 10_000):
        self.register_buffer(name, precompute_freqs_cis(dim, max_len, theta))

    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(
            self,
            input_token_ids: torch.Tensor,
            input_positions: torch.Tensor,
            kv_write_indices: torch.Tensor,
            kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
            mask: torch.Tensor,
            output_positions: torch.Tensor,
            temperatures: Optional[torch.Tensor],
            top_ps: torch.Tensor,
            top_ks: torch.Tensor,
            local_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Gather position‑dependent RoPE tables.
        freqs_lookup = {}
        if self.config.architecture == gemma_config.Architecture.GEMMA_3:
            freqs_lookup[gemma_config.AttentionType.LOCAL_SLIDING] = self.local_freqs_cis.index_select(0, input_positions)
            freqs_lookup[gemma_config.AttentionType.GLOBAL] = self.global_freqs_cis.index_select(0, input_positions)
        else:
            shared = self.freqs_cis.index_select(0, input_positions)
            freqs_lookup[gemma_config.AttentionType.LOCAL_SLIDING] = shared
            freqs_lookup[gemma_config.AttentionType.GLOBAL] = shared

        # ---- Embedding & scale ------------------------------------------------
        hidden_states = self.embedder(input_token_ids) * (self.config.hidden_size ** 0.5)

        hidden_states = self.model(
            hidden_states,
            freqs_lookup,
            kv_write_indices,
            kv_caches,
            mask,
            local_mask,
        )

        embed_weight = self.embedder.weight
        if self.config.quant:
            embed_weight = embed_weight * self.embedder.weight_scaler.unsqueeze(-1)

        next_tokens, logits = self.sampler(
            embedding=embed_weight,
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
