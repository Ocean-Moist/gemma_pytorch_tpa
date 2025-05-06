"""Gemma model variant that *only* supports DK‑SVD‑compressed checkpoints.

This implementation is a minimal fork of ``gemma/model.py`` with the following
simplifications/assumptions:

* ``config.use_dksvd`` **must** be ``True`` and ``config.dksvd_rank`` **must** be
  provided.  (The original, un‑compressed pathway is intentionally *not*
  supported here – defer to ``gemma.model`` for that.)
* The model is expected to have **one** GQA group (``num_key_value_heads == 1``),
  which is true for Gemma‑1b and the conversion script produced by the DK‑SVD
  plan.
* Only GLOBAL attention is implemented; LOCAL_SLIDING can be added later if
  necessary.
* KV‑cache tensor shapes are reduced to match the compressed dimensionalities:
    * **K‑cache:** *(batch, max_seq_len, 1, r)* where *r = dksvd_rank*.
    * **V‑cache:** *(batch, max_seq_len, 1, head_dim)*.

The file purposefully re‑uses a handful of helper classes/functions from the
original implementation to stay lean:
``Linear, Embedding, RMSNorm, GemmaMLP, Sampler,
 precompute_freqs_cis, apply_rotary_emb``.
"""

from __future__ import annotations

import math
from typing import Mapping, Tuple, List, Any, Union

import torch
from torch import nn
import torch.nn.functional as F

from gemma import config as gemma_config
# Re‑use helpers from the canonical model implementation
from gemma.model import (
    Linear,
    Embedding,
    RMSNorm,
    GemmaMLP,
    Sampler,
    precompute_freqs_cis,
    apply_rotary_emb,
)


# ---------------------------------------------------------------------------
#  DK‑SVD Attention
# ---------------------------------------------------------------------------
class DKSVDAttention(nn.Module):
    """Single‑group attention using DK‑SVD‑compressed Q/K projections."""

    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        if not getattr(config, "use_dksvd", False):
            raise ValueError("model_dksvd.py expects GemmaConfig.use_dksvd = True")
        if config.num_key_value_heads != 1:
            raise ValueError(
                "DKSVDAttention only supports num_key_value_heads == 1 (1 GQA group)."
            )

        # ------------------------------------------------------------------
        #  Dimensions
        # ------------------------------------------------------------------
        self.hidden_size: int = config.hidden_size  # d
        self.head_dim: int = config.head_dim        # d_v (=256 for Gemma‑1b)
        self.rank: int = config.dksvd_rank          # r (e.g. 64)

        # Value path keeps original dimensionality (d_v)
        self.v_size: int = self.head_dim  # since one KV head

        # Scaling factor for Q·K^T (like √d_k in vanilla attention)
        self.scaling: float = self.rank ** -0.5

        # ------------------------------------------------------------------
        #  Projections
        # ------------------------------------------------------------------
        quant = getattr(config, "quant", False)
        self.q_proj = Linear(self.hidden_size, self.rank, quant)
        self.k_proj = Linear(self.hidden_size, self.rank, quant)
        self.v_proj = Linear(self.hidden_size, self.v_size, quant)
        self.o_proj = Linear(self.v_size, self.hidden_size, quant)

        # Optional Q/K RMSNorm (same as original Gemma logic)
        self.query_norm = (
            RMSNorm(self.rank, eps=config.rms_norm_eps) if config.use_qk_norm else None
        )
        self.key_norm = (
            RMSNorm(self.rank, eps=config.rms_norm_eps) if config.use_qk_norm else None
        )

        # Misc flags (kept for compatibility – LOCAL masks etc. not supported yet)
        self.attn_logit_softcapping = config.attn_logit_softcapping

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def forward(
            self,
            hidden_states: torch.Tensor,           # (B, S, hidden_size)
            freqs_cis: torch.Tensor,               # (S, rank//2)
            kv_write_indices: torch.Tensor,        # (S,) positions being written
            kv_cache: Tuple[torch.Tensor, torch.Tensor],  # (k_cache, v_cache)
            mask: torch.Tensor,                    # (1, 1, S, max_len)
            local_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute self‑attention with DK‑SVD compressed projections.

        Shapes follow Gemma conventions – see comment above.
        """
        batch_size, seq_len, _ = hidden_states.shape
        k_cache, v_cache = kv_cache  # shapes (B, max_len, 1, r) and (B, max_len, 1, head_dim)

        # ------------------------------------------------------------------
        #  Projections
        # ------------------------------------------------------------------
        q = self.q_proj(hidden_states)  # (B, S, r)
        k = self.k_proj(hidden_states)  # (B, S, r)
        v = self.v_proj(hidden_states)  # (B, S, head_dim)

        # Optional Q/K norm
        if self.query_norm is not None:
            q = self.query_norm(q)
            k = self.key_norm(k)

        # Reshape to (B, S, 1, r) for RoPE then back
        q = q.view(batch_size, seq_len, 1, self.rank)
        k = k.view(batch_size, seq_len, 1, self.rank)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # Collapse num_heads dimension (==1)
        q = q.view(batch_size, seq_len, self.rank)
        k = k.view(batch_size, seq_len, self.rank)

        # ------------------------------------------------------------------
        #  KV cache update
        # ------------------------------------------------------------------
        # k_cache: (B, max_len, 1, r)
        # v_cache: (B, max_len, 1, head_dim)
        k_cache.index_copy_(1, kv_write_indices, k.view(batch_size, seq_len, 1, self.rank))
        v_cache.index_copy_(1, kv_write_indices, v.view(batch_size, seq_len, 1, self.head_dim))

        # Retrieve cached keys/values across full context
        k_all = k_cache  # (B, max_len, 1, r)
        v_all = v_cache  # (B, max_len, 1, head_dim)

        # Transpose to (B, 1, seq_len, r) / (B, 1, max_len, r)
        q_for_scores = q.view(batch_size, seq_len, 1, self.rank).transpose(1, 2)
        k_for_scores = k_all.transpose(1, 2)  # (B, 1, max_len, r)

        # ------------------------------------------------------------------
        #  Scaled dot‑product
        # ------------------------------------------------------------------
        attn_scores = torch.matmul(q_for_scores, k_for_scores.transpose(2, 3))  # (B,1,S,max_len)
        attn_scores = attn_scores * self.scaling

        # Apply mask(s)
        if local_mask is not None:
            mask = local_mask  # Not actually supported; kept for signature compatibility
        attn_scores = attn_scores + mask  # broadcast (1,1,S,max_len)

        if self.attn_logit_softcapping is not None:
            attn_scores = attn_scores / self.attn_logit_softcapping
            attn_scores = torch.tanh(attn_scores) * self.attn_logit_softcapping

        attn_probs = F.softmax(attn_scores.float(), dim=-1).type_as(attn_scores)

        # ------------------------------------------------------------------
        #  Attention output
        # ------------------------------------------------------------------
        v_for_output = v_all.transpose(1, 2)  # (B,1,max_len,head_dim)
        context = torch.matmul(attn_probs, v_for_output)  # (B,1,S,head_dim)

        # Restore to (B, S, head_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.head_dim)

        # Output projection to hidden_size
        output = self.o_proj(context)
        return output


# ---------------------------------------------------------------------------
#  Decoder Layer
# ---------------------------------------------------------------------------
class DKSVDecoderLayer(nn.Module):
    """Minimal decoder layer using DK‑SVD attention + MLP."""

    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.self_attn = DKSVDAttention(config)
        self.mlp = GemmaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant=getattr(config, "quant", False),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Optional extra norms (Gemma‑2 style)
        self.pre_feedforward_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_pre_ffw_norm else None
        )
        self.post_feedforward_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_post_ffw_norm else None
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
        # Self‑attention block
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

        # Feed‑forward block
        residual = hidden_states
        if self.pre_feedforward_layernorm is not None:
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.post_feedforward_layernorm is not None:
            hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
#  Transformer body
# ---------------------------------------------------------------------------
class GemmaModelDKSVD(nn.Module):
    """Stack of DK‑SVD decoder layers + final RMSNorm."""

    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([DKSVDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,                       # (B,S,d)
            freqs_cis: torch.Tensor,                           # (S, r//2)
            kv_write_indices: torch.Tensor,
            kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
            mask: torch.Tensor,
            local_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        for layer, kv_cache in zip(self.layers, kv_caches):
            hidden_states = layer(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis,
                kv_write_indices=kv_write_indices,
                kv_cache=kv_cache,
                mask=mask,
                local_mask=local_mask,
            )
        return self.norm(hidden_states)


# ---------------------------------------------------------------------------
#  Causal‑LM wrapper
# ---------------------------------------------------------------------------
class GemmaForCausalLMDKSVD(nn.Module):
    """Causal‑LM model that consumes DK‑SVD‑compressed checkpoints."""

    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        if not getattr(config, "use_dksvd", False):
            raise ValueError("GemmaForCausalLMDKSVD requires config.use_dksvd = True")
        if config.dksvd_rank % 2 != 0:
            raise ValueError("dksvd_rank must be even to use rotary embeddings.")

        self.config = config
        self.vocab_size = config.vocab_size

        # Tokenizer + embedding reuse – import lazily to avoid heavy deps during compile‑time.
        from gemma import tokenizer as _tok  # pylint: disable=import-outside-toplevel
        self.tokenizer = _tok.Tokenizer(config.tokenizer)

        self.embedder = Embedding(self.vocab_size, config.hidden_size, quant=getattr(config, "quant", False))
        self.model = GemmaModelDKSVD(config)
        self.sampler = Sampler(self.vocab_size, config)

        # Pre‑compute RoPE table for compressed rank
        self._register_freqs_cis("freqs_cis_dksvd", config.dksvd_rank, config.max_position_embeddings * 2,
                                 theta=getattr(config, "rope_wave_length", {gemma_config.AttentionType.GLOBAL: 10_000}).get(
                                     gemma_config.AttentionType.GLOBAL, 10_000))

    # -------------------------- helpers ----------------------------------
    def _register_freqs_cis(self, name: str, dim: int, end: int, theta: int = 10_000):
        self.register_buffer(name, precompute_freqs_cis(dim, end, theta=theta))

    # -------------------------- forward ----------------------------------
    @torch.no_grad()
    def forward(
            self,
            input_token_ids: torch.Tensor,    # (B,S)
            input_positions: torch.Tensor,    # (S,)
            kv_write_indices: torch.Tensor,   # (S,)
            kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
            mask: torch.Tensor,               # (1,1,S,max_len)
            output_positions: torch.Tensor,   # (B,) or (1,)
            temperatures: Union[torch.Tensor, None],
            top_ps: torch.Tensor,
            top_ks: torch.Tensor,
            local_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        freqs_cis_slice = self.freqs_cis_dksvd.index_select(0, input_positions)  # (S, r/2)

        # Input embed + scaling (Gemma style – multiply by √d)
        hidden_states = self.embedder(input_token_ids)
        hidden_states = hidden_states * math.sqrt(self.config.hidden_size)

        hidden_states = self.model(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis_slice,
            kv_write_indices=kv_write_indices,
            kv_caches=kv_caches,
            mask=mask,
            local_mask=local_mask,
        )

        # Final sampling (share weight with embedder)
        embed_W = self.embedder.weight
        if getattr(self.config, "quant", False):
            embed_W = embed_W * self.embedder.weight_scaler.unsqueeze(-1)

        next_tokens, logits = self.sampler(
            embedding=embed_W,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            embedding_bias=None,
        )
        return next_tokens, logits

    # -------------------- utility: load_weights --------------------------
    def load_weights(self, ckpt_path: str):
        """Thin wrapper around ``nn.Module.load_state_dict`` that sets ``strict=False``.

        The conversion script saves only the DK‑SVD‑specific keys (``q_proj``/``k_proj``/``v_proj``/``o_proj``)
        plus all untouched parameters – so missing keys (e.g. original ``qkv_proj``) are ignored.
        """
        state = torch.load(ckpt_path, map_location="cpu")
        if "model_state_dict" in state:  # produced by convert_weights.py
            state = state["model_state_dict"]
        self.load_state_dict(state, strict=False)

"""
Minimal *public* API re‑exports – this makes ``gemma.model_dksvd`` behave like
``gemma.model`` from the perspective of the runner script (import layer).
"""
__all__ = [
    "GemmaForCausalLMDKSVD",
    "GemmaModelDKSVD",
]
