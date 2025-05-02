# gemma_asap/model_asap.py
# ---------------------------------------------------------------
# Implements “Analytical Spectrum-Aware Projection” attention for
# the open-source Gemma model code base.
# ---------------------------------------------------------------
from __future__ import annotations
import math
from typing import Dict, Tuple, Mapping, List, Any

import torch
import torch.nn.functional as F
from torch import nn

from gemma import config as gemma_cfg
from gemma.model import (
    RMSNorm,               # re-use existing utility layers
    apply_rotary_emb,      # identical Rot-ary PE helper
    GemmaMLP,
)

# ---------------------------------------------------------------
# ASAP-aware Attention
# ---------------------------------------------------------------
class ASAPAttention(nn.Module):
    """
    Drop-in replacement for GemmaAttention.

    Expects *buffers*  U_r, V_r, S_r (created by `convert_weights.py`)
    to live under  self.U_r  /  self.V_r  /  self.S_r with shapes

        • U_r : (n_kv_heads, head_dim, r)
        • V_r : (n_heads    , head_dim, r)
        • S_r : (n_heads    , r)        –– singular values

    Only **keys** are projected to rank-`r` for KV-cache compactness;
    queries are projected on the fly each step.
    """
    def __init__(
            self,
            cfg: gemma_cfg.GemmaConfig,
            attn_type: gemma_cfg.AttentionType,
            rank: int,
    ):
        super().__init__()

        self.attn_type = attn_type                # GLOBAL / LOCAL_SLIDING
        self.rank      = rank

        self.num_heads     = cfg.num_attention_heads
        self.num_kv_heads  = cfg.num_key_value_heads
        self.head_dim      = cfg.head_dim
        self.hidden_size   = cfg.hidden_size
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        # shared QKV linear (identical to vanilla Gemma)
        self.qkv_proj = nn.Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            bias=False,
            )
        self.o_proj  = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            )

        # multiplicative scale (same as Gemma)
        self.scale = self.head_dim ** -0.5

        # U_r / V_r / S_r will be loaded by state_dict
        self.register_buffer("U_r", torch.empty(
            self.num_kv_heads, self.head_dim, rank))
        self.register_buffer("V_r", torch.empty(
            self.num_heads,     self.head_dim, rank))
        self.register_buffer("S_r", torch.empty(
            self.num_heads,                rank))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _project_query(self, q_full: torch.Tensor) -> torch.Tensor:
        """
        q_full : (B, T, H, d)
        return : (B, T, H, r)
        """
        # einsum slightly faster than matmul with reshape for small r
        return torch.einsum("b t h d , h d r -> b t h r", q_full, self.V_r)

    def _project_key(self, k_full: torch.Tensor) -> torch.Tensor:
        """
        k_full : (B, T, H_kv, d)
        return : (B, T, H    , r)  (expanded to all heads)
        """
        k_r = torch.einsum("b t hkv d , hkv d r -> b t hkv r", k_full, self.U_r)

        # repeat keys if n_heads > n_kv_heads
        if self.num_kv_heads != self.num_heads:
            k_r = k_r.repeat_interleave(self.num_queries_per_kv, dim=2)

        return k_r  # (B,T,H,r)

    # ------------------------------------------------------------------
    def forward(
            self,
            hidden_states: torch.Tensor,                     # (B,T,hidden)
            freqs_cis: torch.Tensor,                         # rope table slice
            kv_write_indices: torch.Tensor,                  # (T,) positions
            kv_cache: Tuple[torch.Tensor, torch.Tensor],     # (k_cache, v_cache)
            mask: torch.Tensor,                              # (1,1,T_max,T_max) causal/global/local
            local_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        kv_cache shapes:
            • k_cache : (B, T_max, H_kv, r)
            • v_cache : (B, T_max, H_kv, d)
        """
        B, T_in, _ = hidden_states.shape
        r          = self.rank
        d          = self.head_dim

        # ------------------------------------------------------------
        # 1. QKV full projection (same as original)
        # ------------------------------------------------------------
        qkv = self.qkv_proj(hidden_states)                         # (B,T, *)
        q_full, k_full, v_full = torch.split(
            qkv,
            [self.num_heads * d, self.num_kv_heads * d, self.num_kv_heads * d],
            dim=-1)

        q_full = q_full.view(B, T_in, self.num_heads,     d)
        k_full = k_full.view(B, T_in, self.num_kv_heads,  d)
        v_full = v_full.view(B, T_in, self.num_kv_heads,  d)

        # ------------------------------------------------------------
        # 2. RoPE
        # ------------------------------------------------------------
        q_full = apply_rotary_emb(q_full, freqs_cis)      # (B,T,H,d)
        k_full = apply_rotary_emb(k_full, freqs_cis)      # (B,T,H_kv,d)

        # ------------------------------------------------------------
        # 3. Project to rank-r
        # ------------------------------------------------------------
        q_r = self._project_query(q_full)                 # (B,T,H,r)
        k_r = self._project_key(k_full)                   # (B,T,H,r)

        # 3a. write k_r & v_full into cache
        k_cache, v_cache = kv_cache
        k_cache.index_copy_(1, kv_write_indices, k_r)     # write along seq dim
        v_cache.index_copy_(1, kv_write_indices, v_full)

        # gather keys / values for *all* timesteps so far
        keys_r   = k_cache                                # (B,T_max,H_kv,r) → (B,T_max,H,r)
        values   = v_cache.repeat_interleave(
            self.num_queries_per_kv, dim=2)               # (B,T_max,H,d)

        # ------------------------------------------------------------
        # 4. Attention in r-space
        # ------------------------------------------------------------
        # scale queries: (B,T,H,r) * (H,r) → broadcast
        q_scaled = q_r * self.S_r.unsqueeze(0).unsqueeze(0)  # (B,T,H,r)

        # scores: (B,H,T_q,T_k)
        scores = torch.einsum(
            "b t h r , b s h r -> b h t s",
            q_scaled, keys_r) * (1.0 / math.sqrt(r))

        scores = scores + mask                              # causal / local mask
        scores = F.softmax(scores.float(), dim=-1).type_as(q_full)

        # ------------------------------------------------------------
        # 5. Weighted sum with *full-d* values  →  (B,T,H,d)
        # ------------------------------------------------------------
        context = torch.einsum("b h t s , b s h d -> b t h d", scores, values)

        # concat heads                                           (B,T,H*d)
        context = context.reshape(B, T_in, self.num_heads * d)

        # output projection
        out = self.o_proj(context)                           # (B,T,hidden)
        return out


# ---------------------------------------------------------------
# ASAP Decoder layer (identical to Gemma2 but with ASAPAttention)
# ---------------------------------------------------------------
class ASAPDecoderLayer(nn.Module):
    def __init__(self, cfg: gemma_cfg.GemmaConfig, attn_type, rank: int):
        super().__init__()
        self.attn_type = attn_type
        self.self_attn = ASAPAttention(cfg, attn_type, rank)
        self.input_ln  = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attn_ln = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = GemmaMLP(cfg.hidden_size, cfg.intermediate_size, cfg.quant)

    def forward(
            self,
            hidden_states: torch.Tensor,
            freqs_cis: torch.Tensor,
            kv_write_indices: torch.Tensor,
            kv_cache: Tuple[torch.Tensor, torch.Tensor],
            mask: torch.Tensor,
            local_mask: torch.Tensor | None,
    ):
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_ln(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            freqs_cis,
            kv_write_indices,
            kv_cache,
            mask,
            local_mask,
        )
        hidden_states = residual + hidden_states

        # Feed-forward
        residual = hidden_states
        hidden_states = self.post_attn_ln(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# ---------------------------------------------------------------
# Top-level model wrapper  (only diff: layer class, kv-cache shape)
# ---------------------------------------------------------------
class GemmaASAPModel(nn.Module):
    """
    Mirrors GemmaModel but builds `ASAPDecoderLayer`s and exposes
    `kv_cache_shapes(rank)` for the runner script.
    """
    def __init__(self, cfg: gemma_cfg.GemmaConfig, rank: int):
        super().__init__()
        self.cfg  = cfg
        self.rank = rank

        self.layers = nn.ModuleList()
        for i in range(cfg.num_hidden_layers):
            attn_type = (
                cfg.attn_types[i % len(cfg.attn_types)]
                if cfg.attn_types is not None
                else gemma_cfg.AttentionType.GLOBAL
            )
            self.layers.append(ASAPDecoderLayer(cfg, attn_type, rank))

        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    # ---- forward -------------------------------------------------------
    def forward(
            self,
            hidden_states: torch.Tensor,
            freqs_cis: Mapping[gemma_cfg.AttentionType, torch.Tensor],
            kv_write_indices: torch.Tensor,
            kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
            mask: torch.Tensor,
            local_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        for layer, kv in zip(self.layers, kv_caches):
            hidden_states = layer(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis[layer.attn_type],
                kv_write_indices=kv_write_indices,
                kv_cache=kv,
                mask=mask,
                local_mask=local_mask,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states

    # ---- helper for runner --------------------------------------------
    def kv_cache_shapes(self, batch: int, max_seq: int
                        ) -> List[Tuple[torch.Size, torch.Size]]:
        """
        Returns per-layer shapes for   (k_cache , v_cache)
        """
        kshape = torch.Size((batch, max_seq,
                             self.cfg.num_key_value_heads, self.rank))
        vshape = torch.Size((batch, max_seq,
                             self.cfg.num_key_value_heads, self.cfg.head_dim))
        return [(kshape, vshape) for _ in range(self.cfg.num_hidden_layers)]


class GemmaASAPForCausalLM(nn.Module):
    """
    High-level LM wrapper identical to GemmaForCausalLM except it
    embeds `GemmaASAPModel` and allocates smaller KV caches.
    """
    def __init__(self, cfg: gemma_cfg.GemmaConfig, rank: int):
        super().__init__()
        self.cfg   = cfg
        self.rank  = rank
        from gemma.model import Embedding, Sampler            # reuse originals

        self.tokenizer = None  # set by run script if needed
        self.embedder = Embedding(cfg.vocab_size, cfg.hidden_size, cfg.quant)
        self.model    = GemmaASAPModel(cfg, rank)
        self.sampler  = Sampler(cfg.vocab_size, cfg)

    # ------------------------------------------------------------------
    # weight loading (strict=False so ASAP buffers slot in seamlessly)
    # ------------------------------------------------------------------
    def load_weights_asap(self, ckpt: str | Any):
        state = torch.load(ckpt, mmap=True)["model_state_dict"]
        self.load_state_dict(state, strict=False)

    # ------------------------------------------------------------------
    # generation / forward left to runner;
    # simply exposing reduced-rank cache allocation convenience:
    # ------------------------------------------------------------------
    def alloc_kv_caches(self, batch: int, max_seq: int, device) \
            -> List[Tuple[torch.Tensor, torch.Tensor]]:
        caches = []
        for kshape, vshape in self.model.kv_cache_shapes(batch, max_seq):
            k = torch.zeros(kshape, dtype=self.cfg.get_dtype(), device=device)
            v = torch.zeros(vshape, dtype=self.cfg.get_dtype(), device=device)
            caches.append((k, v))
        return caches
