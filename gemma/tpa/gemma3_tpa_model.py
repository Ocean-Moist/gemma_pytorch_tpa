import torch
from torch import nn
import torch.nn.functional as F

from .. import config as gemma_config

class RMSNorm(nn.Module):
    """RMS Normalization module."""
    def __init__(self, dim: int, eps: float = 1e-6, add_unit_offset: bool = True):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # cast to float32 for stability, then back
        out = self._norm(x.float())
        if self.add_unit_offset:
            out = out * (1.0 + self.weight.float())
        else:
            out = out * self.weight.float()
        return out.type_as(x)


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
        #   i.e. repeat each kv-head for heads_per_kv times
        if self.num_kv_heads < self.num_heads:
            heads_per_kv = self.num_heads // self.num_kv_heads
            A_k = A_k.repeat_interleave(heads_per_kv, dim=2)  # expand along #heads dimension
            A_v = A_v.repeat_interleave(heads_per_kv, dim=2)

        # Optionally apply RMSNorm to B_q, B_k if config says use_qk_norm
        if self.query_norm is not None and self.key_norm is not None:
            # We norm each rank slice in B_q and B_k
            for r in range(self.q_rank):
                B_q[:, :, r, :] = self.query_norm(B_q[:, :, r, :])
            for r in range(self.k_rank):
                B_k[:, :, r, :] = self.key_norm(B_k[:, :, r, :])

        # --------------
        # 1) Build all pairwise dot-products among B-factors
        # B_q: [b, q_len, q_rank, d], B_k: [b, kv_len, k_rank, d]
        # => B_dots: [b, q_len, kv_len, q_rank, k_rank]
        # using an einsum for all query-key pairs
        B_dots = torch.einsum('bqrd,bkrd->bqkrs', B_q, B_k)  # shape [b, q_len, kv_len, q_rank, k_rank]

        # 2) scale by 1/sqrt(head_dim)
        B_dots = B_dots * self.scaling

        # 3) incorporate A_q, A_k => attention logits
        # A_q: [b, q_len, #heads, q_rank]
        # A_k: [b, kv_len, #heads, k_rank]
        # B_dots: [b, q_len, kv_len, q_rank, k_rank]
        #
        # We want: attn_logits[b, heads, q_len, kv_len]
        # = sum_{r=1..q_rank} sum_{s=1..k_rank} A_q[h,r] * A_k[h,s] * B_dots[r,s].
        #
        # We'll combine them in an einsum:
        attn_weights = torch.einsum(
            'bqhr,bkhs,bqkrs->bhqk',
            A_q, A_k, B_dots
        )
        # shape: [b, #heads, q_len, kv_len]

        # 4) optional softcapping
        if self.attn_logit_softcapping is not None:
            # e.g. logit / capping => tanh => * capping
            attn_weights = attn_weights / self.attn_logit_softcapping
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * self.attn_logit_softcapping

        # local sliding window mask if needed
        if (
                self.attn_type == gemma_config.AttentionType.LOCAL_SLIDING
                and self.sliding_window_size is not None
                and local_mask is not None
        ):
            mask = local_mask

        # add the mask
        if mask is not None:
            # mask shape might be [b, 1, q_len, kv_len]
            # attn_weights shape is [b, #heads, q_len, kv_len]
            attn_weights = attn_weights + mask

        # softmax
        attn_probs = F.softmax(attn_weights.float(), dim=-1).type_as(hidden_states)

        # --------------
        # 5) Multiply by factorized V => final attention output
        # Instead of forming V = sum_{u=1..v_rank} A_v * B_v, we do a single batched einsum:
        #
        # attn_output[b,h,q,d] =
        #     sum_{k=1..kv_len} attn_probs[b,h,q,k] *
        #                       sum_{u=1..v_rank}  ( A_v[b,k,h,u] * B_v[b,k,u,d] )
        #
        # Combine into one operation:
        # we can first do a partial factor combination, or do it all in a single step:
        #
        # We'll do it in a single step:
        # attn_out = einsum(
        #   'bhqk, bkh u, bkh d -> bhqd',
        #   attn_probs, A_v, B_v
        # )
        # but be mindful that A_v: [b, kv_len, #heads, v_rank],
        # and B_v: [b, kv_len, v_rank, d].
        #
        # We'll rearrange to ensure the correct subscript usage:

        attn_output = torch.einsum(
            'bhqk,bkhu,bkud->bhqd',
            attn_probs,
            A_v,  # shape [b, kv_len, #heads, v_rank]
            B_v   # shape [b, kv_len, v_rank, head_dim]
        )
        # shape => [b, #heads, q_len, head_dim]

        # --------------
        # 6) reshape and final linear projection
        attn_output = attn_output.transpose(1, 2).contiguous()   # => [b, q_len, #heads, head_dim]
        attn_output = attn_output.view(bsz, seq_len, self.num_heads * self.head_dim)
        output = self.o_proj(attn_output)

        # check for NaNs, fallback
        if torch.isnan(output).any():
            print("WARNING: NaN in TPA output, replacing with zero.")
            output = torch.nan_to_num(output, nan=0.0)

        return output

    def _apply_rotary_emb_to_B(self, B: torch.Tensor, freqs_cis: torch.Tensor):
        """
        Factor-only RoPE: apply rotary to each rank slice of B.
        B shape: [b, seq, rank, head_dim]
        freqs_cis shape: [seq, head_dim//2] in complex form (like LLaMA).
        We basically treat each rank separately.
        """
        bsz, seq_len, rank, hd = B.shape
        # ensure hd is even
        if hd % 2 != 0:
            raise ValueError("Head dim must be even to apply rotary embeddings")

        B_out = torch.zeros_like(B)
        # for each token t
        for t in range(seq_len):
            # fetch the complex rotation for position t, shape [head_dim//2]
            # convert it to [1, 1, head_dim//2] if needed
            freqs = freqs_cis[t]  # complex shape [hd//2]
            # We'll expand the real+imag view
            for r in range(rank):
                # interpret B[:, t, r, :] as pairs of real coords => complex => multiply => real
                b_slice = B[:, t, r, :]  # shape [bsz, hd]
                # reshape to complex
                b_complex = torch.view_as_complex(b_slice.float().reshape(bsz, -1, 2))
                # multiply
                b_rotated = b_complex * freqs  # broadcast
                # back to real
                b_rotated = b_rotated.view(bsz, -1, 2)
                b_rotated = torch.view_as_real(b_rotated)
                b_rotated = b_rotated.reshape(bsz, hd)
                B_out[:, t, r, :] = b_rotated.type_as(B)

        return B_out
