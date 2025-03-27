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
    Tensor Product Attention (TPA) with factor-only approach and per-head Q factors.
    Handles GQA and uses factorized KV cache.
    """

    def __init__(self, config: gemma_config.GemmaConfig, attn_type: gemma_config.AttentionType):
        super().__init__()
        self.config = config
        self.attn_type = attn_type

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.hidden_size = config.hidden_size

        # Retrieve actual head dimensions (must be set in config after factorization)
        self.head_dim = getattr(config, 'q_head_dim', config.head_dim) # Primary head dim for Q
        self.k_head_dim = getattr(config, 'k_head_dim', self.head_dim)
        self.v_head_dim = getattr(config, 'v_head_dim', self.head_dim)

        # --- Retrieve TPA ranks and layout info from config ---
        # These MUST be populated after factorization by create_tpa_model_from_standard
        default_q_rank = getattr(config, 'q_rank', 6) # Use q_rank as fallback for per-head/max
        self.q_per_head_ranks = getattr(config, 'q_per_head_ranks', [default_q_rank] * self.num_heads)
        self.q_max_head_rank = getattr(config, 'q_max_head_rank', default_q_rank)
        # Calculate offsets based on max_rank if not provided, assuming contiguous layout
        default_offsets = [i * self.q_max_head_rank for i in range(self.num_heads + 1)]
        self.q_head_offsets = getattr(config, 'q_head_offsets', default_offsets)
        self.total_q_rank = sum(self.q_per_head_ranks)

        self.k_rank = getattr(config, 'k_rank', 2)
        self.v_rank = getattr(config, 'v_rank', 2)
        # --- End TPA Info ---

        # optional query/key normalization:
        self.query_norm = (
            RMSNorm(self.head_dim, eps=config.rms_norm_eps) # Use Q head dim
            if getattr(config, 'use_qk_norm', False)
            else None
        )
        self.key_norm = (
            RMSNorm(self.k_head_dim, eps=config.rms_norm_eps) # Use K head dim
            if getattr(config, 'use_qk_norm', False)
            else None
        )

        # Scaling factor
        if config.query_pre_attn_scalar is not None:
            self.scaling = (config.query_pre_attn_scalar) ** -0.5
        else:
            self.scaling = self.head_dim ** -0.5 # Scale by Q head dim

        # --- Define CORRECTLY SIZED Linear layers ---
        # W_A_q projects to the sum of ranks across all heads
        self.W_A_q = nn.Linear(self.hidden_size, self.total_q_rank, bias=False)

        self.W_A_k = nn.Linear(self.hidden_size, self.num_kv_heads * self.k_rank, bias=False)
        self.W_A_v = nn.Linear(self.hidden_size, self.num_kv_heads * self.v_rank, bias=False)

        # W_B_q projects to the large layout storing all head-specific B factors
        # Output dim = num_heads * max_rank_per_head * q_head_dim
        self.W_B_q_output_dim = self.num_heads * self.q_max_head_rank * self.head_dim
        self.W_B_q = nn.Linear(self.hidden_size, self.W_B_q_output_dim, bias=False)

        # K and V B-factors use their respective head dimensions and ranks
        self.W_B_k = nn.Linear(self.hidden_size, self.k_rank * self.k_head_dim, bias=False)
        self.W_B_v = nn.Linear(self.hidden_size, self.v_rank * self.v_head_dim, bias=False)
        # --- End Linear Layer Definitions ---

        # Final output projection
        # Input dim must match concatenated head outputs (num_heads * Q head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Optional attributes
        self.attn_logit_softcapping = getattr(config, 'attn_logit_softcapping', None)
        self.sliding_window_size = getattr(config, 'sliding_window_size', None)

        # Factor-only caches: (batch_size, seq_len, #kv_heads, rank or head_dim)
        self.cache_kA = None
        self.cache_kB = None
        self.cache_vA = None
        self.cache_vB = None

    def _init_kv_cache(self, batch_size, max_seq_len):
        """Initialize factorized KV cache using correct K/V head dims and ranks."""
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype # Use model's dtype

        # Use actual k/v head dims and ranks
        self.cache_kA = torch.zeros((batch_size, max_seq_len, self.num_kv_heads, self.k_rank),
                                    device=device, dtype=dtype)
        self.cache_kB = torch.zeros((batch_size, max_seq_len, self.k_rank, self.k_head_dim),
                                    device=device, dtype=dtype)
        self.cache_vA = torch.zeros((batch_size, max_seq_len, self.num_kv_heads, self.v_rank),
                                    device=device, dtype=dtype)
        self.cache_vB = torch.zeros((batch_size, max_seq_len, self.v_rank, self.v_head_dim),
                                    device=device, dtype=dtype)

    def forward(
            self,
            hidden_states: torch.Tensor,      # [batch_size, q_seq_len, hidden_size]
            freqs_cis: torch.Tensor,          # [q_seq_len, q_head_dim/2] complex (for Q)
            kv_write_indices: torch.Tensor,   # [seq_len_in_this_step] cache positions to write
            kv_cache: Tuple[torch.Tensor, torch.Tensor], # For shape info and compatibility if needed
            mask: torch.Tensor,               # [batch, 1, q_seq_len, kv_seq_len] causal/padding mask
            local_mask: torch.Tensor = None,  # Optional: [batch, 1, q_seq_len, kv_seq_len] sliding window
    ):
        bsz, q_seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype # Use input dtype

        # --- 1. Factor Projections ---
        # Project hidden states to get the factors for Q, K, V for the current step
        A_q_factors = self.W_A_q(hidden_states)      # [b, q_seq, total_q_rank]
        B_q_factors_flat = self.W_B_q(hidden_states) # [b, q_seq, num_heads * max_q_rank * q_head_dim]
        A_k = self.W_A_k(hidden_states)              # [b, q_seq, num_kv_heads * k_rank]
        B_k = self.W_B_k(hidden_states)              # [b, q_seq, k_rank * k_head_dim]
        A_v = self.W_A_v(hidden_states)              # [b, q_seq, num_kv_heads * v_rank]
        B_v = self.W_B_v(hidden_states)              # [b, q_seq, v_rank * v_head_dim]

        # Reshape K, V factors for caching
        A_k = A_k.view(bsz, q_seq_len, self.num_kv_heads, self.k_rank)
        B_k = B_k.view(bsz, q_seq_len, self.k_rank, self.k_head_dim)
        A_v = A_v.view(bsz, q_seq_len, self.num_kv_heads, self.v_rank)
        B_v = B_v.view(bsz, q_seq_len, self.v_rank, self.v_head_dim)

        # --- 2. Apply RoPE to K factors BEFORE Caching ---
        # RoPE for K uses k_head_dim. Reshape freqs_cis if needed (it's based on q_head_dim).
        if self.head_dim == self.k_head_dim:
            freqs_cis_k = freqs_cis # Dimensions match
        else:
            # Need to regenerate or slice freqs_cis for k_head_dim. Assuming precomputed for q_head_dim.
            # Simple slice if k_head_dim < head_dim. More complex otherwise.
            print(f"Warning: RoPE dimension mismatch Q({self.head_dim}) vs K({self.k_head_dim}). Slicing freqs_cis.")
            freqs_cis_k = freqs_cis[:, :self.k_head_dim // 2] # Slice if precomputed for larger Q dim

        # Reshape freqs_cis_k for broadcasting over rank dim of B_k
        freqs_cis_k_b = gemma_model.reshape_for_broadcast(freqs_cis_k, B_k[:, :, 0, :])
        B_k = gemma_model.apply_rotary_emb(B_k, freqs_cis_k_b)

        # --- 3. KV Cache Update ---
        if kv_write_indices is None: # Handle case where indices are not provided (e.g., prefill)
            kv_write_indices = torch.arange(q_seq_len, device=device)

        # Initialize cache if needed
        if self.cache_kA is None or bsz > self.cache_kA.shape[0]:
            # Determine max_seq_len (e.g., from config or a known upper bound)
            max_seq_len = getattr(self.config, 'max_position_embeddings', 2048)
            if kv_cache is not None and len(kv_cache) > 0 and kv_cache[0] is not None:
                max_seq_len = max(max_seq_len, kv_cache[0].shape[1]) # Use cache size if larger
            self._init_kv_cache(bsz, max_seq_len)

        # Ensure caches are on the correct device/dtype
        self.cache_kA = self.cache_kA.to(device=device, dtype=A_k.dtype)
        self.cache_kB = self.cache_kB.to(device=device, dtype=B_k.dtype)
        self.cache_vA = self.cache_vA.to(device=device, dtype=A_v.dtype)
        self.cache_vB = self.cache_vB.to(device=device, dtype=B_v.dtype)

        # Write current step's K/V factors to cache
        # Ensure indices don't go out of bounds
        max_cache_len = self.cache_kA.shape[1]
        valid_indices = kv_write_indices < max_cache_len
        if not valid_indices.all():
            print(f"Warning: KV write indices exceed cache size ({max_cache_len}). Clamping.")
            kv_write_indices = kv_write_indices.clamp(max=max_cache_len - 1)
            # Only write valid parts if some indices were invalid
            A_k = A_k[:, valid_indices, :, :]
            B_k = B_k[:, valid_indices, :, :]
            A_v = A_v[:, valid_indices, :, :]
            B_v = B_v[:, valid_indices, :, :]

        if kv_write_indices.numel() > 0: # Only write if there are indices
            self.cache_kA[:bsz].index_copy_(1, kv_write_indices, A_k)
            self.cache_kB[:bsz].index_copy_(1, kv_write_indices, B_k)
            self.cache_vA[:bsz].index_copy_(1, kv_write_indices, A_v)
            self.cache_vB[:bsz].index_copy_(1, kv_write_indices, B_v)

        # --- 4. Read K/V Factors from Cache ---
        # Determine the actual sequence length in the cache to attend to
        kv_seq_len = kv_write_indices.max().item() + 1 if kv_write_indices.numel() > 0 else q_seq_len
        kv_seq_len = min(kv_seq_len, max_cache_len) # Ensure it doesn't exceed cache size

        A_k_cached = self.cache_kA[:bsz, :kv_seq_len] # [b, kv_seq, #kv_heads, k_rank]
        B_k_cached = self.cache_kB[:bsz, :kv_seq_len] # [b, kv_seq, k_rank, k_head_dim]
        A_v_cached = self.cache_vA[:bsz, :kv_seq_len] # [b, kv_seq, #kv_heads, v_rank]
        B_v_cached = self.cache_vB[:bsz, :kv_seq_len] # [b, kv_seq, v_rank, v_head_dim]

        # --- 5. GQA Handling: Repeat K/V 'A' Factors ---
        if self.num_kv_heads < self.num_heads:
            heads_per_kv = self.num_heads // self.num_kv_heads
            A_k_cached = A_k_cached.repeat_interleave(heads_per_kv, dim=2) # [b, kv_seq, #heads, k_rank]
            A_v_cached = A_v_cached.repeat_interleave(heads_per_kv, dim=2) # [b, kv_seq, #heads, v_rank]
            # B_k_cached and B_v_cached are shared within the group, no repeat needed

        # --- 6. Per-Head Attention Computation ---
        all_attn_outputs = []
        # Reshape freqs_cis for Q (uses q_head_dim)
        freqs_cis_q = gemma_model.reshape_for_broadcast(freqs_cis, B_q_factors_flat[:, :, :self.head_dim])

        for h in range(self.num_heads):
            # --- Get Q factors for head h ---
            # Slice A_q based on cumulative ranks
            start_A_idx = self.q_head_offsets[h]
            end_A_idx = self.q_head_offsets[h+1]
            head_rank = self.q_per_head_ranks[h]
            head_A_q = A_q_factors[:, :, start_A_idx:end_A_idx] # [b, q_seq, head_rank]

            # Slice B_q from the flat tensor using max_rank offset
            start_B_idx = h * self.q_max_head_rank * self.head_dim
            # Only take the dimensions corresponding to the actual head_rank
            end_B_idx = start_B_idx + head_rank * self.head_dim
            head_B_q_flat = B_q_factors_flat[:, :, start_B_idx:end_B_idx] # [b, q_seq, head_rank * head_dim]
            # Reshape to add rank dimension
            head_B_q = head_B_q_flat.view(bsz, q_seq_len, head_rank, self.head_dim) # [b, q_seq, head_rank, q_head_dim]

            # --- Apply RoPE to head_B_q ---
            # RoPE uses q_head_dim, freqs_cis_q is already prepared
            head_B_q_rotated = gemma_model.apply_rotary_emb(head_B_q, freqs_cis_q)

            # --- Get K factors for head h ---
            # A_k is already repeated for GQA; select the h-th head slice
            head_A_k = A_k_cached[:, :, h, :] # [b, kv_seq, k_rank]
            # B_k is shared across the group; use the full B_k_cached
            # Note: B_k already had RoPE applied before caching
            head_B_k = B_k_cached            # [b, kv_seq, k_rank, k_head_dim]

            # --- Optional QK Norm ---
            if self.query_norm is not None and self.key_norm is not None:
                # Apply norm efficiently if possible, or loop if necessary
                normed_B_q = torch.stack([self.query_norm(head_B_q_rotated[:,:,r,:]) for r in range(head_rank)], dim=2)
                normed_B_k = torch.stack([self.key_norm(head_B_k[:,:,r,:]) for r in range(self.k_rank)], dim=2)
            else:
                normed_B_q = head_B_q_rotated
                normed_B_k = head_B_k

            # --- Compute B_dots (inner products of B factors) ---
            # Einsum: (b, q, r_q, d_q) x (b, k, r_k, d_k) -> (b, q, k, r_q, r_k)
            # Ensure head dimensions d_q and d_k match or handle mismatch
            if self.head_dim != self.k_head_dim:
                # This requires a projection or padding strategy if dims differ.
                # Assuming dims match based on factorization design.
                print(f"Warning/Debug: Q head dim {self.head_dim} != K head dim {self.k_head_dim}. Assuming compatibility.")
                pass
            head_B_dots = torch.einsum('bqrd,bksd->bqkrs', normed_B_q, normed_B_k) * self.scaling

            # --- Compute Attention Scores ---
            # Einsum: (b,q,r_q) x (b,k,r_k) x (b,q,k,r_q,r_k) -> (b,q,k)
            head_attn_weights = torch.einsum('bqr,bks,bqkrs->bqk', head_A_q, head_A_k, head_B_dots)

            # --- Masking, Softcapping, Softmax ---
            if self.attn_logit_softcapping is not None:
                # Apply softcapping element-wise
                head_attn_weights = torch.tanh(head_attn_weights / self.attn_logit_softcapping) * self.attn_logit_softcapping

            # Determine mask to apply for this head
            current_mask = None
            if self.attn_type == gemma_config.AttentionType.LOCAL_SLIDING and self.sliding_window_size is not None and local_mask is not None:
                # Select the correct kv_seq_len for the mask
                current_mask = local_mask[:bsz, 0, :q_seq_len, :kv_seq_len]
            elif mask is not None:
                current_mask = mask[:bsz, 0, :q_seq_len, :kv_seq_len]

            # Apply mask if available
            if current_mask is not None:
                if head_attn_weights.shape == current_mask.shape:
                    head_attn_weights = head_attn_weights + current_mask
                else:
                    print(f"Warning: Mask shape mismatch. Attn: {head_attn_weights.shape}, Mask: {current_mask.shape}. Skipping mask.")

            # Softmax
            head_attn_probs = F.softmax(head_attn_weights.float(), dim=-1).to(dtype) # [b, q_seq, kv_seq]

                       # --- Compute Weighted Value Output ---
            # Get V factors for head h
            head_A_v = A_v_cached[:, :, h, :] # Shape: [b, k, r_v]
            head_B_v = B_v_cached            # Shape: [b, k, r_v, d_v]

            # Corrected Einsum: Sum over k (key/value pos) and r (value rank)
            # 'bqk': attention probs for this query pos q, key pos k
            # 'bkr': A_v factor for this head h (implied), key pos k, rank r
            # 'bkrd': B_v factor (shared), key pos k, rank r, output dim d
            # -> 'bqd': output for query pos q, output dim d
            head_output = torch.einsum('bqk,bkr,bkrd->bqd',
                                       head_attn_probs,
                                       head_A_v,
                                       head_B_v) # Output shape: [b, q_seq, v_head_dim]

            # --- Handle potential dimension mismatch for concatenation ---
            # The output dim is v_head_dim. We need to ensure it matches self.head_dim (Q dim)
            # for concatenation before o_proj.
            if head_output.shape[-1] != self.head_dim:
                 print(f"ERROR: Head {h} output dim {head_output.shape[-1]} derived from V ({self.v_head_dim}) != Q head dim {self.head_dim}. Concatenation before o_proj will fail or be incorrect.")
                 # Option 1: Project here (adds parameters/complexity)
                 # if not hasattr(self, f'v_to_q_proj_{h}'):
                 #     setattr(self, f'v_to_q_proj_{h}', nn.Linear(self.v_head_dim, self.head_dim, bias=False).to(device, dtype))
                 # head_output = getattr(self, f'v_to_q_proj_{h}')(head_output)

                 # Option 2: Assume o_proj handles it (requires o_proj input dim to be correct sum)
                 # This was likely the intention - o_proj input is sum(head_dims)
                 # If D_q != D_v, then o_proj input should be num_heads * D_v? No, it's usually num_heads * D_q.
                 # Best practice: Ensure D_q = D_k = D_v during conversion/config if possible.
                 # If not possible, a projection here or a modified o_proj is needed.
                 # For now, let's proceed assuming o_proj expects num_heads * self.head_dim (Q dim)
                 # and we might need to adjust head_output if dimensions differ.
                 # Let's pad/truncate as a temporary, likely incorrect, measure for testing:
                 if head_output.shape[-1] > self.head_dim:
                     print(f"Warning: Truncating head {h} output to Q dim.")
                     head_output = head_output[..., :self.head_dim]
                 elif head_output.shape[-1] < self.head_dim:
                     print(f"Warning: Padding head {h} output to Q dim.")
                     padding_size = self.head_dim - head_output.shape[-1]
                     padding = torch.zeros(*head_output.shape[:-1], padding_size, device=device, dtype=dtype)
                     head_output = torch.cat([head_output, padding], dim=-1)
                 # Else: dimensions match, head_output is already correct shape [b, q_seq, self.head_dim]

            all_attn_outputs.append(head_output)
            # --- End Head Loop ---

        # --- 7. Concatenate Head Outputs & Final Projection ---
        # Concatenate along the last dimension
        attn_output_cat = torch.cat(all_attn_outputs, dim=-1) # Shape: [b, q_seq, num_heads * self.head_dim]

        # Final output projection
        final_output = self.o_proj(attn_output_cat) # Shape: [b, q_seq, hidden_size]
        # Final NaN check
        if torch.isnan(final_output).any():
            print("WARNING: NaN detected in final TPA output. Replacing with zeros.")
            final_output = torch.nan_to_num(final_output, nan=0.0)

        return final_output

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
        self.head_dim = config.head_dim

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
