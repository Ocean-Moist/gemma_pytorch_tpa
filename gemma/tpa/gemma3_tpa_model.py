"""Inference-only Gemma model implementation with Tensor Product Attention (TPA)."""

import torch
import os
import json
import gc
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Sequence, Tuple, Union, Mapping, Any

from .. import model as gemma_model
from .. import config as gemma_config
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


class TPAAttention(nn.Module):
    """Tensor Product Attention module for Gemma."""
    def __init__(self, config: gemma_config.GemmaConfig, attn_type: gemma_config.AttentionType):
        super().__init__()
        self.config = config
        self.attn_type = attn_type

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads if hasattr(config, 'num_key_value_heads') else self.num_heads

        self.hidden_size = config.hidden_size
        
        # Derive head_dim from hidden_size and num_heads if not explicitly provided
        # This is critical for models with non-standard head dimensions
        if hasattr(config, 'head_dim'):
            self.head_dim = config.head_dim
        else:
            # For most models, head_dim = hidden_size / num_heads
            # But some models might have different proportions
            if hasattr(config, 'q_head_dim'):
                # If q_head_dim is explicitly provided, use it
                self.head_dim = config.q_head_dim
            else:
                # Default calculation
                self.head_dim = self.hidden_size // self.num_heads
        
        # Note: The true head dimension might vary between Q, K, and V in some models
        # So we'll store them separately if provided
        self.q_head_dim = getattr(config, 'q_head_dim', self.head_dim)
        self.k_head_dim = getattr(config, 'k_head_dim', self.head_dim)
        self.v_head_dim = getattr(config, 'v_head_dim', self.head_dim)

        # TPA specific parameters
        self.q_rank = getattr(config, 'q_rank', 6)
        self.k_rank = getattr(config, 'k_rank', 2)
        self.v_rank = getattr(config, 'v_rank', 2)

        # Debug info about dimensions
        print(f"TPAAttention: hidden_size={self.hidden_size}")
        print(f"TPAAttention: num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}")
        print(f"TPAAttention: head_dims: q={self.q_head_dim}, k={self.k_head_dim}, v={self.v_head_dim} (default={self.head_dim})")
        print(f"TPAAttention: ranks: q={self.q_rank}, k={self.k_rank}, v={self.v_rank}")

        # Scaling for attention
        if config.query_pre_attn_scalar is not None:
            self.scaling = config.query_pre_attn_scalar**-0.5
        else:
            self.scaling = self.head_dim**-0.5

        # Define TPA projection matrices with appropriate dimensions
        # W_A projections: hidden_size -> num_heads * rank
        self.W_A_q = nn.Linear(self.hidden_size, self.num_heads * self.q_rank, bias=False)
        self.W_A_k = nn.Linear(self.hidden_size, self.num_kv_heads * self.k_rank, bias=False)
        self.W_A_v = nn.Linear(self.hidden_size, self.num_kv_heads * self.v_rank, bias=False)

        # W_B projections: hidden_size -> rank * head_dim
        self.W_B_q = nn.Linear(self.hidden_size, self.q_rank * self.head_dim, bias=False)
        self.W_B_k = nn.Linear(self.hidden_size, self.k_rank * self.head_dim, bias=False)
        self.W_B_v = nn.Linear(self.hidden_size, self.v_rank * self.head_dim, bias=False)

        # Output projection
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Optional normalization
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

        # Softcapping parameters
        self.attn_logit_softcapping = getattr(config, 'attn_logit_softcapping', None)
        self.sliding_window_size = getattr(config, 'sliding_window_size', None)

        # Flag for using factorized weights
        self.use_factorized_weights = False

        # Initialize KV cache parameters with batch_size, seq_len, num_heads, rank/head_dim
        # These will be created as needed during inference
        self.cache_kA = None
        self.cache_kB = None
        self.cache_vA = None
        self.cache_vB = None

    def _init_kv_cache(self, batch_size, max_seq_len):
        """Initialize KV cache for TPA attention."""
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # For TPA, we need to cache both A and B components for K and V matrices
        # This is a key difference from standard attention where we cache final K and V

        # Ensure that batch_size and max_seq_len are proper integers
        batch_size = int(batch_size)
        max_seq_len = int(max_seq_len)

        try:
            # Shape for A components: [batch_size, max_seq_len, num_kv_heads, rank]
            self.cache_kA = torch.zeros(
                (batch_size, max_seq_len, self.num_kv_heads, self.k_rank),
                device=device, dtype=dtype
            )
            self.cache_vA = torch.zeros(
                (batch_size, max_seq_len, self.num_kv_heads, self.v_rank),
                device=device, dtype=dtype
            )

            # Shape for B components: [batch_size, max_seq_len, rank, head_dim]
            self.cache_kB = torch.zeros(
                (batch_size, max_seq_len, self.k_rank, self.head_dim),
                device=device, dtype=dtype
            )
            self.cache_vB = torch.zeros(
                (batch_size, max_seq_len, self.v_rank, self.head_dim),
                device=device, dtype=dtype
            )
        except RuntimeError as e:
            # Handle out-of-memory scenarios by reducing dimensions if possible
            print(f"Warning: Error creating KV cache with dimensions [batch={batch_size}, seq={max_seq_len}]. Error: {e}")
            print("Trying to create smaller cache...")

            # Reduce batch size if needed
            adjusted_batch_size = max(1, batch_size // 2)
            # Reduce sequence length if needed
            adjusted_seq_len = max(64, max_seq_len // 2)

            print(f"Creating cache with adjusted dimensions: batch={adjusted_batch_size}, seq={adjusted_seq_len}")

            self.cache_kA = torch.zeros(
                (adjusted_batch_size, adjusted_seq_len, self.num_kv_heads, self.k_rank),
                device=device, dtype=dtype
            )
            self.cache_vA = torch.zeros(
                (adjusted_batch_size, adjusted_seq_len, self.num_kv_heads, self.v_rank),
                device=device, dtype=dtype
            )

            self.cache_kB = torch.zeros(
                (adjusted_batch_size, adjusted_seq_len, self.k_rank, self.head_dim),
                device=device, dtype=dtype
            )
            self.cache_vB = torch.zeros(
                (adjusted_batch_size, adjusted_seq_len, self.v_rank, self.head_dim),
                device=device, dtype=dtype
            )

    def forward(
            self,
            hidden_states: torch.Tensor,
            freqs_cis: torch.Tensor,
            kv_write_indices: torch.Tensor,
            kv_cache: Tuple[torch.Tensor, torch.Tensor],
            mask: torch.Tensor,
            local_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        print(f"DEBUG TPA ATTENTION: Input hidden_states shape: {hidden_states.shape}, mean: {hidden_states.mean().item():.6f}, std: {hidden_states.std().item():.6f}")
        
        # A projections
        A_q = self.W_A_q(hidden_states).view(batch_size, seq_len, self.num_heads, self.q_rank)
        A_k = self.W_A_k(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.k_rank)
        A_v = self.W_A_v(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.v_rank)
        
        print(f"DEBUG TPA ATTENTION: After A projections - A_q: {A_q.shape}, mean: {A_q.mean().item():.6f}, std: {A_q.std().item():.6f}")
        print(f"DEBUG TPA ATTENTION: After A projections - A_k: {A_k.shape}, mean: {A_k.mean().item():.6f}, std: {A_k.std().item():.6f}")
        print(f"DEBUG TPA ATTENTION: After A projections - A_v: {A_v.shape}, mean: {A_v.mean().item():.6f}, std: {A_v.std().item():.6f}")

        # B projections
        B_q = self.W_B_q(hidden_states).reshape(batch_size, seq_len, self.q_rank, self.head_dim)
        B_k = self.W_B_k(hidden_states).reshape(batch_size, seq_len, self.k_rank, self.head_dim)
        B_v = self.W_B_v(hidden_states).reshape(batch_size, seq_len, self.v_rank, self.head_dim)
        
        print(f"DEBUG TPA ATTENTION: After B projections - B_q: {B_q.shape}, mean: {B_q.mean().item():.6f}, std: {B_q.std().item():.6f}")
        print(f"DEBUG TPA ATTENTION: After B projections - B_k: {B_k.shape}, mean: {B_k.mean().item():.6f}, std: {B_k.std().item():.6f}")
        print(f"DEBUG TPA ATTENTION: After B projections - B_v: {B_v.shape}, mean: {B_v.mean().item():.6f}, std: {B_v.std().item():.6f}")

        # Apply rotary positional embedding to B_q and B_k
        def apply_rotary_emb_to_B(x, freqs_cis):
            # x shape: [batch, seq_len, rank, head_dim]
            batch_size, seq_len, rank, head_dim = x.shape

            # We need to ensure head_dim is even (required for complex representation)
            if head_dim % 2 != 0:
                raise ValueError(f"Head dimension {head_dim} must be even for RoPE")

            # Process each rank separately to avoid dimension mixing
            output = torch.zeros_like(x)

            for r in range(rank):
                # Extract just this rank: [batch, seq_len, head_dim]
                x_r = x[:, :, r, :]

                # Apply standard RoPE
                x_complex = torch.view_as_complex(
                    torch.stack(torch.chunk(x_r.float(), 2, dim=-1), dim=-1)
                )

                x_rotated = torch.view_as_real(x_complex * freqs_cis)

                # Reshape back to [batch, seq_len, head_dim]
                x_rotated = torch.cat(torch.chunk(x_rotated, 2, dim=-1), dim=-2)
                x_rotated = x_rotated.reshape(batch_size, seq_len, head_dim)

                # Store the result in the original rank's position
                output[:, :, r, :] = x_rotated

            return output.type_as(x)

        # Apply RoPE to B tensors
        B_q = apply_rotary_emb_to_B(B_q, freqs_cis)
        B_k = apply_rotary_emb_to_B(B_k, freqs_cis)

        # Handle KV cache - create if it doesn't exist
        if self.cache_kA is None or batch_size > self.cache_kA.shape[0]:
            self._init_kv_cache(batch_size, kv_cache[0].shape[1])

        # Write to cache
        self.cache_kA = self.cache_kA.to(A_k.device, A_k.dtype)
        self.cache_vA = self.cache_vA.to(A_v.device, A_v.dtype)
        self.cache_kB = self.cache_kB.to(B_k.device, B_k.dtype)
        self.cache_vB = self.cache_vB.to(B_v.device, B_v.dtype)

        self.cache_kA[:batch_size].index_copy_(1, kv_write_indices, A_k)
        self.cache_vA[:batch_size].index_copy_(1, kv_write_indices, A_v)
        self.cache_kB[:batch_size].index_copy_(1, kv_write_indices, B_k)
        self.cache_vB[:batch_size].index_copy_(1, kv_write_indices, B_v)

        # Get the KV cache sections up to the current position
        cache_len = kv_write_indices[-1] + 1 if kv_write_indices.numel() > 0 else 0

        A_k = self.cache_kA[:batch_size, :cache_len]
        A_v = self.cache_vA[:batch_size, :cache_len]
        B_k = self.cache_kB[:batch_size, :cache_len]
        B_v = self.cache_vB[:batch_size, :cache_len]

        # Define kv_seq_len for reshaping
        kv_seq_len = A_k.size(1)

        # Handle GQA if num_kv_heads < num_heads
        if self.num_kv_heads < self.num_heads:
            # Expand A_k and A_v for GQA
            # Each kv_head serves multiple query heads
            heads_per_kv = self.num_heads // self.num_kv_heads
            A_k = A_k.repeat_interleave(heads_per_kv, dim=2)
            A_v = A_v.repeat_interleave(heads_per_kv, dim=2)

        # Factor-only approach: compute attention scores directly without materializing full Q, K, V
        print("DEBUG TPA ATTENTION: Using factor-only computation to reduce memory usage")
        
        # 1. Ensure factors are properly shaped for direct computation
        # [batch, seq, num_heads, rank] for A factors
        # [batch, seq, rank, head_dim] for B factors
        A_q = A_q.view(batch_size, seq_len, self.num_heads, self.q_rank)
        A_k = A_k.view(batch_size, kv_seq_len, self.num_heads, self.k_rank)
        A_v = A_v.view(batch_size, kv_seq_len, self.num_heads, self.v_rank)
        
        B_q = B_q.view(batch_size, seq_len, self.q_rank, self.head_dim)
        B_k = B_k.view(batch_size, kv_seq_len, self.k_rank, self.head_dim)
        B_v = B_v.view(batch_size, kv_seq_len, self.v_rank, self.head_dim)
        
        # 2. Apply query/key normalization to B factors if needed
        # Note: In factor-only approach, we apply normalization to B factors directly
        if self.query_norm is not None and self.key_norm is not None:
            print("DEBUG TPA ATTENTION: Applying query/key normalization to B factors")
            for r in range(self.q_rank):
                B_q[:, :, r, :] = self.query_norm(B_q[:, :, r, :])
            for r in range(self.k_rank):
                B_k[:, :, r, :] = self.key_norm(B_k[:, :, r, :])
        
        # 3. Create storage for attention weights
        attn_weights = torch.zeros(
            (batch_size, self.num_heads, seq_len, kv_seq_len),
            device=hidden_states.device, 
            dtype=hidden_states.dtype
        )
        
        # 4. Precompute all B dot products between query and key factors
        # This is the key optimization - compute once and reuse across heads
        print("DEBUG TPA ATTENTION: Precomputing B factor dot products...")
        
        # Einsum for efficient B dot products computation: [batch, q_len, q_rank, head_dim] x [batch, kv_len, k_rank, head_dim]
        # Resulting shape: [batch, q_len, kv_len, q_rank, k_rank]
        B_dots = torch.zeros(
            (batch_size, seq_len, kv_seq_len, self.q_rank, self.k_rank),
            device=hidden_states.device,
            dtype=hidden_states.dtype
        )
        
        # Compute dot products for all rank combinations
        for r in range(self.q_rank):
            for s in range(self.k_rank):
                # Extract B factors for this rank pair
                # B_q_r: [batch, seq_len, head_dim]
                # B_k_s: [batch, kv_seq_len, head_dim]
                B_q_r = B_q[:, :, r, :]
                B_k_s = B_k[:, :, s, :]
                
                # Compute dot product using einsum for all query-key pairs
                # Result: [batch, seq_len, kv_seq_len]
                B_dots[:, :, :, r, s] = torch.einsum('bqd,bkd->bqk', B_q_r, B_k_s)
        
        # Apply scaling to B_dots 
        B_dots = B_dots * (self.scaling / (self.q_rank * self.k_rank))
                
        # 5. Compute attention weights for each head using factorized form
        print("DEBUG TPA ATTENTION: Computing attention scores from factorized representation...")
        for h in range(self.num_heads):
            # Initialize per-head attention weights
            head_weights = torch.zeros(
                (batch_size, seq_len, kv_seq_len),
                device=hidden_states.device,
                dtype=hidden_states.dtype
            )
            
            # Sum over all rank combinations
            for r in range(self.q_rank):
                for s in range(self.k_rank):
                    # A_q_h_r: [batch, seq_len]
                    # A_k_h_s: [batch, kv_seq_len]
                    A_q_h_r = A_q[:, :, h, r]
                    A_k_h_s = A_k[:, :, h, s]
                    
                    # Compute outer product of A factors: [batch, seq_len, kv_seq_len]
                    A_product = torch.einsum('bq,bk->bqk', A_q_h_r, A_k_h_s)
                    
                    # Multiply with precomputed B dot products and accumulate
                    head_weights += A_product * B_dots[:, :, :, r, s]
            
            # Store result for this head
            attn_weights[:, h, :, :] = head_weights
            
        print(f"DEBUG TPA ATTENTION: factor-only attn_weights: {attn_weights.shape}, mean: {attn_weights.mean().item():.6f}, std: {attn_weights.std().item():.6f}")

        # Apply softcapping if configured
        if self.attn_logit_softcapping is not None:
            attn_weights = attn_weights / self.attn_logit_softcapping
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * self.attn_logit_softcapping

        # Apply sliding window mask if needed
        if (
                self.attn_type == gemma_config.AttentionType.LOCAL_SLIDING
                and self.sliding_window_size is not None
                and local_mask is not None
        ):
            mask = local_mask

        # Apply attention mask
        # First ensure mask has the right shape to match attn_weights
        if mask is not None:
            # Get shapes
            attn_shape = attn_weights.shape
            mask_shape = mask.shape
            
            # Check if shapes don't match in the last dimension
            if mask_shape[3] != attn_shape[3]:
                # Slice the mask to match the KV sequence length
                mask = mask[:, :, :attn_shape[2], :attn_shape[3]]
        
        attn_weights = attn_weights + mask

        # Apply softmax
        attn_probs = F.softmax(attn_weights.float(), dim=-1).type_as(hidden_states)
        print(f"DEBUG TPA ATTENTION: attn_probs: {attn_probs.shape}, mean: {attn_probs.mean().item():.6f}, std: {attn_probs.std().item():.6f}")
        
        # Check if there are NaN values in attention probs (which would propagate through)
        has_nan = torch.isnan(attn_probs).any().item()
        print(f"DEBUG TPA ATTENTION: NaN values in attn_probs: {has_nan}")
        
        if has_nan:
            print("WARNING: Found NaN values in attention probabilities! Replacing with zeros...")
            attn_probs = torch.nan_to_num(attn_probs, nan=0.0)
            # Renormalize
            attn_probs = attn_probs / (attn_probs.sum(dim=-1, keepdim=True) + 1e-6)

        # 6. Compute the attention output directly from factorized V
        # Create output container
        attn_output = torch.zeros(
            (batch_size, self.num_heads, seq_len, self.head_dim),
            device=hidden_states.device, 
            dtype=hidden_states.dtype
        )
        
        print("DEBUG TPA ATTENTION: Computing attention output using factorized values...")
        
        # For each head
        for h in range(self.num_heads):
            # For each output position
            for t in range(seq_len):
                # For each rank of V
                for u in range(self.v_rank):
                    # Compute weighted sum of A_v factors for this rank and head
                    # attn_probs[b, h, t, τ] * A_v[b, τ, h, u]
                    weighted_A_v = torch.einsum(
                        'bhtk,bkhu->bht', 
                        attn_probs[:, h:h+1, t:t+1, :], 
                        A_v
                    )  # [batch, 1, 1]
                    
                    # Compute weighted sum of V vectors for this rank
                    # We need to compute: sum_τ( attn_probs[t,τ] * A_v[τ,u] * B_v[τ,u] )
                    weighted_sum = torch.zeros(
                        (batch_size, self.head_dim),
                        device=hidden_states.device, 
                        dtype=hidden_states.dtype
                    )
                    
                    # This computes the weighted sum across all tokens
                    for k in range(kv_seq_len):
                        # Weight for this token: attn_probs[b, h, t, k] * A_v[b, k, h, u]
                        weight = attn_probs[:, h, t, k].unsqueeze(-1) * A_v[:, k, h, u].unsqueeze(-1)
                        # Accumulate B_v[b, k, u, d] weighted by above
                        weighted_sum += weight * B_v[:, k, u, :]
                    
                    # Add to output for this head and position
                    attn_output[:, h, t, :] += weighted_sum / self.v_rank
        
        print(f"DEBUG TPA ATTENTION: factor-based attn_output: {attn_output.shape}, mean: {attn_output.mean().item():.6f}, std: {attn_output.std().item():.6f}")

        # Reshape to [batch, seq, num_heads*head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        print(f"DEBUG TPA ATTENTION: reshaped attn_output: {attn_output.shape}, mean: {attn_output.mean().item():.6f}, std: {attn_output.std().item():.6f}")

        # Apply output projection
        output = self.o_proj(attn_output)
        print(f"DEBUG TPA ATTENTION: final output: {output.shape}, mean: {output.mean().item():.6f}, std: {output.std().item():.6f}")

        # Check if NaN values appeared in the output
        has_nan = torch.isnan(output).any().item()
        if has_nan:
            print("CRITICAL ERROR: NaN values in attention output!")
            # Replace NaN values with zeros as a fallback
            output = torch.nan_to_num(output, nan=0.0)

        return output


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
        self._register_freqs_cis('local_freqs_cis', head_dim, max_seq_len, theta=rope_lengths.get(
            gemma_config.AttentionType.LOCAL_SLIDING, defaults[gemma_config.AttentionType.LOCAL_SLIDING]
        ))
        self._register_freqs_cis('global_freqs_cis', head_dim, max_seq_len, theta=rope_lengths.get(
            gemma_config.AttentionType.GLOBAL, defaults[gemma_config.AttentionType.GLOBAL]
        ), rope_scaling_factor=config.rope_scaling_factor)

    def _register_freqs_cis(
            self, name: str, head_dim: int, max_seq_len: int, theta: int = 10_000, rope_scaling_factor: int = 1
    ):
        self.register_buffer(
            name, gemma_model.precompute_freqs_cis(head_dim, max_seq_len * 2, theta=theta, rope_scaling_factor=rope_scaling_factor)
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
        print(f"DEBUG TPA FORWARD: input_token_ids shape: {input_token_ids.shape}, device: {input_token_ids.device}")
        print(f"DEBUG TPA FORWARD: input_positions: {input_positions}, device: {input_positions.device}")
        
        # Ensure freqs_cis buffers are on the same device as input_positions
        if self.local_freqs_cis.device != input_positions.device:
            print(f"Moving freqs_cis from {self.local_freqs_cis.device} to {input_positions.device}")
            self.local_freqs_cis = self.local_freqs_cis.to(input_positions.device)
            self.global_freqs_cis = self.global_freqs_cis.to(input_positions.device)
        
        freqs_cis = {}
        freqs_cis[gemma_config.AttentionType.LOCAL_SLIDING] = (
            self.local_freqs_cis.index_select(0, input_positions)
        )
        freqs_cis[gemma_config.AttentionType.GLOBAL] = (
            self.global_freqs_cis.index_select(0, input_positions)
        )
        
        print(f"DEBUG TPA FORWARD: freqs_cis shapes - LOCAL: {freqs_cis[gemma_config.AttentionType.LOCAL_SLIDING].shape}, GLOBAL: {freqs_cis[gemma_config.AttentionType.GLOBAL].shape}")
        
        # Ensure the embedder is on the same device as input_token_ids
        if hasattr(self.text_token_embedder, 'weight') and self.text_token_embedder.weight.device != input_token_ids.device:
            print(f"Moving embedding weights from {self.text_token_embedder.weight.device} to {input_token_ids.device}")
            # Move the entire embedder module to the target device
            self.text_token_embedder = self.text_token_embedder.to(input_token_ids.device)
        
        hidden_states = self.text_token_embedder(input_token_ids)
        print(f"DEBUG TPA FORWARD: After embedding, hidden_states shape: {hidden_states.shape}, mean: {hidden_states.mean().item():.6f}, std: {hidden_states.std().item():.6f}")
        
        # Ensure the entire model is on the correct device
        input_device = input_token_ids.device
        if next(self.model.parameters()).device != input_device:
            print(f"Moving model from {next(self.model.parameters()).device} to {input_device}")
            self.model = self.model.to(input_device)
            # Also move the sampler to ensure the final output is on the right device
            self.sampler = self.sampler.to(input_device)
        
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = hidden_states * normalizer
        print(f"DEBUG TPA FORWARD: After normalization, hidden_states shape: {hidden_states.shape}, mean: {hidden_states.mean().item():.6f}, std: {hidden_states.std().item():.6f}")

        kv_write_indices = input_positions
        
        if kv_caches:
            print(f"DEBUG TPA FORWARD: kv_caches length: {len(kv_caches)}")
            print(f"DEBUG TPA FORWARD: First KV cache shapes: k={kv_caches[0][0].shape}, v={kv_caches[0][1].shape}")
        
        if mask is not None:
            print(f"DEBUG TPA FORWARD: mask shape: {mask.shape}, device: {mask.device}")
        
        hidden_states = self.model(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_caches=kv_caches,
            mask=mask,
            local_mask=local_mask,
        )
        
        print(f"DEBUG TPA FORWARD: After model forward, hidden_states shape: {hidden_states.shape}, mean: {hidden_states.mean().item():.6f}, std: {hidden_states.std().item():.6f}")
        
        embedder_weight = self.text_token_embedder.weight
        print(f"DEBUG TPA FORWARD: embedder_weight shape: {embedder_weight.shape}, mean: {embedder_weight.mean().item():.6f}, std: {embedder_weight.std().item():.6f}")
        
        if self.config.quant:
            embedder_weight = (
                    embedder_weight * self.text_token_embedder.weight_scaler.unsqueeze(-1))
            print(f"DEBUG TPA FORWARD: After quant, embedder_weight shape: {embedder_weight.shape}, mean: {embedder_weight.mean().item():.6f}, std: {embedder_weight.std().item():.6f}")

        print(f"DEBUG TPA FORWARD: output_positions: {output_positions}")
        
        next_tokens, logits = self.sampler(
            embedding=embedder_weight,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
        )
        
        print(f"DEBUG TPA FORWARD: logits shape: {logits.shape}, mean: {logits.mean().item():.6f}, std: {logits.std().item():.6f}")
        print(f"DEBUG TPA FORWARD: Top 10 logits: {torch.topk(logits, min(10, logits.shape[-1]), dim=-1).values[0].tolist()}")
        
        return next_tokens, logits

    def create_attention_mask(self, input_ids, sequence_length):
        """
        Creates a causal attention mask (standard for auto-regressive models).
        """
        batch_size = input_ids.shape[0]
        # Create causal mask
        # [batch_size, 1, sequence_length, sequence_length]
        mask = torch.tril(torch.ones((batch_size, 1, sequence_length, sequence_length), dtype=torch.bool, device=input_ids.device))

        # Create local mask for sliding window attention if needed
        if self.config.sliding_window_size is not None:
            # Create mask for local sliding window attention
            # Logical AND between causal mask and sliding window mask
            local_mask = torch.logical_and(
                mask,
                torch.triu(
                    torch.ones((1, 1, sequence_length, sequence_length), dtype=torch.bool, device=input_ids.device),
                    diagonal=-(self.config.sliding_window_size-1)
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
        """Generates responses for given prompts using Gemma model."""

        # Determine if device is provided, otherwise use model device
        if device is None:
            device = next(self.parameters()).device

        # Handle different prompt formats
        if isinstance(prompts, str):
            prompts = [prompts]
            single_prompt = True
        else:
            single_prompt = False

        batch_size = len(prompts)

        # Tokenize prompts
        prompt_tokens = [self.tokenizer.encode(prompt) for prompt in prompts]
        min_prompt_len = min(len(p) for p in prompt_tokens)
        max_prompt_len = max(len(p) for p in prompt_tokens)
        total_seq_len = max_prompt_len + max_tokens

        # Ensure total sequence length doesn't exceed model limits
        if total_seq_len > self.config.max_position_embeddings:
            total_seq_len = self.config.max_position_embeddings
            max_tokens = total_seq_len - max_prompt_len
            print(f"Warning: Reduced generation length to {max_tokens} tokens due to model context limit.")

        # Create attention mask
        min_dtype = torch.finfo(self.dtype).min
        boolean_mask, local_boolean_mask = self.create_attention_mask(
            torch.zeros((batch_size, total_seq_len), dtype=torch.long, device=device),
            total_seq_len
        )
        mask_tensor = torch.where(boolean_mask, 0, torch.tensor(min_dtype, dtype=torch.float32, device=device))
        if local_boolean_mask is not None:
            local_mask_tensor = torch.where(local_boolean_mask, 0, torch.tensor(min_dtype, dtype=torch.float32, device=device))
        else:
            local_mask_tensor = None

        # Build KV caches
        kv_caches = []
        for _ in range(self.config.num_hidden_layers):
            size = (batch_size, total_seq_len, self.config.num_attention_heads,
                    self.config.head_dim)
            dtype = self.config.get_dtype()
            k_cache = torch.zeros(size=size, dtype=dtype, device=device)
            v_cache = torch.zeros(size=size, dtype=dtype, device=device)
            kv_caches.append((k_cache, v_cache))

        # Prepare input tensors
        input_token_ids_tensor = torch.full((batch_size, min_prompt_len),
                                            self.tokenizer.pad_id,
                                            dtype=torch.int64, device=device)
        token_ids_tensor = torch.full((batch_size, total_seq_len),
                                      self.tokenizer.pad_id,
                                      dtype=torch.int64, device=device)

        # Fill in prompt tokens
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

        # Generate tokens
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

            # Determine whether to use prompt tokens or generated tokens
            curr_prompt_mask = prompt_mask_tensor.index_select(1, output_index).squeeze(dim=1)
            curr_token_ids = token_ids_tensor.index_select(1, output_index).squeeze(dim=1)
            output_token_ids = torch.where(curr_prompt_mask, curr_token_ids, next_token_ids).unsqueeze(dim=1)
            token_ids_tensor.index_copy_(1, output_index, output_token_ids)

            # Check if all sequences have reached EOS
            if (output_token_ids == self.tokenizer.eos_id).all():
                break

            # Prepare for next token
            input_token_ids_tensor = output_token_ids
            input_positions_tensor = output_index.unsqueeze(dim=-1)
            curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
            curr_local_mask_tensor = local_mask_tensor.index_select(2, input_positions_tensor) if local_mask_tensor is not None else None
            output_positions_tensor = torch.tensor(0, dtype=torch.int64, device=device)
            output_index = output_index + 1

        # Detokenize output
        token_ids = token_ids_tensor.tolist()
        results = []
        for i, tokens in enumerate(token_ids):
            # Extract the prompt length for this sequence
            prompt_len = len(prompt_tokens[i])

            # Extract generated tokens (after the prompt)
            output = tokens[prompt_len:prompt_len + max_tokens]

            # Truncate at EOS if present
            if self.tokenizer.eos_id in output:
                eos_index = output.index(self.tokenizer.eos_id)
                output = output[:eos_index]

            results.append(self.tokenizer.decode(output))

        # Return single string if single prompt was provided
        return results[0] if single_prompt else results

    def load_weights(self, model_path: str):
        """Load weights from checkpoint file or directory."""
        if os.path.isfile(model_path):
            self.load_state_dict(
                torch.load(
                    model_path, mmap=True, weights_only=True,
                )['model_state_dict'],
                strict=False,
            )
        else:
            index_path = os.path.join(model_path, 'pytorch_model.bin.index.json')
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            shard_files = list(set(index["weight_map"].values()))
            for shard_file in shard_files:
                shard_path = os.path.join(model_path, shard_file)
                state_dict = torch.load(shard_path, map_location="cpu", weights_only=True)
                self.load_state_dict(state_dict, strict=False)
                del state_dict  # Save memory.
                gc.collect()