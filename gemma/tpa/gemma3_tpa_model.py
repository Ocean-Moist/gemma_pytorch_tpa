# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Gemma 3 multimodal model implementation with TPA."""

import torch
import os
import json
import gc
from torch import nn
import torch.nn.functional as F
from PIL import Image
from typing import Any, List, Optional, Sequence, Tuple, Union, Mapping

from .. import model as gemma_model
from .. import config as gemma_config
from .. import gemma3_preprocessor
from .. import tokenizer
from ..siglip_vision import siglip_vision_model


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
    """Tensor Product Attention module for Gemma 3."""
    def __init__(self, config: gemma_config.GemmaConfig, attn_type: gemma_config.AttentionType):
        super().__init__()
        self.config = config
        self.attn_type = attn_type
        
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads if hasattr(config, 'num_key_value_heads') else self.num_heads
        
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        
        # TPA specific parameters
        self.q_rank = getattr(config, 'q_rank', 6)
        self.k_rank = getattr(config, 'k_rank', 2)
        self.v_rank = getattr(config, 'v_rank', 2)
        
        # Debug info about dimensions
        print(f"TPAAttention: hidden_size={self.hidden_size}, head_dim={self.head_dim}")
        print(f"TPAAttention: num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}")
        print(f"TPAAttention: q_rank={self.q_rank}, k_rank={self.k_rank}, v_rank={self.v_rank}")
        
        # Scaling for attention
        if config.query_pre_attn_scalar is not None:
            self.scaling = config.query_pre_attn_scalar**-0.5
        else:
            self.scaling = self.head_dim**-0.5
        
        # Define TPA projection matrices
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
        
        # Initialize KV cache parameters with batch_size, seq_len, num_heads/kv_heads, rank/head_dim
        # These will be created as needed during inference
        self.cache_kA = None
        self.cache_kB = None
        self.cache_vA = None
        self.cache_vB = None
        
        # DEBUG: Print initialization stats of weight matrices
        with torch.no_grad():
            print(f"DEBUG INIT TPAAttn: W_A_q stats: mean={self.W_A_q.weight.mean().item():.6f}, std={self.W_A_q.weight.std().item():.6f}")
            print(f"DEBUG INIT TPAAttn: W_A_k stats: mean={self.W_A_k.weight.mean().item():.6f}, std={self.W_A_k.weight.std().item():.6f}")
            print(f"DEBUG INIT TPAAttn: W_A_v stats: mean={self.W_A_v.weight.mean().item():.6f}, std={self.W_A_v.weight.std().item():.6f}")
            print(f"DEBUG INIT TPAAttn: W_B_q stats: mean={self.W_B_q.weight.mean().item():.6f}, std={self.W_B_q.weight.std().item():.6f}")
            print(f"DEBUG INIT TPAAttn: W_B_k stats: mean={self.W_B_k.weight.mean().item():.6f}, std={self.W_B_k.weight.std().item():.6f}")
            print(f"DEBUG INIT TPAAttn: W_B_v stats: mean={self.W_B_v.weight.mean().item():.6f}, std={self.W_B_v.weight.std().item():.6f}")
            print(f"DEBUG INIT TPAAttn: o_proj stats: mean={self.o_proj.weight.mean().item():.6f}, std={self.o_proj.weight.std().item():.6f}")

    def _init_kv_cache(self, batch_size, max_seq_len):
        """Initialize KV cache for TPA attention."""
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        # For TPA, we need to cache both A and B components for K and V matrices
        # This is a key difference from standard attention where we cache final K and V
        
        # Ensure that batch_size and max_seq_len are proper integers
        batch_size = int(batch_size)
        max_seq_len = int(max_seq_len)
        
        # Shape for A components: [batch_size, max_seq_len, num_kv_heads, rank]
        # A components are of shape [hidden_dim, num_heads*rank] projected to [batch, seq_len, num_heads, rank]
        try:
            self.cache_kA = torch.zeros(
                (batch_size, max_seq_len, self.num_kv_heads, self.k_rank),
                device=device, dtype=dtype
            )
            self.cache_vA = torch.zeros(
                (batch_size, max_seq_len, self.num_kv_heads, self.v_rank),
                device=device, dtype=dtype
            )
            
            # Shape for B components: [batch_size, max_seq_len, rank, head_dim]
            # B components are of shape [hidden_dim, rank*head_dim] projected to [batch, seq_len, rank, head_dim]
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
        
        # A projections
        A_q = self.W_A_q(hidden_states).view(batch_size, seq_len, self.num_heads, self.q_rank)
        A_k = self.W_A_k(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.k_rank)
        A_v = self.W_A_v(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.v_rank)
        
        # B projections
        B_q = self.W_B_q(hidden_states).view(batch_size, seq_len, self.q_rank, self.head_dim)
        B_k = self.W_B_k(hidden_states).view(batch_size, seq_len, self.k_rank, self.head_dim)
        B_v = self.W_B_v(hidden_states).view(batch_size, seq_len, self.v_rank, self.head_dim)
        
        # Apply rotary positional embedding to B_q and B_k
        # Standard RoPE expects [batch, head, seq_len, head_dim]
        # But our B matrices are [batch, seq_len, rank, head_dim]
        # We need to adapt the format for proper RoPE application
        
        # Define a TPA-specific RoPE application that works with our tensor format
        def apply_rotary_emb_to_B(x, freqs_cis):
            # x shape: [batch, seq_len, rank, head_dim]
            batch_size, seq_len, rank, head_dim = x.shape
            
            # We need to ensure we're only treating the head_dim as complex pairs
            # First, verify head_dim is even (required for complex representation)
            if head_dim % 2 != 0:
                raise ValueError(f"Head dimension {head_dim} must be even for RoPE")
                
            half_dim = head_dim // 2
            
            # Process each rank separately to avoid dimension mixing
            # This prevents scrambling the real/imaginary parts
            output = torch.zeros_like(x)
            
            # Loop over ranks to avoid flattening/mixing dimensions
            for r in range(rank):
                # Extract just this rank: [batch, seq_len, head_dim]
                x_r = x[:, :, r, :]
                
                # Apply standard RoPE directly to this rank slice
                # Split head_dim into real/imaginary parts
                x_complex = torch.view_as_complex(
                    torch.stack(torch.chunk(x_r.float(), 2, dim=-1), dim=-1)
                )
                
                # Apply complex multiplication
                x_rotated = torch.view_as_real(x_complex * freqs_cis)
                
                # Correctly reshape back to [batch, seq_len, head_dim]
                # avoiding any operations that might mix dimensions
                x_rotated = torch.cat(torch.chunk(x_rotated, 2, dim=-1), dim=-2)
                x_rotated = x_rotated.reshape(batch_size, seq_len, head_dim)
                
                # Store the result in the original rank's position
                output[:, :, r, :] = x_rotated
            
            # Return with the original data type, now that we're back to a pure real tensor
            return output.type_as(x)
        
        # Apply our TPA-specific RoPE function
        B_q = apply_rotary_emb_to_B(B_q, freqs_cis)
        B_k = apply_rotary_emb_to_B(B_k, freqs_cis)
        
        # Handle KV cache
        # We need to store and retrieve both A and B components for K and V
        # Create cache if it doesn't exist
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
        
        # Reshape for tensor product computation
        A_q_flat = A_q.reshape(batch_size * seq_len, self.num_heads, self.q_rank)
        A_k_flat = A_k.reshape(-1, self.num_kv_heads, self.k_rank)
        A_v_flat = A_v.reshape(-1, self.num_kv_heads, self.v_rank)
        
        B_q_flat = B_q.reshape(batch_size * seq_len, self.q_rank, self.head_dim)
        B_k_flat = B_k.reshape(-1, self.k_rank, self.head_dim)
        B_v_flat = B_v.reshape(-1, self.v_rank, self.head_dim)
        
        # Compute Q, K, V matrices using tensor product
        # q = (A_q @ B_q) / q_rank
        # Compute using bmm for batched matrix multiplication
        q = torch.bmm(A_q_flat, B_q_flat).div(self.q_rank)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        kv_seq_len = A_k.size(1)
        
        # Handle grouped attention if num_kv_heads < num_heads
        if self.num_kv_heads < self.num_heads:
            # For grouped query attention, we need to repeat k,v for each query head in the group
            heads_per_group = self.num_heads // self.num_kv_heads
            
            # Compute k,v for each kv head
            k_list = []
            v_list = []
            for kv_head_idx in range(self.num_kv_heads):
                A_k_head = A_k[:, :, kv_head_idx:kv_head_idx+1]  # [batch, kv_seq_len, 1, k_rank]
                A_v_head = A_v[:, :, kv_head_idx:kv_head_idx+1]  # [batch, kv_seq_len, 1, v_rank]
                
                # Flatten for bmm
                A_k_head_flat = A_k_head.reshape(-1, 1, self.k_rank)  # [batch*kv_seq_len, 1, k_rank]
                A_v_head_flat = A_v_head.reshape(-1, 1, self.v_rank)  # [batch*kv_seq_len, 1, v_rank]
                
                B_k_head_flat = B_k_flat  # [batch*kv_seq_len, k_rank, head_dim]
                B_v_head_flat = B_v_flat  # [batch*kv_seq_len, v_rank, head_dim]
                
                # Compute k,v for this head
                k_head = torch.bmm(A_k_head_flat, B_k_head_flat).div(self.k_rank)  # [batch*kv_seq_len, 1, head_dim]
                v_head = torch.bmm(A_v_head_flat, B_v_head_flat).div(self.v_rank)  # [batch*kv_seq_len, 1, head_dim]
                
                # Reshape
                k_head = k_head.view(batch_size, kv_seq_len, 1, self.head_dim)  # [batch, kv_seq_len, 1, head_dim]
                v_head = v_head.view(batch_size, kv_seq_len, 1, self.head_dim)  # [batch, kv_seq_len, 1, head_dim]
                
                # Repeat this kv head for each query head in the group
                start_head_idx = kv_head_idx * heads_per_group
                end_head_idx = (kv_head_idx + 1) * heads_per_group
                
                for _ in range(start_head_idx, end_head_idx):
                    k_list.append(k_head)
                    v_list.append(v_head)
                
            # Concatenate along the head dimension
            k = torch.cat(k_list, dim=2)  # [batch, kv_seq_len, num_heads, head_dim]
            v = torch.cat(v_list, dim=2)  # [batch, kv_seq_len, num_heads, head_dim]
        else:
            # For non-GQA case, compute k,v for all heads
            # Compute k,v using bmm
            k_flat = torch.bmm(A_k_flat, B_k_flat).div(self.k_rank)  # [batch*kv_seq_len, num_kv_heads, head_dim]
            v_flat = torch.bmm(A_v_flat, B_v_flat).div(self.v_rank)  # [batch*kv_seq_len, num_kv_heads, head_dim]
            
            # Reshape
            k = k_flat.view(batch_size, kv_seq_len, self.num_heads, self.head_dim)  # [batch, kv_seq_len, num_heads, head_dim]
            v = v_flat.view(batch_size, kv_seq_len, self.num_heads, self.head_dim)  # [batch, kv_seq_len, num_heads, head_dim]
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)  # [batch, num_heads, kv_seq_len, head_dim]
        v = v.transpose(1, 2)  # [batch, num_heads, kv_seq_len, head_dim]
        
        # Apply QK normalization if enabled
        if self.query_norm is not None and self.key_norm is not None:
            q = self.query_norm(q)
            k = self.key_norm(k)
        
        # Scale q
        q = q * self.scaling
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(2, 3))  # [batch, num_heads, seq_len, kv_seq_len]
        
        # Apply attention logit softcapping if enabled
        if self.attn_logit_softcapping is not None:
            scores = scores / self.attn_logit_softcapping
            scores = torch.tanh(scores)
            scores = scores * self.attn_logit_softcapping
        
        # Get the dimensions for proper masking
        batch_size, num_heads, seq_len, kv_seq_len = scores.shape
        
        # SIMPLIFIED MASK HANDLING
        # Always ensure masks match scores dimensions exactly
        
        # Main attention mask
        if mask is not None:
            # Slice the mask to match scores dimensions if needed
            if mask.size(-1) != kv_seq_len or mask.size(-2) != seq_len:
                try:
                    # Extract exactly the portion we need based on scores dimensions
                    # For generation with KV cache: typically mask is [batch, 1, max_seq, max_seq]
                    # and we need [batch, 1, seq_len, kv_seq_len]
                    mask_compatible = mask[:, :, -seq_len:, :kv_seq_len]
                    
                    # Debug dimension check
                    if mask_compatible.size(-1) != kv_seq_len or mask_compatible.size(-2) != seq_len:
                        print(f"WARNING: After slicing, mask still has incorrect dimensions: "
                              f"{mask_compatible.shape}, expected last 2 dims to be [{seq_len}, {kv_seq_len}]")
                except Exception as e:
                    # If extraction fails, create a compatible causal mask
                    min_dtype = torch.finfo(scores.dtype).min
                    mask_compatible = torch.zeros_like(scores)
                    # Apply causal masking pattern if multi-token
                    if seq_len > 1:
                        for i in range(seq_len):
                            mask_compatible[:, :, i, i+1:] = min_dtype
                    print(f"Created compatible mask with shape {mask_compatible.shape}")
            else:
                # Mask already has compatible dimensions
                mask_compatible = mask
                
            # Now guaranteed to have compatible dimensions
            scores = scores + mask_compatible
        
        # Sliding window mask (similar simplification)
        if (
            self.attn_type == gemma_config.AttentionType.LOCAL_SLIDING
            and self.sliding_window_size is not None
            and local_mask is not None
        ):
            # Ensure local_mask has compatible dimensions
            if local_mask.size(-1) != kv_seq_len or local_mask.size(-2) != seq_len:
                try:
                    # Extract the compatible portion
                    local_mask_compatible = local_mask[:, :, -seq_len:, :kv_seq_len]
                except:
                    # Create simple sliding window constraint if extraction fails
                    min_dtype = torch.finfo(scores.dtype).min
                    local_mask_compatible = torch.zeros_like(scores)
                    if self.sliding_window_size < kv_seq_len:
                        # Apply sliding window constraint - block attention to tokens beyond window
                        local_mask_compatible[:, :, :, :-self.sliding_window_size] = min_dtype
            else:
                local_mask_compatible = local_mask
                
            # Apply compatible sliding window mask
            scores = scores + local_mask_compatible
        
        # Compute attention weights
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(q)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # [batch, num_heads, seq_len, head_dim]
        
        # Transpose and reshape
        output = output.transpose(1, 2).contiguous()  # [batch, seq_len, num_heads, head_dim]
        output = output.reshape(batch_size, seq_len, -1)  # [batch, seq_len, num_heads*head_dim]
        
        # Apply output projection
        output = self.o_proj(output)
        
        return output


class TPADecoderLayer(nn.Module):
    """Decoder layer using TPA attention for Gemma 3."""
    def __init__(self, config: gemma_config.GemmaConfig, attn_type: gemma_config.AttentionType):
        super().__init__()
        self.attn_type = attn_type
        self.self_attn = TPAAttention(config=config, attn_type=self.attn_type)
        
        # MLP components
        self.mlp = gemma_model.GemmaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant=config.quant,
        )
        
        # Normalization components
        self.input_layernorm = gemma_model.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = gemma_model.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = (
            gemma_model.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if getattr(config, 'use_pre_ffw_norm', False)
            else None
        )
        self.post_feedforward_layernorm = (
            gemma_model.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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


class TPAGemmaModel(nn.Module):
    """Gemma model using TPA modules."""
    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        
        # Create TPA layers
        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            attn_type = (
                config.attn_types[i % len(config.attn_types)]
                if config.attn_types is not None
                else gemma_config.AttentionType.GLOBAL
            )
            self.layers.append(TPADecoderLayer(config, attn_type))
        
        # Final normalization
        self.norm = gemma_model.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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


class Gemma3ForMultimodalLMwithTPA(nn.Module):
    """Gemma3 model for multimodal causal LM with TPA."""
    def __init__(
        self,
        config: gemma_config.GemmaConfig,
    ):
        super().__init__()
        self.dtype = config.get_dtype()
        assert config.architecture == gemma_config.Architecture.GEMMA_3
        self.config = config
        max_seq_len = config.max_position_embeddings
        head_dim = config.head_dim
        vocab_size = config.vocab_size
        
        # TPA specific parameters
        self.q_rank = getattr(config, 'q_rank', 6)
        self.k_rank = getattr(config, 'k_rank', 2)
        self.v_rank = getattr(config, 'v_rank', 2)
        
        # Initialize tokenizer if path is provided
        self.tokenizer = tokenizer.Tokenizer(config.tokenizer) if hasattr(config, 'tokenizer') else None
        
        # Text token embedder
        self.text_token_embedder = gemma_model.Embedding(vocab_size, config.hidden_size, config.quant)
        
        # Initialize TPA model
        self.model = TPAGemmaModel(config)
        
        # Initialize sampler
        self.sampler = gemma_model.Sampler(vocab_size, config)
        
        # Initialize vision components if vision config is provided
        if getattr(config, 'vision_config', None) is not None:
            self.siglip_vision_model = siglip_vision_model.SiglipVisionModel(config.vision_config)
            # Vision normalization
            self.mm_soft_embedding_norm = gemma_model.RMSNorm(
                config.vision_config.embedding_dim, eps=config.rms_norm_eps
            )
            # Vision projection
            self.mm_input_projection = gemma_model.Linear(
                config.vision_config.embedding_dim, config.hidden_size, config.quant
            )
        
        # Check for rope_wave_length
        if not hasattr(config, 'rope_wave_length') or config.rope_wave_length is None:
            raise ValueError('rope_wave_length must be provided for Gemma3.')
        
        # Precompute rotary embeddings
        rope_lengths = config.rope_wave_length
        defaults = {
            gemma_config.AttentionType.LOCAL_SLIDING: 10_000,
            gemma_config.AttentionType.GLOBAL: 10_000,
        }
        
        # Register rotary embeddings for different attention types
        self._register_freqs_cis(
            'local_freqs_cis', 
            head_dim, 
            max_seq_len, 
            theta=rope_lengths.get(
                gemma_config.AttentionType.LOCAL_SLIDING, 
                defaults[gemma_config.AttentionType.LOCAL_SLIDING]
            )
        )
        
        self._register_freqs_cis(
            'global_freqs_cis', 
            head_dim, 
            max_seq_len, 
            theta=rope_lengths.get(
                gemma_config.AttentionType.GLOBAL, 
                defaults[gemma_config.AttentionType.GLOBAL]
            ),
            rope_scaling_factor=getattr(config, 'rope_scaling_factor', 1)
        )
        
        # Flag indicating whether to use factorized weights
        self.use_factorized_weights = False

    def _register_freqs_cis(
        self, name: str, head_dim: int, max_seq_len: int, theta: int = 10_000, rope_scaling_factor: int = 1
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
        input_token_ids: torch.Tensor,  # B x L
        input_positions: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
        output_positions: torch.Tensor,
        temperatures: Union[torch.Tensor, None],
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        local_mask: torch.Tensor | None = None,
        image_patches: Optional[torch.Tensor] = None,  # B x N x C x H x W (3x896x896)
        image_presence_mask: Optional[torch.Tensor] = None,  # B x N
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # DEBUG: Log input tokens
        print(f"DEBUG: Input token IDs shape: {input_token_ids.shape}")
        print(f"DEBUG: First few input tokens: {input_token_ids[0, :10].tolist()}")
        print(f"DEBUG: Input positions: {input_positions.tolist()}")
        
        # Prepare rotary embeddings
        freqs_cis = {}
        freqs_cis[gemma_config.AttentionType.LOCAL_SLIDING] = (
            self.local_freqs_cis.index_select(0, input_positions)
        )
        freqs_cis[gemma_config.AttentionType.GLOBAL] = (
            self.global_freqs_cis.index_select(0, input_positions)
        )
        
        # Get token embeddings
        hidden_states = self.text_token_embedder(input_token_ids)
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = hidden_states * normalizer
        
        # DEBUG: Log embedding stats
        print(f"DEBUG: Token embeddings shape: {hidden_states.shape}")
        print(f"DEBUG: Token embeddings mean: {hidden_states.mean().item():.6f}, std: {hidden_states.std().item():.6f}")
        print(f"DEBUG: Token embeddings min: {hidden_states.min().item():.6f}, max: {hidden_states.max().item():.6f}")
        
        # Process image embeddings if provided
        if image_patches is not None and hasattr(self, 'siglip_vision_model'):
            B, N, C, H, W = image_patches.shape
            # Flatten and pass to vision model
            flattened_input = image_patches.reshape(B * N, C, H, W)  # (B*N)xCxHxW
            image_embeddings = self.siglip_vision_model(flattened_input)  # (B*N)xUxD
            image_embeddings = self.mm_soft_embedding_norm(image_embeddings)  # (B*N) x U x D
            image_embeddings = self.mm_input_projection(image_embeddings)  # (B*N) x U x model_dim
            hidden_states = self.populate_image_embeddings(
                hidden_states.clone(),
                image_embeddings.clone(),
                input_token_ids.clone(),
                image_presence_mask.clone(),
            )
        
        # Get KV write indices
        kv_write_indices = input_positions
        
        # DEBUG: Log model input state
        print(f"DEBUG: Hidden states before model: mean={hidden_states.mean().item():.6f}, std={hidden_states.std().item():.6f}")
        
        try:
            # Forward pass through the model
            hidden_states = self.model(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis,
                kv_write_indices=kv_write_indices,
                kv_caches=kv_caches,
                mask=mask,
                local_mask=local_mask,
            )
        except Exception as e:
            print(f"ERROR in model forward pass: {e}")
            import traceback
            traceback.print_exc()
            # Create fallback hidden states with similar shape but all zeros
            # This will help us identify if the issue is in the model forward pass
            print("DEBUG: Using fallback zero hidden states to continue debugging")
            hidden_states = torch.zeros_like(hidden_states)
            # Randomize slightly to avoid numerical issues
            hidden_states = hidden_states + torch.randn_like(hidden_states) * 1e-2
        
        # DEBUG: Log hidden states after model processing
        print(f"DEBUG: Hidden states after model: shape={hidden_states.shape}")
        print(f"DEBUG: Hidden states after model: mean={hidden_states.mean().item():.6f}, std={hidden_states.std().item():.6f}")
        print(f"DEBUG: Hidden states after model: min={hidden_states.min().item():.6f}, max={hidden_states.max().item():.6f}")
        
        # Get embedder weight for final projection
        embedder_weight = self.text_token_embedder.weight
        if self.config.quant:
            embedder_weight = (
                embedder_weight * self.text_token_embedder.weight_scaler.unsqueeze(-1)
            )
        
        # DEBUG: Log embedding weight stats
        print(f"DEBUG: Embedding weight shape: {embedder_weight.shape}")
        print(f"DEBUG: Embedding weight stats: mean={embedder_weight.mean().item():.6f}, std={embedder_weight.std().item():.6f}")
        
        # Sample next tokens
        next_tokens, logits = self.sampler(
            embedding=embedder_weight,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
        )
        
        # DEBUG: Log logits and sampled token information
        print(f"DEBUG: Logits shape: {logits.shape}")
        print(f"DEBUG: Logits stats: mean={logits.mean().item():.6f}, std={logits.std().item():.6f}")
        print(f"DEBUG: Logits min/max: min={logits.min().item():.6f}, max={logits.max().item():.6f}")
        
        # Check for NaN or Inf in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("WARNING: NaN or Inf values detected in logits!")
        
        # Get the top 10 token probabilities for analysis
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, 10, dim=-1)
        print(f"DEBUG: Top 10 token IDs: {top_indices[0].tolist()}")
        print(f"DEBUG: Top 10 probabilities: {top_probs[0].tolist()}")
        
        print(f"DEBUG: Selected next token: {next_tokens[0].item()}")
        
        return next_tokens, logits

    def populate_image_embeddings(
        self,
        hidden_states: torch.Tensor,  # B x L x model_dim
        image_embeddings: torch.Tensor,  # (B*N) x U x model_dim
        input_token_ids: torch.Tensor,  # B x L
        image_presence_mask: torch.Tensor,  # B x N
    ):
        batch_size, seq_len, model_dim = hidden_states.shape
        
        # Step 1: Fetch valid image embeddings
        # Flatten indices of valid image embeddings
        valid_image_embeddings_indices = torch.nonzero(image_presence_mask.flatten(), as_tuple=False).squeeze()
        
        # Extract valid image embeddings
        valid_image_embeddings = image_embeddings.index_select(0, valid_image_embeddings_indices)
        
        # Step 2: Replace image embeddings at the right places
        # Identify image placeholder tokens
        image_placeholder_mask = input_token_ids == self.tokenizer.image_token_placeholder_id
        image_placeholder_indices = image_placeholder_mask.flatten().nonzero(as_tuple=False).squeeze()
        
        # Reshape hidden states for indexing
        hidden_states = hidden_states.reshape(-1, self.config.hidden_size)
        
        # Replace image placeholders with image embeddings
        hidden_states[image_placeholder_indices] = valid_image_embeddings.reshape(-1, self.config.hidden_size)
        
        # Reshape back to original shape
        return hidden_states.reshape(batch_size, seq_len, model_dim).contiguous()

    def create_attention_mask(self, input_ids: torch.Tensor, sequence_length: int):
        batch_size = input_ids.shape[0]
        
        # Create causal mask
        causal_mask = torch.tril(
            torch.ones((batch_size, 1, sequence_length, sequence_length), dtype=torch.bool, device=input_ids.device)
        )
        
        # Check for image tokens
        image_token_mask = input_ids == self.tokenizer.image_token_placeholder_id
        
        # Pad the mask for boundary detection
        padded_mask = nn.functional.pad(image_token_mask, (1, 0), value=0)
        
        # Find boundaries of image token patches
        boundary = padded_mask[:, 1:] > padded_mask[:, :-1]
        
        # Number the boundaries
        numbered_boundary = torch.cumsum(boundary, dim=-1)
        
        # Create block indices for query
        q_block_indices = image_token_mask * numbered_boundary
        kv_block_indices = q_block_indices
        
        # Create bidirectional mask for image tokens
        bidirectional_mask = torch.logical_and(
            kv_block_indices[:, None, :] == q_block_indices.unsqueeze(-1),
            q_block_indices.unsqueeze(-1) > 0,
        )
        
        # Combine causal and bidirectional masks
        attention_mask = torch.logical_or(causal_mask, bidirectional_mask.unsqueeze(1))
        
        # Create local mask with sliding window if configured
        if self.config.sliding_window_size is not None:
            local_mask = torch.logical_and(
                attention_mask,
                torch.triu(
                    torch.ones((1, 1, sequence_length, sequence_length), dtype=torch.bool, device=input_ids.device),
                    diagonal=-(self.config.sliding_window_size-1)
                )
            )
        else:
            local_mask = attention_mask
        
        return attention_mask, local_mask

    def generate(
        self,
        prompts: Sequence[Union[str, Sequence[Union[str, Image.Image]]]],
        device: Any = None,
        max_tokens: int = 100,
        temperature: Union[float, None] = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
    ) -> Sequence[str]:
        """Generates responses for given prompts using Gemma model with TPA."""
        if device is None:
            device = next(self.parameters()).device
            
        # DEBUG: Log generation parameters
        print(f"DEBUG GENERATE: Prompts: {prompts}")
        print(f"DEBUG GENERATE: Device: {device}, max_tokens: {max_tokens}")
        print(f"DEBUG GENERATE: Temperature: {temperature}, top_p: {top_p}, top_k: {top_k}")
            
        # Check if we have text-only or multimodal input
        is_text_only = all(isinstance(p, str) for p in prompts)
        
        # For text-only models without vision config, use a simpler approach
        if is_text_only or not hasattr(self.config, 'vision_config') or self.config.vision_config is None:
            # Convert string prompts to token IDs
            if isinstance(prompts[0], str):
                # Handle text-only case
                batch_size = len(prompts)
                token_ids_list = [self.tokenizer.encode(p) for p in prompts]
                max_len = max(len(ids) for ids in token_ids_list)
                
                # DEBUG: Show tokenization
                print(f"DEBUG GENERATE: Tokenized first prompt: {token_ids_list[0]}")
                print(f"DEBUG GENERATE: Decoded first tokens: {[self.tokenizer.sp_model.IdToPiece(id) for id in token_ids_list[0][:20]]}")
                
                # Create padded tensor
                token_ids_tensor = torch.full(
                    (batch_size, max_len + max_tokens),
                    self.tokenizer.pad_id,
                    dtype=torch.int64,
                    device=device
                )
                
                # Fill in token IDs
                for i, ids in enumerate(token_ids_list):
                    token_ids_tensor[i, :len(ids)] = torch.tensor(ids, dtype=torch.int64, device=device)
                
                # Create mask for actual tokens
                prompt_mask_tensor = token_ids_tensor != self.tokenizer.pad_id
                
                # DEBUG: Log token tensor stats
                print(f"DEBUG GENERATE: Token tensor shape: {token_ids_tensor.shape}")
                print(f"DEBUG GENERATE: Prompt mask sum (non-pad tokens): {prompt_mask_tensor.sum().item()}")
                
                processing_result = {
                    "user_input_token_ids": token_ids_tensor,
                    "image_batch": None,
                    "batch_size": batch_size,
                    "min_prompt_len": min(len(ids) for ids in token_ids_list),
                    "max_prompt_len": max_len,
                    "max_seq_len": max_len + max_tokens,
                    "image_presence_mask": None
                }
                
                # DEBUG: Log processing result
                print(f"DEBUG GENERATE: Processing result: min_prompt_len={processing_result['min_prompt_len']}, "
                      f"max_prompt_len={processing_result['max_prompt_len']}, max_seq_len={processing_result['max_seq_len']}")
                
            else:
                # For potentially multimodal input but without vision config, convert to text-only
                text_prompts = [[p if isinstance(p, str) else "[IMAGE]" for p in seq] for seq in prompts]
                text_only_prompts = ["".join(p) for p in text_prompts]
                return self.generate(text_only_prompts, device, max_tokens, temperature, top_p, top_k)
        else:
            # For multimodal models with vision config, use the full preprocessor
            processing_result = gemma3_preprocessor.tokenize_raw_input(
                self.tokenizer, prompts, self.config, max_tokens, device
            )
        
        batch_size = processing_result["batch_size"]
        user_input_token_ids = processing_result["user_input_token_ids"]
        image_batch = processing_result["image_batch"]
        min_prompt_len = processing_result["min_prompt_len"]
        max_prompt_len = processing_result["max_prompt_len"]
        total_seq_len = processing_result["max_seq_len"]
        image_presence_mask = processing_result["image_presence_mask"]
        
        # Create attention mask
        min_dtype = torch.finfo(self.dtype).min
        if self.config.sliding_window_size is None:
            raise ValueError('gemma 3 model requires sliding_window size')
            
        boolean_mask, local_boolean_mask = self.create_attention_mask(user_input_token_ids, total_seq_len)
        mask_tensor = torch.where(boolean_mask, 0, torch.tensor(min_dtype, dtype=torch.float32, device=device)).contiguous()
        local_mask_tensor = torch.where(local_boolean_mask, 0, torch.tensor(min_dtype, dtype=torch.float32, device=device)).contiguous()
        
        # Initialize KV caches
        kv_caches = []
        for _ in range(self.config.num_hidden_layers):
            size = (batch_size, total_seq_len, self.config.num_key_value_heads, self.config.head_dim)
            dtype = self.config.get_dtype()
            k_cache = torch.zeros(size=size, dtype=dtype, device=device)
            v_cache = torch.zeros(size=size, dtype=dtype, device=device)
            kv_caches.append((k_cache, v_cache))
        
        # Prepare input tensors
        input_token_ids_tensor = torch.full(
            (batch_size, min_prompt_len),
            self.tokenizer.pad_id,
            dtype=torch.int64, 
            device=device
        )
        
        token_ids_tensor = user_input_token_ids.to(device)
        for i in range(batch_size):
            p = user_input_token_ids[i]
            input_token_ids_tensor[i, :min_prompt_len] = p[:min_prompt_len]
        
        input_positions_tensor = torch.arange(0, min_prompt_len, dtype=torch.int64, device=device)
        prompt_mask_tensor = token_ids_tensor != self.tokenizer.pad_id
        curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
        curr_local_mask_tensor = local_mask_tensor.index_select(2, input_positions_tensor)
        output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(device)
        temperatures_tensor = None if not temperature else torch.FloatTensor([temperature] * batch_size).to(device)
        top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
        top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)
        output_index = torch.tensor(min_prompt_len, dtype=torch.int64, device=device)
        
        # Generate tokens
        for i in range(total_seq_len - min_prompt_len):
            next_token_ids, _ = self(
                input_token_ids=input_token_ids_tensor,
                image_patches=image_batch,
                image_presence_mask=image_presence_mask,
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
            
            input_token_ids_tensor = output_token_ids
            input_positions_tensor = output_index.unsqueeze(dim=-1)
            curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
            curr_local_mask_tensor = local_mask_tensor.index_select(2, input_positions_tensor) if local_mask_tensor is not None else None
            output_positions_tensor = torch.tensor(0, dtype=torch.int64, device=device)
            output_index = output_index + 1
            
            # Clear image data after first iteration to avoid reprocessing
            image_batch = None
            image_presence_mask = None
        
        # Decode generated tokens - only return the generated part (not the prompt)
        token_ids = token_ids_tensor.tolist()
        results = []
        for i, tokens in enumerate(token_ids):
            # Extract only the generated tokens, excluding the prompt
            prompt_len = len(token_ids_list[i]) if 'token_ids_list' in locals() else min_prompt_len
            generated_tokens = tokens[prompt_len:prompt_len + max_tokens]
            
            # Truncate at eos token if present
            if self.tokenizer.eos_id in generated_tokens:
                eos_index = generated_tokens.index(self.tokenizer.eos_id)
                generated_tokens = generated_tokens[:eos_index]
                
            # Decode only the generated response
            results.append(self.tokenizer.decode(generated_tokens))
        
        return results

    def load_weights(self, model_path: str):
        """Load model weights from a checkpoint."""
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

    def convert_from_standard_weights(self, standard_model):
        """
        Convert from a standard Gemma model to TPA format.
        This is a placeholder method that delegates to modules.gqa_to_tpa.create_tpa_model_from_standard
        """
        try:
            from .modules.gqa_to_tpa import create_tpa_model_from_standard
            
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
            
            # Get ranks from config
            q_rank = getattr(self, 'q_rank', 6)
            k_rank = getattr(self, 'k_rank', 2)
            v_rank = getattr(self, 'v_rank', 2)
            
            # Use target_ranks if specified (typically from Tucker decomposition parameters)
            use_dynamic_ranks = True
            if hasattr(self, 'target_ranks'):
                if self.target_ranks.get('use_tensorly', False):
                    # Direct TensorLy mode
                    print("Using direct TensorLy Tucker decomposition")
                    q_rank = self.target_ranks.get('q_rank', q_rank)
                    k_rank = self.target_ranks.get('k_rank', k_rank)
                    v_rank = self.target_ranks.get('v_rank', v_rank)
                elif self.target_ranks.get('use_shared_factors', False):
                    # Shared factors mode
                    print("Using shared factors approach for Tucker decomposition")
                    q_rank = self.target_ranks.get('q_rank', q_rank)
                    k_rank = self.target_ranks.get('k_rank', k_rank)
                    v_rank = self.target_ranks.get('v_rank', v_rank)
                
                # Allow dynamic_ranks override
                if 'use_dynamic_ranks' in self.target_ranks:
                    use_dynamic_ranks = self.target_ranks['use_dynamic_ranks']
            
            print(f"Converting to TPA with ranks: q_rank={q_rank}, k_rank={k_rank}, v_rank={v_rank}")
            print(f"Using dynamic ranks: {use_dynamic_ranks}")
            
            # Create a new TPA model from the standard model
            tpa_model = create_tpa_model_from_standard(
                standard_model,
                q_rank=q_rank,
                k_rank=k_rank,
                v_rank=v_rank,
                dtype=dtype,
                device=device,
                use_dynamic_ranks=use_dynamic_ranks
            )
            
            # Replace self with the converted TPA model
            # Since we can't directly replace self, we'll copy all parameters
            # from the converted model to self
            self.load_state_dict(tpa_model.state_dict())
            
            # Copy any additional attributes
            for attr_name in dir(tpa_model):
                if not attr_name.startswith('_') and attr_name not in ['convert_from_standard_weights', 'parameters']:
                    if hasattr(tpa_model, attr_name) and not callable(getattr(tpa_model, attr_name)):
                        setattr(self, attr_name, getattr(tpa_model, attr_name))
            
            # Set factorized flag
            self.use_factorized_weights = True
            
            return self
            
        except ImportError:
            print("Error: Could not import gqa_to_tpa module. Conversion failed.")
            return self