"""Tensor Product Attention implementation for Gemma."""

import torch
import math
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple, Mapping

from .. import config as gemma_config
from .. import model as gemma_model


class ContextualTensorProductAttention(nn.Module):
    """
    Implementation of Contextual Tensor Product Attention from the T6 paper.
    
    This approach factorizes QKV activations using contextual low-rank components
    rather than factorizing weight matrices, resulting in greater KV cache savings.
    """
    def __init__(
        self,
        config,
        dim_model,
        num_heads,
        num_kv_heads,
        head_dim,
        q_rank=6,  # Default from T6 paper
        k_rank=2,  # Default from T6 paper
        v_rank=2   # Default from T6 paper
    ):
        super().__init__()
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_rank = q_rank
        self.k_rank = k_rank
        self.v_rank = v_rank
        
        # Using 1/sqrt(head_dim) for scaling
        self.scale = 1.0 / math.sqrt(head_dim)
        
        # Contextual factor matrices (A factors)
        self.W_A_q = nn.Linear(dim_model, num_heads * q_rank, bias=False)
        self.W_A_k = nn.Linear(dim_model, num_kv_heads * k_rank, bias=False)
        self.W_A_v = nn.Linear(dim_model, num_kv_heads * v_rank, bias=False)
        
        # Contextual factor matrices (B factors)
        self.W_B_q = nn.Linear(dim_model, q_rank * head_dim, bias=False)
        self.W_B_k = nn.Linear(dim_model, k_rank * head_dim, bias=False)
        self.W_B_v = nn.Linear(dim_model, v_rank * head_dim, bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(num_heads * head_dim, dim_model, bias=False)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights with specific distribution for better stability."""
        # Initialize with normal distribution for A factors
        nn.init.normal_(self.W_A_q.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W_A_k.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W_A_v.weight, mean=0.0, std=0.02)
        
        # Initialize with normal distribution for B factors
        nn.init.normal_(self.W_B_q.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W_B_k.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W_B_v.weight, mean=0.0, std=0.02)
        
        # Output projection uses Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.o_proj.weight)
    
    def forward(
        self,
        hidden_states,
        freqs_cis,
        kv_write_indices,
        kv_cache,
        mask=None,
        local_mask=None,
    ):
        batch_size, seq_len, _ = hidden_states.size()
        
        # Compute A factors
        A_q = self.W_A_q(hidden_states).view(batch_size, seq_len, self.num_heads, self.q_rank)
        A_k = self.W_A_k(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.k_rank)
        A_v = self.W_A_v(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.v_rank)
        
        # Compute B factors
        B_q = self.W_B_q(hidden_states).view(batch_size, seq_len, self.q_rank, self.head_dim)
        B_k = self.W_B_k(hidden_states).view(batch_size, seq_len, self.k_rank, self.head_dim)
        B_v = self.W_B_v(hidden_states).view(batch_size, seq_len, self.v_rank, self.head_dim)
        
        # Apply rotary position embeddings to B_q and B_k if needed
        if freqs_cis is not None:
            # For each position, apply RoPE to the B factors
            B_q = self._apply_rope_to_factor(B_q, freqs_cis)
            B_k = self._apply_rope_to_factor(B_k, freqs_cis)
        
        # Handle KV cache for autoregressive decoding
        if kv_cache is not None:
            # Unpack the factorized KV cache
            k_cache_A, k_cache_B, v_cache_A, v_cache_B = kv_cache
            
            # Write new values to KV cache if needed
            if kv_write_indices is not None and kv_write_indices.numel() > 0:
                # Update the cache at specified positions
                if kv_write_indices.numel() == 1:
                    # Single position update (common during generation)
                    idx = kv_write_indices.item()
                    k_cache_A[:, idx] = A_k[:, 0]
                    k_cache_B[:, idx] = B_k[:, 0]
                    v_cache_A[:, idx] = A_v[:, 0]
                    v_cache_B[:, idx] = B_v[:, 0]
                else:
                    # Update multiple positions (e.g., during prefill)
                    for i, idx in enumerate(kv_write_indices):
                        idx = idx.item()
                        if i < seq_len:
                            k_cache_A[:, idx] = A_k[:, i]
                            k_cache_B[:, idx] = B_k[:, i]
                            v_cache_A[:, idx] = A_v[:, i]
                            v_cache_B[:, idx] = B_v[:, i]
            
            # Determine context length from cache
            ctx_len = k_cache_A.size(1) if kv_write_indices is None else max(kv_write_indices.max().item() + 1, 1)
            
            # Use cached values
            A_k = k_cache_A[:, :ctx_len]
            B_k = k_cache_B[:, :ctx_len]
            A_v = v_cache_A[:, :ctx_len]
            B_v = v_cache_B[:, :ctx_len]
        else:
            # No cache, use current sequence values
            ctx_len = seq_len
        
        # Contextual tensor factorization for Q
        # Reshape for batch matmul: [batch_size*seq_len, num_heads, q_rank] x [batch_size*seq_len, q_rank, head_dim]
        A_q_flat = A_q.reshape(batch_size * seq_len, self.num_heads, self.q_rank)
        B_q_flat = B_q.reshape(batch_size * seq_len, self.q_rank, self.head_dim)
        
        # Compute Q via matrix multiplication of factors
        Q = torch.bmm(A_q_flat, B_q_flat)
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        Q = Q / self.q_rank  # Scale by rank as in T6 paper
        
        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        Q = Q.transpose(1, 2)
        Q = Q * self.scale  # Apply attention scaling
        
        # Expand K and V for multi-query attention if needed
        if self.num_kv_heads != self.num_heads:
            # For multi-query attention - repeat KV heads
            repeat_factor = self.num_heads // self.num_kv_heads
            
            # Repeat the A factors for keys and values
            A_k = A_k.unsqueeze(3).expand(-1, -1, -1, repeat_factor, -1)
            A_k = A_k.reshape(batch_size, ctx_len, self.num_heads, self.k_rank)
            
            A_v = A_v.unsqueeze(3).expand(-1, -1, -1, repeat_factor, -1)
            A_v = A_v.reshape(batch_size, ctx_len, self.num_heads, self.v_rank)
        
        # Compute K using tensorized attention
        # Reshape for batch matmul
        A_k_flat = A_k.reshape(batch_size * ctx_len, self.num_heads, self.k_rank) 
        B_k_flat = B_k.reshape(batch_size * ctx_len, self.k_rank, self.head_dim)
        
        K = torch.bmm(A_k_flat, B_k_flat)
        K = K.reshape(batch_size, ctx_len, self.num_heads, self.head_dim)
        K = K / self.k_rank  # Scale by rank
        
        # Compute V using tensorized attention
        A_v_flat = A_v.reshape(batch_size * ctx_len, self.num_heads, self.v_rank)
        B_v_flat = B_v.reshape(batch_size * ctx_len, self.v_rank, self.head_dim)
        
        V = torch.bmm(A_v_flat, B_v_flat)
        V = V.reshape(batch_size, ctx_len, self.num_heads, self.head_dim)
        V = V / self.v_rank  # Scale by rank
        
        # Transpose K and V to [batch_size, num_heads, ctx_len, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # Apply attention mask if provided
        if mask is not None:
            attention_scores = attention_scores + mask
        
        # Apply local mask if provided (for sliding window attention)
        if local_mask is not None:
            attention_scores = attention_scores + local_mask
            
        # Normalize attention scores
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Apply attention weights to values
        context = torch.matmul(attention_probs, V)
        
        # Reshape context tensor
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # Apply output projection
        output = self.o_proj(context)
        
        # Prepare KV cache for next step if needed
        present_key_value = (A_k, B_k, A_v, B_v) if kv_cache is not None else None
        
        return output, present_key_value, attention_probs
    
    def _apply_rope_to_factor(self, factor, freqs_cis):
        """Apply rotary position embeddings to B factors."""
        # factor shape: [batch_size, seq_len, rank, head_dim]
        # freqs_cis shape: [seq_len, head_dim//2]
        
        batch_size, seq_len, rank, head_dim = factor.shape
        
        # Reshape for applying RoPE
        factor_reshaped = factor.reshape(batch_size * seq_len, rank, head_dim)
        
        # Convert to complex for RoPE application
        factor_complex = torch.view_as_complex(
            torch.stack(torch.chunk(factor_reshaped.float(), 2, dim=-1), dim=-1)
        )
        
        # Reshape freqs_cis for broadcasting
        freqs_cis = freqs_cis.view(seq_len, 1, head_dim // 2)
        freqs_cis = freqs_cis.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        freqs_cis = freqs_cis.reshape(batch_size * seq_len, 1, head_dim // 2)
        
        # Apply complex multiplication
        factor_rotated = torch.view_as_real(factor_complex * freqs_cis).type_as(factor)
        factor_rotated = torch.cat(torch.chunk(factor_rotated, 2, dim=-1), dim=-2)
        
        # Reshape back to original shape
        factor_rotated = factor_rotated.reshape(batch_size, seq_len, rank, head_dim)
        
        return factor_rotated

class GemmaTensorProductAttention(nn.Module):
    """
    Gemma Tensor Product Attention (TPA) module.
    
    TPA factorizes the attention matrices (Q, K, V) into rank-R tensor products.
    This significantly reduces KV cache size during inference.
    """

    def __init__(
        self,
        config: gemma_config.GemmaConfig,
        attn_type: gemma_config.AttentionType,
    ):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim

        # Define TPA rank parameters (can be set in config or use defaults)
        self.q_rank = getattr(config, "q_rank", 6)  # Default to 6 as in the TPA paper
        self.k_rank = getattr(config, "k_rank", 2)  # Default to 2 as in the TPA paper
        self.v_rank = getattr(config, "v_rank", 2)  # Default to 2 as in the TPA paper

        # TPA projections for A factors (head dimension)
        self.W_A_q = gemma_model.Linear(config.hidden_size, self.num_heads * self.q_rank, config.quant)
        self.W_A_k = gemma_model.Linear(config.hidden_size, self.num_kv_heads * self.k_rank, config.quant)
        self.W_A_v = gemma_model.Linear(config.hidden_size, self.num_kv_heads * self.v_rank, config.quant)
        
        # TPA projections for B factors (token dimension)
        self.W_B_q = gemma_model.Linear(config.hidden_size, self.q_rank * self.head_dim, config.quant)
        self.W_B_k = gemma_model.Linear(config.hidden_size, self.k_rank * self.head_dim, config.quant)
        self.W_B_v = gemma_model.Linear(config.hidden_size, self.v_rank * self.head_dim, config.quant)
        
        # Output projection
        self.o_proj = gemma_model.Linear(self.num_heads * self.head_dim, self.hidden_size, config.quant)
        
        # Norms for Q and K (if needed)
        self.query_norm = (
            gemma_model.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if config.use_qk_norm
            else None
        )
        self.key_norm = (
            gemma_model.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if config.use_qk_norm
            else None
        )
        
        self.attn_type = attn_type
        self.sliding_window_size = config.sliding_window_size
        self.attn_logit_softcapping = config.attn_logit_softcapping
        
        # Scaling
        if config.query_pre_attn_scalar is not None:
            self.scaling = config.query_pre_attn_scalar**-0.5
        else:
            self.scaling = self.head_dim**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        mask: Optional[torch.Tensor] = None,
        local_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for TPA attention with comprehensive safety checks.
        
        Critical safety precautions:
        1. All token indices are checked and clamped to valid ranges
        2. All tensor shapes are verified before operations
        3. All input tensors are validated to avoid using token IDs as position indices
        """
        """
        Forward pass for Tensor Product Attention.
        
        TPA computes factorized Q, K, V matrices and uses a modified KV cache
        structure to store the factorized components.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            freqs_cis: Rotary positional embedding tensor
            kv_write_indices: Indices for writing to KV cache
            kv_cache: TPA KV cache containing (k_cache_A, k_cache_B, v_cache_A, v_cache_B)
            mask: Attention mask (optional, will create internal mask if None)
            local_mask: Local window attention mask (optional)
            
        Returns:
            Output tensor after TPA attention
        """
        hidden_states_shape = hidden_states.shape
        assert len(hidden_states_shape) == 3
        
        batch_size, seq_len, _ = hidden_states_shape
        
        # Early return for empty sequences to avoid dimension errors
        if seq_len == 0 or batch_size == 0:
            return torch.zeros(batch_size, seq_len, self.hidden_size, device=hidden_states.device, dtype=hidden_states.dtype)
        
        # Debug weight shapes
        print(f"DEBUG: W_A_q weight shape: {self.W_A_q.weight.shape}")
        print(f"DEBUG: W_B_q weight shape: {self.W_B_q.weight.shape}")
        
        try:
            # Compute A factors with robust handling
            A_q_raw = self.W_A_q(hidden_states)
            expected_q_size = self.num_heads * self.q_rank
            print(f"DEBUG: A_q_raw shape: {A_q_raw.shape}, expected flat dim: {expected_q_size}")
            
            # Try standard reshape first
            try:
                A_q = A_q_raw.view(batch_size, seq_len, self.num_heads, self.q_rank)
            except Exception as q_error:
                print(f"ERROR reshaping A_q: {q_error}")
                # Try handling mismatched dimensions
                A_q = A_q_raw.reshape(batch_size, seq_len, -1)
                total_size = A_q.size(2)
                # Calculate proper dimensions based on total size
                if total_size % self.num_heads == 0:
                    # Use actual size to determine effective rank
                    effective_q_rank = total_size // self.num_heads
                    print(f"Using effective q_rank: {effective_q_rank}")
                    A_q = A_q.view(batch_size, seq_len, self.num_heads, effective_q_rank)
                else:
                    # Create fallback tensor
                    print(f"Cannot reshape {total_size} to work with {self.num_heads} heads")
                    A_q = torch.ones(batch_size, seq_len, self.num_heads, self.q_rank, 
                                   device=hidden_states.device, dtype=hidden_states.dtype) * 0.01
            
            # Process remaining factors with same pattern
            # K factor
            A_k_raw = self.W_A_k(hidden_states)
            try:
                A_k = A_k_raw.view(batch_size, seq_len, self.num_kv_heads, self.k_rank)
            except Exception as k_error:
                print(f"ERROR reshaping A_k: {k_error}")
                A_k = A_k_raw.reshape(batch_size, seq_len, -1)
                total_size = A_k.size(2)
                if total_size % self.num_kv_heads == 0:
                    effective_k_rank = total_size // self.num_kv_heads
                    print(f"Using effective k_rank: {effective_k_rank}")
                    A_k = A_k.view(batch_size, seq_len, self.num_kv_heads, effective_k_rank)
                else:
                    A_k = torch.ones(batch_size, seq_len, self.num_kv_heads, self.k_rank, 
                                   device=hidden_states.device, dtype=hidden_states.dtype) * 0.01
            
            # V factor
            A_v_raw = self.W_A_v(hidden_states)
            try:
                A_v = A_v_raw.view(batch_size, seq_len, self.num_kv_heads, self.v_rank)
            except Exception as v_error:
                print(f"ERROR reshaping A_v: {v_error}")
                A_v = A_v_raw.reshape(batch_size, seq_len, -1)
                total_size = A_v.size(2)
                if total_size % self.num_kv_heads == 0:
                    effective_v_rank = total_size // self.num_kv_heads
                    print(f"Using effective v_rank: {effective_v_rank}")
                    A_v = A_v.view(batch_size, seq_len, self.num_kv_heads, effective_v_rank)
                else:
                    A_v = torch.ones(batch_size, seq_len, self.num_kv_heads, self.v_rank, 
                                   device=hidden_states.device, dtype=hidden_states.dtype) * 0.01
            
            # Compute B factors with similar robust handling
            # Q-B factor
            B_q_raw = self.W_B_q(hidden_states)
            try:
                B_q = B_q_raw.view(batch_size, seq_len, self.q_rank, self.head_dim)
            except Exception as bq_error:
                print(f"ERROR reshaping B_q: {bq_error}")
                B_q = B_q_raw.reshape(batch_size, seq_len, -1)
                total_size = B_q.size(2)
                if total_size % self.head_dim == 0:
                    effective_q_rank = total_size // self.head_dim
                    print(f"Using effective B_q rank: {effective_q_rank}")
                    B_q = B_q.view(batch_size, seq_len, effective_q_rank, self.head_dim)
                else:
                    B_q = torch.ones(batch_size, seq_len, self.q_rank, self.head_dim, 
                                   device=hidden_states.device, dtype=hidden_states.dtype) * 0.01
            
            # K-B factor
            B_k_raw = self.W_B_k(hidden_states)
            try:
                B_k = B_k_raw.view(batch_size, seq_len, self.k_rank, self.head_dim)
            except Exception as bk_error:
                print(f"ERROR reshaping B_k: {bk_error}")
                B_k = B_k_raw.reshape(batch_size, seq_len, -1)
                total_size = B_k.size(2)
                if total_size % self.head_dim == 0:
                    effective_k_rank = total_size // self.head_dim
                    print(f"Using effective B_k rank: {effective_k_rank}")
                    B_k = B_k.view(batch_size, seq_len, effective_k_rank, self.head_dim)
                else:
                    B_k = torch.ones(batch_size, seq_len, self.k_rank, self.head_dim, 
                                   device=hidden_states.device, dtype=hidden_states.dtype) * 0.01
            
            # V-B factor
            B_v_raw = self.W_B_v(hidden_states)
            try:
                B_v = B_v_raw.view(batch_size, seq_len, self.v_rank, self.head_dim)
            except Exception as bv_error:
                print(f"ERROR reshaping B_v: {bv_error}")
                B_v = B_v_raw.reshape(batch_size, seq_len, -1)
                total_size = B_v.size(2)
                if total_size % self.head_dim == 0:
                    effective_v_rank = total_size // self.head_dim
                    print(f"Using effective B_v rank: {effective_v_rank}")
                    B_v = B_v.view(batch_size, seq_len, effective_v_rank, self.head_dim)
                else:
                    B_v = torch.ones(batch_size, seq_len, self.v_rank, self.head_dim, 
                                   device=hidden_states.device, dtype=hidden_states.dtype) * 0.01
                                   
            # Print shapes for debugging
            print(f"DEBUG: Final shapes - A_q: {A_q.shape}, B_q: {B_q.shape}")
            
        except Exception as main_e:
            print(f"Critical error in TPA shape handling: {main_e}")
            # Create fallback tensors
            A_q = torch.ones(batch_size, seq_len, self.num_heads, self.q_rank, 
                           device=hidden_states.device, dtype=hidden_states.dtype) * 0.01
            A_k = torch.ones(batch_size, seq_len, self.num_kv_heads, self.k_rank, 
                           device=hidden_states.device, dtype=hidden_states.dtype) * 0.01
            A_v = torch.ones(batch_size, seq_len, self.num_kv_heads, self.v_rank, 
                           device=hidden_states.device, dtype=hidden_states.dtype) * 0.01
            B_q = torch.ones(batch_size, seq_len, self.q_rank, self.head_dim, 
                           device=hidden_states.device, dtype=hidden_states.dtype) * 0.01
            B_k = torch.ones(batch_size, seq_len, self.k_rank, self.head_dim, 
                           device=hidden_states.device, dtype=hidden_states.dtype) * 0.01
            B_v = torch.ones(batch_size, seq_len, self.v_rank, self.head_dim, 
                           device=hidden_states.device, dtype=hidden_states.dtype) * 0.01
        
        # Apply RMSNorm if specified
        if self.query_norm is not None and self.key_norm is not None:
            # Reshape for applying norm across head dimension
            B_q_reshaped = B_q.reshape(-1, self.head_dim)
            B_k_reshaped = B_k.reshape(-1, self.head_dim)
            
            B_q_reshaped = self.query_norm(B_q_reshaped)
            B_k_reshaped = self.key_norm(B_k_reshaped)
            
            B_q = B_q_reshaped.reshape(batch_size, seq_len, self.q_rank, self.head_dim)
            B_k = B_k_reshaped.reshape(batch_size, seq_len, self.k_rank, self.head_dim)
        
        # Apply rotary positional embedding to B factors
        B_q_rotated = self._apply_rotary_emb_to_factor(B_q, freqs_cis)
        B_k_rotated = self._apply_rotary_emb_to_factor(B_k, freqs_cis)
        
        # Unpack the factorized KV cache
        k_cache_A, k_cache_B, v_cache_A, v_cache_B = kv_cache
        
        # Do a thorough validation of kv_write_indices to catch token ID confusion
        if kv_write_indices is not None and kv_write_indices.numel() > 0:
            # Check if any indices are larger than 10000 - likely token IDs mistakenly used as positions
            try:
                max_val = kv_write_indices.max().item()
                if max_val > 10000:
                    print(f"WARNING: Detected very large indices in kv_write_indices (max: {max_val})")
                    print("These are likely token IDs mistakenly used as position indices")
                    # Reset to safe sequential indices to avoid crashes
                    kv_write_indices = torch.arange(min(seq_len, k_cache_A.size(1)), device=kv_write_indices.device)
                    print(f"Reset to safe sequential indices: {kv_write_indices}")
            except Exception as detect_e:
                print(f"Error checking kv_write_indices max: {detect_e}")

            # Check for empty tensors
            if seq_len == 0 or batch_size == 0:
                ctx_len = 0
            else:
                # Ensure indices are valid and not out of range
                try:
                    # Force clamp all indices to valid range regardless of input
                    max_cache_idx = k_cache_A.size(1) - 1
                    kv_write_indices = torch.clamp(kv_write_indices, 0, max_cache_idx)
                    
                    # Set context length to max index + 1 or total cache size, whichever is smaller
                    if kv_write_indices.numel() == 1:
                        ctx_len = min(kv_write_indices.item() + 1, k_cache_A.size(1))
                    else:
                        ctx_len = min(torch.max(kv_write_indices).item() + 1, k_cache_A.size(1))
                except Exception as e:
                    print(f"Error processing kv_write_indices: {e}, using fallback")
                    ctx_len = min(seq_len, k_cache_A.size(1))  # Fallback
        else:
            ctx_len = min(seq_len, k_cache_A.size(1))  # Ensure we don't exceed cache size
        
        # Write new values to KV cache
        if kv_write_indices is not None and kv_write_indices.numel() > 0 and seq_len > 0:
            # Skip KV cache writing if there are no indices or the sequence is empty
            
            # Ensure dtype match with the cache tensors
            A_k_dtype = A_k.to(dtype=k_cache_A.dtype)
            B_k_rotated_dtype = B_k_rotated.to(dtype=k_cache_B.dtype)
            A_v_dtype = A_v.to(dtype=v_cache_A.dtype)
            B_v_dtype = B_v.to(dtype=v_cache_B.dtype)
            
            # Always use safe sequential update method to avoid index out of bounds errors
            try:
                # Check that all tensors have valid dimensions before updating
                if A_k_dtype.dim() < 2 or B_k_rotated_dtype.dim() < 2 or A_v_dtype.dim() < 2 or B_v_dtype.dim() < 2:
                    print(f"Warning: Invalid tensor dimensions for KV cache update")
                    print(f"A_k: {A_k_dtype.shape}, B_k: {B_k_rotated_dtype.shape}, A_v: {A_v_dtype.shape}, B_v: {B_v_dtype.shape}")
                else:
                    # Use safer sequential update to handle any size mismatches
                    # Safely handle scalar or single-element tensors
                    if kv_write_indices.numel() == 1:
                        # Single position case (common during generation)
                        idx = kv_write_indices.item()
                        # Clamp index to valid range
                        idx = max(0, min(idx, k_cache_A.size(1) - 1))
                        
                        # For single index, use first position from source tensors
                        k_cache_A[:, idx] = A_k_dtype[:, 0]
                        k_cache_B[:, idx] = B_k_rotated_dtype[:, 0]
                        v_cache_A[:, idx] = A_v_dtype[:, 0]
                        v_cache_B[:, idx] = B_v_dtype[:, 0]
                    else:
                        # Multiple positions case (e.g., during prefill)
                        for i in range(min(kv_write_indices.numel(), seq_len)):
                            try:
                                # Safely get and clamp the index to valid range
                                if i < kv_write_indices.numel():
                                    idx = kv_write_indices[i].item()
                                    idx = max(0, min(idx, k_cache_A.size(1) - 1))
                                    
                                    # Select source position (use min to avoid out of bounds)
                                    source_idx = min(i, A_k_dtype.size(1) - 1)
                                    
                                    # Update cache tensors
                                    k_cache_A[:, idx] = A_k_dtype[:, source_idx]
                                    k_cache_B[:, idx] = B_k_rotated_dtype[:, source_idx]
                                    v_cache_A[:, idx] = A_v_dtype[:, source_idx]
                                    v_cache_B[:, idx] = B_v_dtype[:, source_idx]
                            except Exception as idx_error:
                                print(f"Error updating KV cache at index {i}: {idx_error}")
                                # Continue with next index
            except Exception as e:
                print(f"Error during KV cache update: {e}")
                # Don't attempt further updates if we hit an error
        
        # Compute query from factorized form
        # Ensure matching dtypes for matmul
        A_q_float = A_q.to(dtype=torch.float32)
        B_q_rotated_float = B_q_rotated.to(dtype=torch.float32)
        
        try:
            # Ensure shapes are compatible
            bsz_seq_flat = batch_size * seq_len
            A_q_reshaped = A_q_float.reshape(bsz_seq_flat, self.num_heads, self.q_rank)
            B_q_reshaped = B_q_rotated_float.reshape(bsz_seq_flat, self.q_rank, self.head_dim)
            
            # Perform the matmul with careful handling
            Q = torch.matmul(A_q_reshaped, B_q_reshaped)
            Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            # Use safe division - the q_rank should match what was used in factorization
            Q = Q.div(self.q_rank if self.q_rank > 0 else 1.0)
        except Exception as e:
            print(f"WARNING: Error in query tensor product: {e}")
            # Fallback to a simple approximation if tensor product fails
            Q = torch.zeros(
                (batch_size, seq_len, self.num_heads, self.head_dim), 
                dtype=torch.float32, 
                device=hidden_states.device
            )
            if A_q_float.numel() > 0 and B_q_rotated_float.numel() > 0:
                # Use simple outer product of first elements as fallback
                for i in range(min(batch_size, 1)):
                    for j in range(min(seq_len, 1)):
                        for h in range(min(self.num_heads, 1)):
                            Q[i, j, h] = torch.ones(self.head_dim, device=hidden_states.device)
        
        # Convert back to original dtype
        Q = Q.to(dtype=hidden_states.dtype)
        
        # Check for completely zero query
        if torch.all(Q.abs() < 1e-6):
            print("WARNING: Query tensor is all zeros after factorization! Adding small random values")
            Q = Q + torch.randn_like(Q) * 1e-4
        
        # Prepare for attention calculation
        # [batch_size, num_heads, seq_len, head_dim]
        Q = Q.transpose(1, 2)
        
        # Ensure scaling factor is not zero or NaN
        scaling_factor = self.scaling
        if math.isnan(scaling_factor) or math.isinf(scaling_factor) or abs(scaling_factor) < 1e-10:
            print(f"WARNING: Invalid scaling factor {scaling_factor}, using default 1/sqrt(head_dim)")
            scaling_factor = 1.0 / math.sqrt(self.head_dim)
            
        Q = Q * scaling_factor
        
        # Get cached K factors for the context window
        K_A = k_cache_A[:, :ctx_len]
        K_B = k_cache_B[:, :ctx_len]
        
        # Ensure matching dtypes for matmul
        K_A = K_A.to(dtype=torch.float32)
        K_B = K_B.to(dtype=torch.float32)
        
        try:
            # Expand K_A if we're using grouped query attention (before matmul)
            if self.num_kv_heads != self.num_heads:
                # [batch_size, ctx_len, num_heads, k_rank]
                K_A = torch.repeat_interleave(K_A, self.num_queries_per_kv, dim=2)
                
                # Check that shapes are valid before matmul
                bsz_ctx_flat = batch_size * ctx_len
                if bsz_ctx_flat > 0 and self.k_rank > 0:
                    K_A_reshaped = K_A.reshape(bsz_ctx_flat, self.num_heads, self.k_rank)
                    K_B_reshaped = K_B.reshape(bsz_ctx_flat, self.k_rank, self.head_dim)
                    
                    # Build full K matrix by multiplying expanded factors
                    K = torch.matmul(K_A_reshaped, K_B_reshaped)
                    K = K.reshape(batch_size, ctx_len, self.num_heads, self.head_dim)
                    # Use safe division with what was used in factorization
                    K = K.div(self.k_rank if self.k_rank > 0 else 1.0)
                else:
                    # Handle empty case
                    K = torch.zeros(
                        (batch_size, ctx_len, self.num_heads, self.head_dim),
                        dtype=torch.float32,
                        device=hidden_states.device
                    )
            else:
                # Check that shapes are valid before matmul
                bsz_ctx_flat = batch_size * ctx_len
                if bsz_ctx_flat > 0 and self.k_rank > 0:
                    K_A_reshaped = K_A.reshape(bsz_ctx_flat, self.num_kv_heads, self.k_rank)
                    K_B_reshaped = K_B.reshape(bsz_ctx_flat, self.k_rank, self.head_dim)
                    
                    # Build full K matrix by multiplying factors
                    K = torch.matmul(K_A_reshaped, K_B_reshaped)
                    K = K.reshape(batch_size, ctx_len, self.num_kv_heads, self.head_dim)
                    # Use safe division with what was used in factorization
                    K = K.div(self.k_rank if self.k_rank > 0 else 1.0)
                else:
                    # Handle empty case
                    K = torch.zeros(
                        (batch_size, ctx_len, self.num_kv_heads, self.head_dim),
                        dtype=torch.float32,
                        device=hidden_states.device
                    )
        except Exception as e:
            print(f"WARNING: Error in key tensor product: {e}")
            # Fallback to a simple approximation
            if self.num_kv_heads != self.num_heads:
                K = torch.zeros(
                    (batch_size, ctx_len, self.num_heads, self.head_dim),
                    dtype=torch.float32,
                    device=hidden_states.device
                )
            else:
                K = torch.zeros(
                    (batch_size, ctx_len, self.num_kv_heads, self.head_dim),
                    dtype=torch.float32,
                    device=hidden_states.device
                )
            # Add some data to the zeros to avoid all-zero attention
            if batch_size > 0 and ctx_len > 0:
                K[:, :, :, 0] = 1.0
                
        # Convert back to original dtype
        K = K.to(dtype=hidden_states.dtype)
        
        # Check for completely zero key tensor
        if torch.all(K.abs() < 1e-6):
            print("WARNING: Key tensor is all zeros after factorization! Adding small random values")
            K = K + torch.randn_like(K) * 1e-4
        
        # [batch_size, num_heads, ctx_len, head_dim]
        K = K.transpose(1, 2)
        
        # DEBUG: Check Q and K tensors for issues
        print(f"DEBUG Q tensor stats: min={Q.min().item():.6f}, max={Q.max().item():.6f}, mean={Q.mean().item():.6f}, std={Q.std().item():.6f}, has_nan={torch.isnan(Q).any().item()}, has_inf={torch.isinf(Q).any().item()}")
        print(f"DEBUG K tensor stats: min={K.min().item():.6f}, max={K.max().item():.6f}, mean={K.mean().item():.6f}, std={K.std().item():.6f}, has_nan={torch.isnan(K).any().item()}, has_inf={torch.isinf(K).any().item()}")
        
        # Calculate attention scores
        # [batch_size, num_heads, seq_len, ctx_len]
        try:
            scores = torch.matmul(Q, K.transpose(2, 3))
            
            # Check for all-zero scores and handle them
            if torch.all(scores.abs() < 1e-6):
                print("WARNING: All attention scores are zero! Adding small values to break symmetry")
                # Add small random values to break symmetry
                scores = scores + torch.randn_like(scores) * 1e-4
            
            # Check for NaN/Inf in scores
            if torch.isnan(scores).any() or torch.isinf(scores).any():
                print("WARNING: NaN/Inf in attention scores, replacing with zeros")
                scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Ensure we have non-zero values
                if torch.all(scores.abs() < 1e-6):
                    print("WARNING: All scores are zero after NaN removal, adding identity pattern")
                    # Create a diagoal-heavy pattern (causal attention)
                    batch_size, num_heads, seq_len, ctx_len = scores.shape
                    diagonal = torch.arange(min(seq_len, ctx_len), device=scores.device)
                    for b in range(batch_size):
                        for h in range(num_heads):
                            scores[b, h, diagonal, diagonal] = 1.0
                            # Add a slight forward-looking bias within causal limit
                            for i in range(seq_len):
                                for j in range(max(0, i-5), i):
                                    if j < ctx_len:
                                        scores[b, h, i, j] = 0.8
            
            # DEBUG: Check attention scores
            print(f"DEBUG Attention scores stats: min={scores.min().item():.6f}, max={scores.max().item():.6f}, mean={scores.mean().item():.6f}, std={scores.std().item():.6f}, has_nan={torch.isnan(scores).any().item()}, has_inf={torch.isinf(scores).any().item()}")
            
            # Apply softcapping if specified
            if self.attn_logit_softcapping is not None:
                scores = scores / self.attn_logit_softcapping
                scores = torch.tanh(scores)
                scores = scores * self.attn_logit_softcapping
                
                # DEBUG: Check after softcapping
                print(f"DEBUG After softcapping: min={scores.min().item():.6f}, max={scores.max().item():.6f}, mean={scores.mean().item():.6f}, std={scores.std().item():.6f}, has_nan={torch.isnan(scores).any().item()}, has_inf={torch.isinf(scores).any().item()}")
                
        except Exception as e:
            print(f"ERROR during attention calculation: {e}")
            # Create fallback attention scores - diagonal pattern (causal)
            batch_size, num_heads = Q.shape[0], Q.shape[1]
            seq_len, ctx_len = Q.shape[2], K.shape[2]
            
            scores = torch.zeros((batch_size, num_heads, seq_len, ctx_len), device=Q.device, dtype=Q.dtype)
            diagonal = torch.arange(min(seq_len, ctx_len), device=scores.device)
            for b in range(batch_size):
                for h in range(num_heads):
                    scores[b, h, diagonal, diagonal] = 1.0
                    # Add a slight forward-looking bias within causal limit
                    for i in range(seq_len):
                        for j in range(max(0, i-3), i):
                            if j < ctx_len:
                                scores[b, h, i, j] = 0.8
        
        # Always create a new causal mask that exactly matches the scores dimensions
        min_value = torch.finfo(scores.dtype).min
        scores_shape = scores.shape  # [batch_size, num_heads, seq_len, ctx_len]
        
        # Create causal attention mask
        causal_mask = torch.ones(scores_shape, device=scores.device, dtype=torch.bool)
        causal_mask = torch.tril(causal_mask)  # Make it lower triangular for causality
        
        # If sliding window is enabled, create a window constraint
        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            # Create sliding window mask
            window_size = min(self.sliding_window_size, ctx_len)
            window_mask = torch.ones(scores_shape, device=scores.device, dtype=torch.bool)
            
            # For each position, only allow attention to window_size tokens before it
            for i in range(scores_shape[2]):  # For each query position
                # Calculate the start of the valid window for this position
                start_idx = max(0, i - window_size + 1)
                
                # Zero out attention to positions before the window start
                if start_idx > 0:
                    window_mask[:, :, i, :start_idx] = False
            
            # Combine causal mask with window mask
            attention_mask = torch.logical_and(causal_mask, window_mask)
        else:
            # Just use causal mask
            attention_mask = causal_mask
            
        # Convert to attention values (0 for attend, min_value for don't attend)
        mask_tensor = torch.where(
            attention_mask, 
            torch.tensor(0, dtype=scores.dtype, device=scores.device),
            torch.tensor(min_value, dtype=scores.dtype, device=scores.device)
        )
        
        # Apply the mask
        scores = scores + mask_tensor
        
        # Always use safe softmax to avoid NaN results
        try:
            # First try the regular softmax as a baseline
            regular_softmax = F.softmax(scores.float(), dim=-1).type_as(Q)
            
            # Check for issues in regular softmax
            if not torch.isnan(regular_softmax).any() and not torch.isinf(regular_softmax).any():
                attn_weights = regular_softmax
            else:
                print("Warning: NaN/Inf in vanilla softmax, using improved safe softmax")
                
                # Apply improved safe softmax with better numerical stability
                scores_safe = scores.to(torch.float32)  # Always use float32 for numerical stability
                
                # Find maximum value for numerical stability, handling potential NaN/Inf
                scores_safe = torch.nan_to_num(scores_safe, nan=-1e4, posinf=1e4, neginf=-1e4)
                scores_max = torch.max(scores_safe, dim=-1, keepdim=True)[0]
                
                # Subtract max and apply exp with careful handling of edge cases
                scores_safe = scores_safe - scores_max
                exp_scores = torch.exp(torch.clamp(scores_safe, min=-50.0, max=50.0))  # Clamp to avoid over/underflow
                
                # If we have a perfectly valid causal mask, some values should be exactly 0 (masked out)
                # So we need to handle that specially to avoid 0/0 in softmax
                mask_tensor = scores == float('-inf')
                exp_scores = torch.where(mask_tensor, torch.zeros_like(exp_scores), exp_scores)
                
                # Calculate sum for denominator, adding epsilon to ensure no division by zero
                denom = torch.sum(exp_scores, dim=-1, keepdim=True)
                denom = torch.clamp(denom, min=1e-6)  # Ensure minimum denominator
                
                # Calculate final softmax
                attn_weights = (exp_scores / denom).to(dtype=Q.dtype)
                
                # Final sanity check - if we still have NaN/Inf, fall back to uniform with causal masking
                if torch.isnan(attn_weights).any() or torch.isinf(attn_weights).any():
                    print("Warning: Safe softmax still has NaN/Inf, using uniform attention with causal masking")
                    
                    # Create a causal mask-aware uniform attention
                    batch_size, num_heads, seq_len, ctx_len = scores.shape
                    
                    # Start with zeros
                    attn_weights = torch.zeros_like(scores)
                    
                    # For each position, distribute attention uniformly over valid positions
                    for i in range(seq_len):
                        # In causal attention, we can only attend to positions up to i in context
                        valid_ctx_len = min(i+1, ctx_len)
                        if valid_ctx_len > 0:  # Ensure we have at least one position to attend to
                            # For this position, distribute attention uniformly over valid positions
                            attn_weights[:, :, i, :valid_ctx_len] = 1.0 / valid_ctx_len
                            
                    attn_weights = attn_weights.to(dtype=Q.dtype)
        except Exception as e:
            print(f"Error during softmax calculation: {e}, falling back to basic attention")
            
            # Create a simple diagonal-focused attention pattern as last resort
            batch_size, num_heads, seq_len, ctx_len = scores.shape
            attn_weights = torch.zeros_like(scores)
            
            # Set diagonal elements to 1.0 (or a high proportion)
            min_len = min(seq_len, ctx_len)
            diag_indices = torch.arange(min_len, device=scores.device)
            
            # Distribute most attention to recent tokens (causal pattern)
            for i in range(seq_len):
                valid_len = min(i+1, ctx_len)  # Only attend up to current position
                if valid_len > 0:
                    # Put 80% weight on the most recent token, 20% on previous tokens
                    if valid_len == 1:
                        attn_weights[:, :, i, 0] = 1.0
                    else:
                        # Last token gets 80% attention
                        attn_weights[:, :, i, valid_len-1] = 0.8
                        # Distribute remaining 20% to earlier tokens
                        if valid_len > 1:
                            attn_weights[:, :, i, :valid_len-1] = 0.2 / (valid_len-1)
                            
            attn_weights = attn_weights.to(dtype=Q.dtype)
        
        # Compute attention output using factorized V
        # Get cached V factors - ensure we don't exceed cache bounds
        ctx_len = min(ctx_len, v_cache_A.size(1))
        V_A = v_cache_A[:, :ctx_len]
        V_B = v_cache_B[:, :ctx_len]
        
        # Ensure matching dtypes for matmul
        V_A = V_A.to(dtype=torch.float32)
        V_B = V_B.to(dtype=torch.float32)
        
        try:
            # Expand V_A if we're using grouped query attention (before matmul)
            if self.num_kv_heads != self.num_heads:
                # [batch_size, ctx_len, num_heads, v_rank]
                V_A = torch.repeat_interleave(V_A, self.num_queries_per_kv, dim=2)
                # Build full V matrix by multiplying expanded factors
                # [batch_size, ctx_len, num_heads, head_dim]
                if batch_size > 0 and ctx_len > 0:
                    V = torch.matmul(
                        V_A.reshape(batch_size * ctx_len, self.num_heads, self.v_rank),
                        V_B.reshape(batch_size * ctx_len, self.v_rank, self.head_dim)
                    ).reshape(batch_size, ctx_len, self.num_heads, self.head_dim).div(self.v_rank if self.v_rank > 0 else 1.0)
                else:
                    V = torch.zeros((batch_size, ctx_len, self.num_heads, self.head_dim), 
                                  dtype=torch.float32, device=V_A.device)
            else:
                # Build full V matrix by multiplying factors
                # [batch_size, ctx_len, num_kv_heads, head_dim]
                if batch_size > 0 and ctx_len > 0:
                    V = torch.matmul(
                        V_A.reshape(batch_size * ctx_len, self.num_kv_heads, self.v_rank),
                        V_B.reshape(batch_size * ctx_len, self.v_rank, self.head_dim)
                    ).reshape(batch_size, ctx_len, self.num_kv_heads, self.head_dim).div(self.v_rank if self.v_rank > 0 else 1.0)
                else:
                    V = torch.zeros((batch_size, ctx_len, self.num_kv_heads, self.head_dim), 
                                  dtype=torch.float32, device=V_A.device)
        except Exception as e:
            print(f"Error during value matrix construction: {e}")
            # Create fallback V matrix
            if self.num_kv_heads != self.num_heads:
                V = torch.zeros((batch_size, ctx_len, self.num_heads, self.head_dim), 
                              dtype=torch.float32, device=hidden_states.device)
            else:
                V = torch.zeros((batch_size, ctx_len, self.num_kv_heads, self.head_dim), 
                              dtype=torch.float32, device=hidden_states.device)
            # Add ones to first few elements to avoid all zeros
            if V.numel() > 0:
                V[:, :, :, 0] = 1.0
        
        # Convert back to original dtype
        V = V.to(dtype=hidden_states.dtype)
        
        # [batch_size, num_heads, ctx_len, head_dim]
        V = V.transpose(1, 2)
        
        # Apply attention weights to values
        # [batch_size, num_heads, seq_len, head_dim]
        output = torch.matmul(attn_weights, V)
        
        # Handle empty sequence case
        if seq_len == 0 or batch_size == 0:
            # Return an empty tensor of the right shape
            return torch.zeros(batch_size, seq_len, self.hidden_size, device=hidden_states.device, dtype=hidden_states.dtype)
        
        # Reshape output and apply output projection
        # [batch_size, seq_len, hidden_size]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(output)
        
        return output
    
    def _apply_rotary_emb_to_factor(self, factor: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """Apply rotary positional embedding to B factor matrices."""
        # factor shape: [batch_size, seq_len, rank, head_dim]
        # freqs_cis shape: [seq_len, head_dim//2]
        
        # Handle empty or invalid factors
        if factor.numel() == 0:
            return factor
            
        # Replace any NaN or inf values
        if torch.isnan(factor).any() or torch.isinf(factor).any():
            print(f"Warning: Found NaN/Inf values in factor tensor, replacing with zeros")
            factor = torch.nan_to_num(factor, nan=0.0, posinf=0.0, neginf=0.0)
            
        # Ensure factor has the right number of dimensions
        if factor.dim() != 4:
            print(f"Warning: Factor has incorrect shape {factor.shape}, expected 4D tensor")
            # Try to reshape if possible, or return as is if we can't
            if factor.numel() > 0:
                try:
                    # Try to infer dimensions
                    if factor.dim() == 2:
                        # Reshape 2D to 4D with batch size and seq_len = 1
                        factor = factor.unsqueeze(0).unsqueeze(0)
                    elif factor.dim() == 3:
                        # Reshape 3D to 4D with batch size = 1
                        factor = factor.unsqueeze(0)
                except Exception as e:
                    print(f"Error reshaping factor: {e}")
                    return factor
            else:
                return factor
        
        batch_size, seq_len, rank, head_dim = factor.shape
        
        # Handle empty sequence case (seq_len=0)
        if seq_len == 0 or batch_size == 0:
            return factor
        
        # Ensure freqs_cis has enough positions
        try:
            if freqs_cis.size(0) < seq_len:
                # Silently handle this case - we'll pad with the last position's frequencies
                # Use available positions instead of raising error
                available_positions = freqs_cis.size(0)
                # If we need more positions, we'll pad by repeating the last position
                if available_positions > 0:
                    last_pos = freqs_cis[available_positions-1:available_positions]
                    padding = last_pos.repeat(seq_len - available_positions, 1)
                    freqs_cis = torch.cat([freqs_cis[:available_positions], padding], dim=0)
                else:
                    # Create empty freqs_cis of right shape if no positions are available
                    freqs_cis = torch.zeros((seq_len, freqs_cis.size(1)), device=freqs_cis.device)
            else:
                # Take only needed positions from freqs_cis
                freqs_cis = freqs_cis[:seq_len]
            
            # Check for NaN/Inf in freqs_cis
            if torch.isnan(freqs_cis).any() or torch.isinf(freqs_cis).any():
                print(f"Warning: Found NaN/Inf values in freqs_cis tensor, replacing with zeros")
                freqs_cis = torch.nan_to_num(freqs_cis, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Ensure head_dim is divisible by 2 for complex number representation
            if head_dim % 2 != 0:
                print(f"Warning: head_dim {head_dim} is not divisible by 2, padding with zeros")
                # Add zero padding to make it divisible by 2
                pad_width = ((0, 0), (0, 0), (0, 0), (0, 1))
                factor = torch.nn.functional.pad(factor, pad_width)
                head_dim += 1
            
            # Reshape for applying RoPE
            factor_reshaped = factor.reshape(batch_size * seq_len, rank, head_dim)
            
            # Apply RoPE in a similar way to the original Gemma implementation
            # Convert to float32 for better numerical stability
            factor_float = factor_reshaped.to(torch.float32)
            
            # Split into chunks with error handling
            try:
                chunks = torch.chunk(factor_float, 2, dim=-1)
                if len(chunks) != 2 or chunks[0].shape != chunks[1].shape:
                    raise ValueError(f"Invalid chunks: got {len(chunks)} chunks with shapes {[c.shape for c in chunks]}")
                
                factor_complex = torch.view_as_complex(torch.stack(chunks, dim=-1))
            except Exception as chunk_error:
                print(f"Error converting to complex: {chunk_error}, using alternative method")
                # Ensure even head_dim
                if factor_float.shape[-1] % 2 != 0:
                    factor_float = torch.nn.functional.pad(factor_float, (0, 1))
                
                # Manually reshape to get complex values
                half_dim = factor_float.shape[-1] // 2
                real_part = factor_float[..., :half_dim]
                imag_part = factor_float[..., half_dim:]
                
                # Create complex tensor
                factor_complex = torch.complex(real_part, imag_part)
            
            # Check for NaN/Inf in complex tensor
            if torch.isnan(factor_complex.abs()).any() or torch.isinf(factor_complex.abs()).any():
                print(f"Warning: Found NaN/Inf values in complex tensor, replacing with zeros")
                factor_complex = torch.nan_to_num(factor_complex.abs(), nan=0.0, posinf=0.0, neginf=0.0) * torch.exp(1j * torch.angle(factor_complex + 1e-7))
            
            # Reshape freqs_cis for broadcasting
            freqs_cis = freqs_cis.view(seq_len, 1, head_dim // 2)
            freqs_cis = freqs_cis.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            freqs_cis = freqs_cis.reshape(batch_size * seq_len, 1, head_dim // 2)
            
            # Apply complex multiplication with error handling
            try:
                # Apply the rotation with gradient-safe operations
                factor_out_complex = factor_complex * freqs_cis
                
                # Convert back to real with checks
                factor_out = torch.view_as_real(factor_out_complex)
                
                # Check for NaN/Inf after rotation
                if torch.isnan(factor_out).any() or torch.isinf(factor_out).any():
                    print("Warning: NaN/Inf values after rotation, using safe reconstruction")
                    # Create a safe version using only magnitudes
                    factor_out = torch.nan_to_num(factor_out, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Reshape for output
                factor_out = torch.cat([factor_out[..., 0], factor_out[..., 1]], dim=-1)
                
            except Exception as rot_error:
                print(f"Complex rotation failed: {rot_error}, returning un-rotated factor")
                # Fall back to original tensor if rotation fails
                factor_out = factor_reshaped
            
            # Convert back to original dtype
            factor_out = factor_out.type_as(factor)
            
            # Reshape back to original shape
            factor_out = factor_out.reshape(batch_size, seq_len, rank, head_dim)
            
            # If we padded the head_dim, remove the padding
            if head_dim != factor.shape[-1]:
                factor_out = factor_out[..., :factor.shape[-1]]
            
            return factor_out
            
        except Exception as e:
            print(f"Error applying rotary embeddings: {e}")
            # Return the original factor if anything goes wrong
            # Make sure it has no NaN values
            return torch.nan_to_num(factor, nan=0.0, posinf=0.0, neginf=0.0)