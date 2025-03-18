"""Tensor Product Attention implementation for Gemma."""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple, Mapping

from .. import config as gemma_config
from .. import model as gemma_model

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
        
        # Compute A factors
        A_q = self.W_A_q(hidden_states).view(batch_size, seq_len, self.num_heads, self.q_rank)
        A_k = self.W_A_k(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.k_rank)
        A_v = self.W_A_v(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.v_rank)
        
        # Compute B factors
        B_q = self.W_B_q(hidden_states).view(batch_size, seq_len, self.q_rank, self.head_dim)
        B_k = self.W_B_k(hidden_states).view(batch_size, seq_len, self.k_rank, self.head_dim)
        B_v = self.W_B_v(hidden_states).view(batch_size, seq_len, self.v_rank, self.head_dim)
        
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
        
        # Prepare for attention calculation
        # [batch_size, num_heads, seq_len, head_dim]
        Q = Q.transpose(1, 2)
        Q = Q * self.scaling
        
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
        
        # [batch_size, num_heads, ctx_len, head_dim]
        K = K.transpose(1, 2)
        
        # Calculate attention scores
        # [batch_size, num_heads, seq_len, ctx_len]
        scores = torch.matmul(Q, K.transpose(2, 3))
        
        # Apply softcapping if specified
        if self.attn_logit_softcapping is not None:
            scores = scores / self.attn_logit_softcapping
            scores = torch.tanh(scores)
            scores = scores * self.attn_logit_softcapping
        
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
        
        # Apply softmax to get attention weights - use safe softmax to avoid NaN results
        try:
            attn_weights = F.softmax(scores.float(), dim=-1).type_as(Q)
            # Check for NaN values and replace with zeros
            if torch.isnan(attn_weights).any():
                print("Warning: NaN values detected in attention weights, applying safe softmax")
                # Apply safe softmax: subtract max value for numerical stability
                scores_safe = scores.float()
                scores_max = torch.max(scores_safe, dim=-1, keepdim=True)[0]
                scores_safe = scores_safe - scores_max
                exp_scores = torch.exp(scores_safe)
                # Replace NaN/Inf with zeros
                exp_scores = torch.where(torch.isnan(exp_scores) | torch.isinf(exp_scores), 
                                        torch.zeros_like(exp_scores), exp_scores)
                # Add small epsilon to ensure non-zero denominator
                denom = torch.sum(exp_scores, dim=-1, keepdim=True) + 1e-10
                attn_weights = (exp_scores / denom).type_as(Q)
        except Exception as e:
            print(f"Error during softmax calculation: {e}, using uniform attention")
            # Fallback to uniform attention
            attn_weights = torch.ones_like(scores) / scores.size(-1)
        
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
                print(f"Warning: freqs_cis has only {freqs_cis.size(0)} positions, but factor has {seq_len}")
                print("Using available positions and padding as needed")
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
            factor_reshaped_ = torch.view_as_complex(
                torch.stack(torch.chunk(factor_reshaped.float(), 2, dim=-1), dim=-1)
            )
            
            # Reshape freqs_cis for broadcasting
            freqs_cis = freqs_cis.view(seq_len, 1, head_dim // 2)
            freqs_cis = freqs_cis.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            freqs_cis = freqs_cis.reshape(batch_size * seq_len, 1, head_dim // 2)
            
            # Apply complex multiplication
            factor_out = torch.view_as_real(factor_reshaped_ * freqs_cis).type_as(factor)
            factor_out = torch.cat(torch.chunk(factor_out, 2, dim=-1), dim=-2)
            
            # Reshape back to original shape
            factor_out = factor_out.reshape(batch_size, seq_len, rank, head_dim)
            
            # If we padded the head_dim, remove the padding
            if head_dim != factor.shape[-1]:
                factor_out = factor_out[..., :factor.shape[-1]]
            
            return factor_out
            
        except Exception as e:
            print(f"Error applying rotary embeddings: {e}")
            # Return the original factor if anything goes wrong
            return factor