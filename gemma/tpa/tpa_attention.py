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
        
        # Get the total context length we'll be dealing with
        ctx_len = kv_write_indices[-1].item() + 1 if kv_write_indices is not None else seq_len
        
        # Write new values to KV cache
        if kv_write_indices is not None:
            # Ensure dtype match with the cache tensors
            A_k_dtype = A_k.to(dtype=k_cache_A.dtype)
            B_k_rotated_dtype = B_k_rotated.to(dtype=k_cache_B.dtype)
            A_v_dtype = A_v.to(dtype=v_cache_A.dtype)
            B_v_dtype = B_v.to(dtype=v_cache_B.dtype)
            
            k_cache_A.index_copy_(1, kv_write_indices, A_k_dtype)
            k_cache_B.index_copy_(1, kv_write_indices, B_k_rotated_dtype)
            v_cache_A.index_copy_(1, kv_write_indices, A_v_dtype)
            v_cache_B.index_copy_(1, kv_write_indices, B_v_dtype)
        
        # Compute query from factorized form
        # Ensure matching dtypes for matmul
        A_q_float = A_q.to(dtype=torch.float32)
        B_q_rotated_float = B_q_rotated.to(dtype=torch.float32)
        
        Q = torch.matmul(
            A_q_float.view(batch_size * seq_len, self.num_heads, self.q_rank),
            B_q_rotated_float.view(batch_size * seq_len, self.q_rank, self.head_dim)
        ).view(batch_size, seq_len, self.num_heads, self.head_dim).div(self.q_rank)
        
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
        
        # Build full K matrix by multiplying factors
        # [batch_size, ctx_len, num_kv_heads, head_dim]
        K = torch.matmul(
            K_A.view(batch_size * ctx_len, self.num_kv_heads, self.k_rank),
            K_B.view(batch_size * ctx_len, self.k_rank, self.head_dim)
        ).view(batch_size, ctx_len, self.num_kv_heads, self.head_dim).div(self.k_rank)
        
        # Convert back to original dtype
        K = K.to(dtype=hidden_states.dtype)
        
        # Expand K if we're using grouped query attention (num_kv_heads < num_heads)
        if self.num_kv_heads != self.num_heads:
            K = torch.repeat_interleave(K, self.num_queries_per_kv, dim=2)
        
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
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(Q)
        
        # Compute attention output using factorized V
        # Get cached V factors
        V_A = v_cache_A[:, :ctx_len]
        V_B = v_cache_B[:, :ctx_len]
        
        # Ensure matching dtypes for matmul
        V_A = V_A.to(dtype=torch.float32)
        V_B = V_B.to(dtype=torch.float32)
        
        # Build full V matrix by multiplying factors
        # [batch_size, ctx_len, num_kv_heads, head_dim]
        V = torch.matmul(
            V_A.view(batch_size * ctx_len, self.num_kv_heads, self.v_rank),
            V_B.view(batch_size * ctx_len, self.v_rank, self.head_dim)
        ).view(batch_size, ctx_len, self.num_kv_heads, self.head_dim).div(self.v_rank)
        
        # Convert back to original dtype
        V = V.to(dtype=hidden_states.dtype)
        
        # Expand V if we're using grouped query attention
        if self.num_kv_heads != self.num_heads:
            V = torch.repeat_interleave(V, self.num_queries_per_kv, dim=2)
        
        # [batch_size, num_heads, ctx_len, head_dim]
        V = V.transpose(1, 2)
        
        # Apply attention weights to values
        # [batch_size, num_heads, seq_len, head_dim]
        output = torch.matmul(attn_weights, V)
        
        # Reshape output and apply output projection
        # [batch_size, seq_len, hidden_size]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(output)
        
        return output
    
    def _apply_rotary_emb_to_factor(self, factor: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """Apply rotary positional embedding to B factor matrices."""
        # factor shape: [batch_size, seq_len, rank, head_dim]
        # freqs_cis shape: [seq_len, head_dim//2]
        
        batch_size, seq_len, rank, head_dim = factor.shape
        
        # Handle empty sequence case (seq_len=0)
        if seq_len == 0 or batch_size == 0:
            return factor
        
        # Ensure freqs_cis has enough positions
        if freqs_cis.size(0) < seq_len:
            raise ValueError(f"freqs_cis has only {freqs_cis.size(0)} positions, but factor has {seq_len}")
            
        # Take only needed positions from freqs_cis
        freqs_cis = freqs_cis[:seq_len]
        
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
        
        return factor_out