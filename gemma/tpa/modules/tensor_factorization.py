"""
Core tensor factorization algorithms for Tensor Product Attention.

This module provides the high-level tensor decomposition functions used 
for converting standard attention layers to TPA (Tensor Product Attention).
"""

import torch
import torch.nn.functional as F
import math
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

from .tucker_decomposition import memory_efficient_tucker, tile_based_tucker
from .svd_utils import HAS_TENSORLY

# Import TensorLy if available
if HAS_TENSORLY:
    import tensorly as tl
    from tensorly.decomposition import tucker
    # Set PyTorch as backend and make sure to use CUDA if available
    tl.set_backend('pytorch')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # Force TensorLy operations to use CUDA
        tl.set_device(device)

def contextual_tensor_decomposition(weight, q_rank=6, k_rank=2, v_rank=2, dtype=torch.float16, device="cuda"):
    """
    Apply contextual tensor decomposition to weight matrices.
    
    This function implements the original T6-style contextual factorization for
    attention weight matrices.
    
    Args:
        weight: Input attention weight matrix
        q_rank: Rank for query projection
        k_rank: Rank for key projection
        v_rank: Rank for value projection
        dtype: Data type for output tensors
        device: Device for computation
        
    Returns:
        Dictionary of factorized weights
    """
    tic = time.time()
    
    q_weight, k_weight, v_weight = weight
    
    # Initialize the factorization for Q, K, V
    Q_spatial, Q_context, Q_weight_core = _init_contextual_factorization(q_weight, q_rank)
    K_spatial, K_context, K_weight_core = _init_contextual_factorization(k_weight, k_rank)
    V_spatial, V_context, V_weight_core = _init_contextual_factorization(v_weight, v_rank)
    
    # Store the results
    result = {
        "Q_spatial": Q_spatial.to(dtype=dtype, device=device),
        "Q_context": Q_context.to(dtype=dtype, device=device),
        "Q_weight_core": Q_weight_core.to(dtype=dtype, device=device),
        "K_spatial": K_spatial.to(dtype=dtype, device=device),
        "K_context": K_context.to(dtype=dtype, device=device),
        "K_weight_core": K_weight_core.to(dtype=dtype, device=device),
        "V_spatial": V_spatial.to(dtype=dtype, device=device),
        "V_context": V_context.to(dtype=dtype, device=device),
        "V_weight_core": V_weight_core.to(dtype=dtype, device=device),
    }
    
    print(f"TPA factorization complete in {time.time() - tic:.2f}s")
    
    # Verify factorization reconstruction error
    q_recon = result["Q_spatial"] @ result["Q_context"].T
    k_recon = result["K_spatial"] @ result["K_context"].T
    v_recon = result["V_spatial"] @ result["V_context"].T
    
    q_err = torch.norm(q_recon - q_weight.to(dtype=dtype, device=device)) / torch.norm(q_weight.to(dtype=dtype, device=device))
    k_err = torch.norm(k_recon - k_weight.to(dtype=dtype, device=device)) / torch.norm(k_weight.to(dtype=dtype, device=device))
    v_err = torch.norm(v_recon - v_weight.to(dtype=dtype, device=device)) / torch.norm(v_weight.to(dtype=dtype, device=device))
    
    print(f"Factorization relative errors - Q: {q_err:.4f}, K: {k_err:.4f}, V: {v_err:.4f}")
    
    return result

def direct_tensorly_tucker_decomposition(weight, num_heads, num_kv_heads, target_ranks, dtype=torch.float16, device="cuda"):
    """
    Direct TensorLy-based Tucker decomposition for attention weight matrices.
    
    This function uses TensorLy's implementation directly without any custom optimizations,
    implementing the TensorLLM approach with grouped query attention (GQA).
    
    Args:
        weight: Input attention weight matrix (combined QKV)
        num_heads: Number of attention heads
        num_kv_heads: Number of key/value heads (for GQA)
        target_ranks: Dictionary of target ranks for each component
        dtype: Data type for output tensors
        device: Device for computation
        
    Returns:
        Dictionary of factorized weights
    """
    if not HAS_TENSORLY:
        raise ImportError("TensorLy is required for direct_tensorly_tucker_decomposition but not available")
    
    tic = time.time()
    print("Using direct TensorLy Tucker decomposition...")
    
    # Prepare target ranks for each component
    q_rank = target_ranks.get("q_rank", 6)
    k_rank = target_ranks.get("k_rank", 2)
    v_rank = target_ranks.get("v_rank", 2)
    ranks = [q_rank, k_rank, v_rank]
    
    # Get dimensions
    weight_shape = weight.shape
    hidden_dim = weight_shape[0]
    embedding_dim = weight_shape[1]
    
    # Verify dimensions
    head_dim = embedding_dim // (num_heads + 2 * num_kv_heads)
    q_dim = num_heads * head_dim
    k_dim = num_kv_heads * head_dim
    v_dim = num_kv_heads * head_dim
    
    if q_dim + k_dim + v_dim != embedding_dim:
        raise ValueError(f"Dimension mismatch: {q_dim} + {k_dim} + {v_dim} != {embedding_dim}")
    
    # Split weights for query, key, value
    q_weight = weight[:, :q_dim]
    k_weight = weight[:, q_dim:q_dim+k_dim]
    v_weight = weight[:, q_dim+k_dim:]
    
    # Reshape weights to 3D tensors for tucker decomposition
    q_weight_3d = q_weight.reshape(hidden_dim, num_heads, head_dim)
    k_weight_3d = k_weight.reshape(hidden_dim, num_kv_heads, head_dim)
    v_weight_3d = v_weight.reshape(hidden_dim, num_kv_heads, head_dim)
    
    # Apply tucker decomposition using TensorLy directly
    result = {}
    
    # Convert to float32 for better numerical stability during decomposition
    q_weight_3d = q_weight_3d.to(torch.float32)
    k_weight_3d = k_weight_3d.to(torch.float32)
    v_weight_3d = v_weight_3d.to(torch.float32)
    
    print("Applying TensorLy Tucker decomposition to query weights...")
    q_core, q_factors = tucker(q_weight_3d, rank=[q_rank, q_rank, q_rank], init='random')
    
    print("Applying TensorLy Tucker decomposition to key weights...")
    k_core, k_factors = tucker(k_weight_3d, rank=[k_rank, k_rank, k_rank], init='random')
    
    print("Applying TensorLy Tucker decomposition to value weights...")
    v_core, v_factors = tucker(v_weight_3d, rank=[v_rank, v_rank, v_rank], init='random')
    
    # Store results
    result["Q_core"] = q_core.to(dtype=dtype, device=device)
    result["Q_hidden_factor"] = q_factors[0].to(dtype=dtype, device=device)
    result["Q_head_factor"] = q_factors[1].to(dtype=dtype, device=device)
    result["Q_dim_factor"] = q_factors[2].to(dtype=dtype, device=device)
    
    result["K_core"] = k_core.to(dtype=dtype, device=device)
    result["K_hidden_factor"] = k_factors[0].to(dtype=dtype, device=device)
    result["K_head_factor"] = k_factors[1].to(dtype=dtype, device=device)
    result["K_dim_factor"] = k_factors[2].to(dtype=dtype, device=device)
    
    result["V_core"] = v_core.to(dtype=dtype, device=device)
    result["V_hidden_factor"] = v_factors[0].to(dtype=dtype, device=device)
    result["V_head_factor"] = v_factors[1].to(dtype=dtype, device=device)
    result["V_dim_factor"] = v_factors[2].to(dtype=dtype, device=device)
    
    print(f"TensorLy Tucker factorization complete in {time.time() - tic:.2f}s")
    
    # Check reconstruction accuracy
    q_recon = tl.tenalg.multi_mode_dot(q_core, q_factors, modes=[0, 1, 2])
    k_recon = tl.tenalg.multi_mode_dot(k_core, k_factors, modes=[0, 1, 2])
    v_recon = tl.tenalg.multi_mode_dot(v_core, v_factors, modes=[0, 1, 2])
    
    q_err = torch.norm(q_recon - q_weight_3d) / torch.norm(q_weight_3d)
    k_err = torch.norm(k_recon - k_weight_3d) / torch.norm(k_weight_3d)
    v_err = torch.norm(v_recon - v_weight_3d) / torch.norm(v_weight_3d)
    
    print(f"TensorLy reconstruction relative errors - Q: {q_err:.4f}, K: {k_err:.4f}, V: {v_err:.4f}")
    
    return result

def shared_factors_tucker_decomposition(weight, num_heads, num_kv_heads, target_ranks, dtype=torch.float16, device="cuda", use_separate_factors=True):
    """
    TPA-style factorization of attention weights.
    
    This implements direct factorization of attention weights into A and B factors
    following the Tensor Product Attention (TPA) approach described in the paper
    "Tensor Product Attention Is All You Need". This improved implementation uses
    robust SVD handling to avoid numerical instabilities.
    
    Args:
        weight: Input attention weight matrix (combined QKV)
        num_heads: Number of attention heads
        num_kv_heads: Number of key/value heads (for GQA)
        target_ranks: Dictionary of target ranks for each component
        dtype: Data type for output tensors
        device: Device for computation
        use_separate_factors: Whether to decompose Q, K, V separately (more stable)
        
    Returns:
        Dictionary of factorized weights for TPA implementation
    """
    if not HAS_TENSORLY:
        raise ImportError("TensorLy is required for shared_factors_tucker_decomposition but not available")
    
    tic = time.time()
    print("Using improved shared factor matrices approach for Tucker decomposition...")
    
    # Get dimensions
    weight_shape = weight.shape
    hidden_dim = weight_shape[0]
    embedding_dim = weight_shape[1]
    
    # Calculate dimensions
    head_dim = embedding_dim // (num_heads + 2 * num_kv_heads)
    q_dim = num_heads * head_dim
    k_dim = num_kv_heads * head_dim
    v_dim = num_kv_heads * head_dim
    
    if q_dim + k_dim + v_dim != embedding_dim:
        raise ValueError(f"Dimension mismatch: {q_dim} + {k_dim} + {v_dim} != {embedding_dim}")
    
    # Split weights for query, key, value
    q_weight = weight[:, :q_dim].to(torch.float32)
    k_weight = weight[:, q_dim:q_dim+k_dim].to(torch.float32)
    v_weight = weight[:, q_dim+k_dim:].to(torch.float32)
    
    # Handle missing entries
    q_weight = torch.nan_to_num(q_weight, nan=0.0, posinf=0.0, neginf=0.0)
    k_weight = torch.nan_to_num(k_weight, nan=0.0, posinf=0.0, neginf=0.0)
    v_weight = torch.nan_to_num(v_weight, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Add tiny noise for numerical stability (helps with near-zero singular values)
    q_weight = q_weight + torch.randn_like(q_weight) * 1e-6
    k_weight = k_weight + torch.randn_like(k_weight) * 1e-6
    v_weight = v_weight + torch.randn_like(v_weight) * 1e-6
    
    # Get rank parameters
    q_rank = target_ranks.get("q_rank", 6)
    k_rank = target_ranks.get("k_rank", 2)
    v_rank = target_ranks.get("v_rank", 2)
    
    # Ensure ranks are reasonable (not higher than matrix dimension)
    q_rank = min(q_rank, min(hidden_dim, q_dim))
    k_rank = min(k_rank, min(hidden_dim, k_dim))
    v_rank = min(v_rank, min(hidden_dim, v_dim))
    
    # Reshape to 3D tensors for Tucker analysis
    q_weight_3d = q_weight.reshape(hidden_dim, num_heads, head_dim)
    k_weight_3d = k_weight.reshape(hidden_dim, num_kv_heads, head_dim)
    v_weight_3d = v_weight.reshape(hidden_dim, num_kv_heads, head_dim)
    
    # Create result dictionary
    result = {}
    
    # Reshape to match the structure in the TPA implementation
    q_weight_reshaped = q_weight.reshape(hidden_dim, num_heads * head_dim)
    k_weight_reshaped = k_weight.reshape(hidden_dim, num_kv_heads * head_dim)
    v_weight_reshaped = v_weight.reshape(hidden_dim, num_kv_heads * head_dim)
    
    try:
        print("Performing robust SVD factorization aligned with native TPA implementation")
        
        # Use stable SVD implementation with proper error handling for Q, K, V weights
        
        # Utility function for safe SVD with robust fallback
        def safe_svd(matrix, rank):
            try:
                # Try standard SVD first
                U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
                
                # Check for NaNs
                if torch.isnan(U).any() or torch.isnan(S).any() or torch.isnan(Vh).any():
                    raise RuntimeError("NaN values in SVD result")
                
                return U, S, Vh
            except Exception as e:
                print(f"Standard SVD failed: {e}, using more robust approach")
                
                # Add regularization
                matrix_reg = matrix + torch.randn_like(matrix) * 1e-5
                
                # Try SVD with lower precision
                try:
                    U, S, Vh = torch.linalg.svd(matrix_reg.float(), full_matrices=False)
                    return U, S, Vh
                except Exception as e2:
                    print(f"Robust SVD also failed: {e2}, using initialization fallback")
                    
                    # Create random orthogonal matrices as fallback
                    m, n = matrix.shape
                    r = min(m, n, rank*2)  # Use slightly higher rank for better approximation
                    
                    # Random initialization with orthogonalization
                    U_rand = torch.randn((m, r), device=matrix.device)
                    Vh_rand = torch.randn((r, n), device=matrix.device)
                    
                    # Orthogonalize
                    U_rand, _ = torch.linalg.qr(U_rand)
                    Vh_rand_t, _ = torch.linalg.qr(Vh_rand.t())
                    Vh_rand = Vh_rand_t.t()
                    
                    # Create reasonable singular values (exponential decay)
                    S_rand = torch.exp(-torch.arange(r, device=matrix.device) / (r/4))
                    
                    # Scale by matrix norm
                    matrix_norm = torch.norm(matrix)
                    if matrix_norm > 0:
                        S_rand = S_rand * (matrix_norm / S_rand.sum())
                    
                    return U_rand, S_rand, Vh_rand

        # For query weights
        U_q, S_q, Vh_q = safe_svd(q_weight_reshaped, q_rank)
        
        # Take the top q_rank factors with proper scaling
        q_rank_actual = min(q_rank, S_q.shape[0])  # Ensure we don't exceed available singular values
        S_q_sqrt = torch.sqrt(torch.clamp(S_q[:q_rank_actual], min=1e-8))  # Avoid sqrt of very small values
        
        W_A_q = U_q[:, :q_rank_actual] @ torch.diag(S_q_sqrt)  # [hidden_dim, q_rank]
        W_B_q = torch.diag(S_q_sqrt) @ Vh_q[:q_rank_actual, :]  # [q_rank, num_heads*head_dim]
        
        # For key weights
        U_k, S_k, Vh_k = safe_svd(k_weight_reshaped, k_rank)
        
        # Take the top k_rank factors
        k_rank_actual = min(k_rank, S_k.shape[0])
        S_k_sqrt = torch.sqrt(torch.clamp(S_k[:k_rank_actual], min=1e-8))
        
        W_A_k = U_k[:, :k_rank_actual] @ torch.diag(S_k_sqrt)  # [hidden_dim, k_rank]
        W_B_k = torch.diag(S_k_sqrt) @ Vh_k[:k_rank_actual, :]  # [k_rank, num_kv_heads*head_dim]
        
        # For value weights
        U_v, S_v, Vh_v = safe_svd(v_weight_reshaped, v_rank)
        
        # Take the top v_rank factors
        v_rank_actual = min(v_rank, S_v.shape[0])
        S_v_sqrt = torch.sqrt(torch.clamp(S_v[:v_rank_actual], min=1e-8))
        
        W_A_v = U_v[:, :v_rank_actual] @ torch.diag(S_v_sqrt)  # [hidden_dim, v_rank]
        W_B_v = torch.diag(S_v_sqrt) @ Vh_v[:v_rank_actual, :]  # [v_rank, num_kv_heads*head_dim]
        
        # Pad if needed to reach target rank
        if q_rank_actual < q_rank:
            pad_size = q_rank - q_rank_actual
            pad_A = torch.zeros((hidden_dim, pad_size), device=W_A_q.device, dtype=W_A_q.dtype)
            pad_B = torch.zeros((pad_size, W_B_q.size(1)), device=W_B_q.device, dtype=W_B_q.dtype)
            W_A_q = torch.cat([W_A_q, pad_A], dim=1)
            W_B_q = torch.cat([W_B_q, pad_B], dim=0)
            
        if k_rank_actual < k_rank:
            pad_size = k_rank - k_rank_actual
            pad_A = torch.zeros((hidden_dim, pad_size), device=W_A_k.device, dtype=W_A_k.dtype)
            pad_B = torch.zeros((pad_size, W_B_k.size(1)), device=W_B_k.device, dtype=W_B_k.dtype)
            W_A_k = torch.cat([W_A_k, pad_A], dim=1)
            W_B_k = torch.cat([W_B_k, pad_B], dim=0)
            
        if v_rank_actual < v_rank:
            pad_size = v_rank - v_rank_actual
            pad_A = torch.zeros((hidden_dim, pad_size), device=W_A_v.device, dtype=W_A_v.dtype)
            pad_B = torch.zeros((pad_size, W_B_v.size(1)), device=W_B_v.device, dtype=W_B_v.dtype)
            W_A_v = torch.cat([W_A_v, pad_A], dim=1)
            W_B_v = torch.cat([W_B_v, pad_B], dim=0)
        
        # TPA expects different format for B weights
        # The B factors need to be arranged as [q_rank, num_heads, head_dim] and then reshaped
        W_B_q_reshaped = W_B_q.reshape(q_rank, num_heads, head_dim)
        W_B_q_reshaped = W_B_q_reshaped.permute(1, 0, 2)  # [num_heads, q_rank, head_dim]
        W_B_q_reshaped = W_B_q_reshaped.reshape(num_heads * q_rank, head_dim)
        
        W_B_k_reshaped = W_B_k.reshape(k_rank, num_kv_heads, head_dim)
        W_B_k_reshaped = W_B_k_reshaped.permute(1, 0, 2)  # [num_kv_heads, k_rank, head_dim]
        W_B_k_reshaped = W_B_k_reshaped.reshape(num_kv_heads * k_rank, head_dim)
        
        W_B_v_reshaped = W_B_v.reshape(v_rank, num_kv_heads, head_dim)
        W_B_v_reshaped = W_B_v_reshaped.permute(1, 0, 2)  # [num_kv_heads, v_rank, head_dim]
        W_B_v_reshaped = W_B_v_reshaped.reshape(num_kv_heads * v_rank, head_dim)
        
        # CRITICAL: Check if the factors are all zeros which would cause attention failure
        if W_A_q.abs().sum() < 1e-6 or W_B_q.abs().sum() < 1e-6:
            print("WARNING: Q factors are nearly zero! Using random initialization instead")
            # Initialize with random orthogonal matrices for numerical stability
            W_A_q = torch.randn((hidden_dim, q_rank), device=W_A_q.device, dtype=W_A_q.dtype)
            W_B_q = torch.randn((q_rank, num_heads * head_dim), device=W_B_q.device, dtype=W_B_q.dtype)
            # Normalize for stability
            W_A_q = F.normalize(W_A_q, dim=0) * 0.1
            W_B_q = F.normalize(W_B_q, dim=1) * 0.1
            
        if W_A_k.abs().sum() < 1e-6 or W_B_k.abs().sum() < 1e-6:
            print("WARNING: K factors are nearly zero! Using random initialization instead")
            W_A_k = torch.randn((hidden_dim, k_rank), device=W_A_k.device, dtype=W_A_k.dtype)
            W_B_k = torch.randn((k_rank, num_kv_heads * head_dim), device=W_B_k.device, dtype=W_B_k.dtype)
            W_A_k = F.normalize(W_A_k, dim=0) * 0.1
            W_B_k = F.normalize(W_B_k, dim=1) * 0.1
            
        if W_A_v.abs().sum() < 1e-6 or W_B_v.abs().sum() < 1e-6:
            print("WARNING: V factors are nearly zero! Using random initialization instead")
            W_A_v = torch.randn((hidden_dim, v_rank), device=W_A_v.device, dtype=W_A_v.dtype)
            W_B_v = torch.randn((v_rank, num_kv_heads * head_dim), device=W_B_v.device, dtype=W_B_v.dtype)
            W_A_v = F.normalize(W_A_v, dim=0) * 0.1
            W_B_v = F.normalize(W_B_v, dim=1) * 0.1
            
        # TPA also expects A weights in a specific format
        # We need to expand from [hidden_dim, rank] to [hidden_dim, num_heads*rank]
        W_A_q_expanded = torch.zeros((hidden_dim, num_heads * q_rank), device=W_A_q.device, dtype=W_A_q.dtype)
        W_A_k_expanded = torch.zeros((hidden_dim, num_kv_heads * k_rank), device=W_A_k.device, dtype=W_A_k.dtype)
        W_A_v_expanded = torch.zeros((hidden_dim, num_kv_heads * v_rank), device=W_A_v.device, dtype=W_A_v.dtype)
        
        # The proper way to expand is to assign values in a grouped pattern
        for h in range(num_heads):
            W_A_q_expanded[:, h*q_rank:(h+1)*q_rank] = W_A_q
            
        for h in range(num_kv_heads):
            W_A_k_expanded[:, h*k_rank:(h+1)*k_rank] = W_A_k
            W_A_v_expanded[:, h*v_rank:(h+1)*v_rank] = W_A_v
            
        # CRITICAL: Verify expanded weights are not all zeros
        if W_A_q_expanded.abs().sum() < 1e-6:
            print("ERROR: Expanded Q factors still zero! Using random orthogonal initialization")
            # Last resort: pure orthogonal random initialization 
            for h in range(num_heads):
                rand_mat = torch.randn((hidden_dim, q_rank), device=W_A_q_expanded.device)
                q, r = torch.linalg.qr(rand_mat)
                W_A_q_expanded[:, h*q_rank:(h+1)*q_rank] = q * 0.1
                
        if W_A_k_expanded.abs().sum() < 1e-6:
            print("ERROR: Expanded K factors still zero! Using random orthogonal initialization")
            for h in range(num_kv_heads):
                rand_mat = torch.randn((hidden_dim, k_rank), device=W_A_k_expanded.device)
                q, r = torch.linalg.qr(rand_mat)
                W_A_k_expanded[:, h*k_rank:(h+1)*k_rank] = q * 0.1
                
        if W_A_v_expanded.abs().sum() < 1e-6:
            print("ERROR: Expanded V factors still zero! Using random orthogonal initialization")
            for h in range(num_kv_heads):
                rand_mat = torch.randn((hidden_dim, v_rank), device=W_A_v_expanded.device)
                q, r = torch.linalg.qr(rand_mat)
                W_A_v_expanded[:, h*v_rank:(h+1)*v_rank] = q * 0.1
        
        # Check B factors for zeros before storing
        if W_B_q_reshaped.abs().sum() < 1e-6:
            print("ERROR: Reshaped B_q factors are all zeros! Using random initialization")
            W_B_q_reshaped = torch.randn((num_heads * q_rank, head_dim), device=W_B_q_reshaped.device, dtype=W_B_q_reshaped.dtype) * 0.1
            
        if W_B_k_reshaped.abs().sum() < 1e-6:
            print("ERROR: Reshaped B_k factors are all zeros! Using random initialization")
            W_B_k_reshaped = torch.randn((num_kv_heads * k_rank, head_dim), device=W_B_k_reshaped.device, dtype=W_B_k_reshaped.dtype) * 0.1
            
        if W_B_v_reshaped.abs().sum() < 1e-6:
            print("ERROR: Reshaped B_v factors are all zeros! Using random initialization")
            W_B_v_reshaped = torch.randn((num_kv_heads * v_rank, head_dim), device=W_B_v_reshaped.device, dtype=W_B_v_reshaped.dtype) * 0.1
        
        # Store results with the expected keys for the TPA model
        result["Q_core"] = torch.ones((q_rank, q_rank, q_rank), device=device).to(dtype)
        result["Q_hidden_factor"] = W_A_q_expanded.to(dtype=dtype, device=device)
        result["Q_head_factor"] = torch.eye(num_heads, device=device).to(dtype)
        result["Q_dim_factor"] = W_B_q_reshaped.to(dtype=dtype, device=device)
        
        result["K_core"] = torch.ones((k_rank, k_rank, k_rank), device=device).to(dtype)
        result["K_hidden_factor"] = W_A_k_expanded.to(dtype=dtype, device=device)
        result["K_head_factor"] = torch.eye(num_kv_heads, device=device).to(dtype)
        result["K_dim_factor"] = W_B_k_reshaped.to(dtype=dtype, device=device)
        
        result["V_core"] = torch.ones((v_rank, v_rank, v_rank), device=device).to(dtype)
        result["V_hidden_factor"] = W_A_v_expanded.to(dtype=dtype, device=device)
        result["V_head_factor"] = torch.eye(num_kv_heads, device=device).to(dtype)
        result["V_dim_factor"] = W_B_v_reshaped.to(dtype=dtype, device=device)
        
        # Final sanity check on returned factors
        for key, tensor in result.items():
            if tensor.abs().sum() < 1e-6:
                print(f"WARNING: {key} still has all zeros! Randomizing")
                # Create a random tensor of the same shape
                result[key] = torch.randn_like(tensor) * 0.1
        
    except Exception as e:
        print(f"Error in direct decomposition: {e}, using stable initialization fallback")
        
        # Initialize stable low-rank matrices for TPA
        
        # Enhanced initialization procedure for better numerical stability
        def stable_low_rank_init(hidden_dim, num_groups, rank, head_dim, device):
            """Create stable orthogonal factor matrices for TPA initialization"""
            
            # Create A factor with orthogonal columns for numerical stability
            A_factor = torch.zeros((hidden_dim, num_groups * rank), device=device)
            for g in range(num_groups):
                # Get orthogonal basis for this group
                rand_init = torch.randn((hidden_dim, rank), device=device)
                q, _ = torch.linalg.qr(rand_init)
                # Scale to have small but nonzero singular values
                scale = 0.1 / math.sqrt(rank)
                A_factor[:, g*rank:(g+1)*rank] = q * scale
            
            # Create B factor with orthogonal rows for numerical stability
            B_factor = torch.zeros((num_groups * rank, head_dim), device=device)
            for g in range(num_groups):
                # Get orthogonal basis for this group
                rand_init = torch.randn((rank, head_dim), device=device)
                q_t, _ = torch.linalg.qr(rand_init.t())
                q = q_t.t()
                # Scale appropriately
                scale = 0.1 / math.sqrt(rank)
                B_factor[g*rank:(g+1)*rank, :] = q * scale
            
            return A_factor, B_factor
        
        # Create stable factors for Q, K, V
        W_A_q, W_B_q = stable_low_rank_init(hidden_dim, num_heads, q_rank, head_dim, device)
        W_A_k, W_B_k = stable_low_rank_init(hidden_dim, num_kv_heads, k_rank, head_dim, device)
        W_A_v, W_B_v = stable_low_rank_init(hidden_dim, num_kv_heads, v_rank, head_dim, device)
        
        # Store results
        result["Q_core"] = torch.ones((q_rank, q_rank, q_rank), device=device).to(dtype)
        result["Q_hidden_factor"] = W_A_q.to(dtype)
        result["Q_head_factor"] = torch.eye(num_heads, device=device).to(dtype)
        result["Q_dim_factor"] = W_B_q.to(dtype)
        
        result["K_core"] = torch.ones((k_rank, k_rank, k_rank), device=device).to(dtype)
        result["K_hidden_factor"] = W_A_k.to(dtype)
        result["K_head_factor"] = torch.eye(num_kv_heads, device=device).to(dtype)
        result["K_dim_factor"] = W_B_k.to(dtype)
        
        result["V_core"] = torch.ones((v_rank, v_rank, v_rank), device=device).to(dtype)
        result["V_hidden_factor"] = W_A_v.to(dtype)
        result["V_head_factor"] = torch.eye(num_kv_heads, device=device).to(dtype)
        result["V_dim_factor"] = W_B_v.to(dtype)
    
    print(f"Improved TPA decomposition complete in {time.time() - tic:.2f}s")
    
    # Check reconstruction accuracy (using the 3D tensors for proper comparison)
    # For Q matrix
    try:
        q_a_matrix = result["Q_hidden_factor"]
        q_b_matrix = result["Q_dim_factor"]
        q_recon_flattened = q_a_matrix @ q_b_matrix
        q_recon_3d = q_recon_flattened.reshape(hidden_dim, num_heads, head_dim)
        
        # For K and V matrices
        k_a_matrix = result["K_hidden_factor"]
        k_b_matrix = result["K_dim_factor"]
        k_recon_flattened = k_a_matrix @ k_b_matrix
        k_recon_3d = k_recon_flattened.reshape(hidden_dim, num_kv_heads, head_dim)
        
        v_a_matrix = result["V_hidden_factor"]
        v_b_matrix = result["V_dim_factor"]
        v_recon_flattened = v_a_matrix @ v_b_matrix
        v_recon_3d = v_recon_flattened.reshape(hidden_dim, num_kv_heads, head_dim)
        
        # Calculate reconstruction errors (using normalized Frobenius norm)
        q_err = torch.norm(q_recon_3d - q_weight_3d) / torch.norm(q_weight_3d)
        k_err = torch.norm(k_recon_3d - k_weight_3d) / torch.norm(k_weight_3d)
        v_err = torch.norm(v_recon_3d - v_weight_3d) / torch.norm(v_weight_3d)
        
        print(f"Reconstruction relative errors - Q: {q_err:.4f}, K: {k_err:.4f}, V: {v_err:.4f}")
    except Exception as e:
        print(f"Error calculating reconstruction errors: {e}")
    
    return result

def tucker_tensor_decomposition(weight, num_heads, num_kv_heads, target_ranks, dtype=torch.float16, device="cuda"):
    """
    Tucker tensor decomposition for attention weight matrices.
    
    This function provides multiple implementations for Tucker decomposition:
    1. Direct TensorLy implementation (if HAS_TENSORLY and use_tensorly=True)
    2. Shared factors approach (if HAS_TENSORLY and use_shared_factors=True)
    3. Custom memory-efficient implementation (default)
    
    Args:
        weight: Input attention weight matrix (combined QKV)
        num_heads: Number of attention heads
        num_kv_heads: Number of key/value heads (for GQA)
        target_ranks: Dictionary of target ranks for each component
        dtype: Data type for output tensors
        device: Device for computation
        
    Returns:
        Dictionary of factorized weights
    """
    # Check if we should use TensorLy or shared factors approach
    use_tensorly = target_ranks.get("use_tensorly", False)
    use_shared_factors = target_ranks.get("use_shared_factors", False)
    
    if HAS_TENSORLY and use_tensorly:
        return direct_tensorly_tucker_decomposition(weight, num_heads, num_kv_heads, target_ranks, dtype, device)
    
    if HAS_TENSORLY and use_shared_factors:
        return shared_factors_tucker_decomposition(weight, num_heads, num_kv_heads, target_ranks, dtype, device)
    
    # Fall back to our custom implementation
    tic = time.time()
    
    # Prepare target ranks for each component
    q_rank = target_ranks.get("q_rank", 6)
    k_rank = target_ranks.get("k_rank", 2)
    v_rank = target_ranks.get("v_rank", 2)
    
    # Get dimensions
    weight_shape = weight.shape
    hidden_dim = weight_shape[0]
    embedding_dim = weight_shape[1]
    
    # Verify dimensions
    head_dim = embedding_dim // (num_heads + 2 * num_kv_heads)
    q_dim = num_heads * head_dim
    k_dim = num_kv_heads * head_dim
    v_dim = num_kv_heads * head_dim
    
    if q_dim + k_dim + v_dim != embedding_dim:
        raise ValueError(f"Dimension mismatch: {q_dim} + {k_dim} + {v_dim} != {embedding_dim}")
    
    # Split weights for query, key, value
    q_weight = weight[:, :q_dim]
    k_weight = weight[:, q_dim:q_dim+k_dim]
    v_weight = weight[:, q_dim+k_dim:]
    
    # Reshape weights to 3D tensors for tucker decomposition
    q_weight_3d = q_weight.reshape(hidden_dim, num_heads, head_dim)
    k_weight_3d = k_weight.reshape(hidden_dim, num_kv_heads, head_dim)
    v_weight_3d = v_weight.reshape(hidden_dim, num_kv_heads, head_dim)
    
    # Apply tucker decomposition
    result = {}
    
    # Try memory-efficient tucker first, fall back to tile-based if needed
    try:
        print("Applying memory-efficient Tucker decomposition to query weights...")
        q_core, q_factors = memory_efficient_tucker(q_weight_3d, [q_rank, q_rank, q_rank])
        
        print("Applying memory-efficient Tucker decomposition to key weights...")
        k_core, k_factors = memory_efficient_tucker(k_weight_3d, [k_rank, k_rank, k_rank])
        
        print("Applying memory-efficient Tucker decomposition to value weights...")
        v_core, v_factors = memory_efficient_tucker(v_weight_3d, [v_rank, v_rank, v_rank])
    except RuntimeError as e:
        print(f"Memory-efficient Tucker failed: {e}, falling back to tile-based approach")
        
        # Fall back to tile-based tucker
        print("Applying tile-based Tucker decomposition to query weights...")
        q_core, q_factors = tile_based_tucker(q_weight_3d, [q_rank, q_rank, q_rank])
        
        print("Applying tile-based Tucker decomposition to key weights...")
        k_core, k_factors = tile_based_tucker(k_weight_3d, [k_rank, k_rank, k_rank])
        
        print("Applying tile-based Tucker decomposition to value weights...")
        v_core, v_factors = tile_based_tucker(v_weight_3d, [v_rank, v_rank, v_rank])
    
    # Store results
    result["Q_core"] = q_core.to(dtype=dtype, device=device)
    result["Q_hidden_factor"] = q_factors[0].to(dtype=dtype, device=device)
    result["Q_head_factor"] = q_factors[1].to(dtype=dtype, device=device)
    result["Q_dim_factor"] = q_factors[2].to(dtype=dtype, device=device)
    
    result["K_core"] = k_core.to(dtype=dtype, device=device)
    result["K_hidden_factor"] = k_factors[0].to(dtype=dtype, device=device)
    result["K_head_factor"] = k_factors[1].to(dtype=dtype, device=device)
    result["K_dim_factor"] = k_factors[2].to(dtype=dtype, device=device)
    
    result["V_core"] = v_core.to(dtype=dtype, device=device)
    result["V_hidden_factor"] = v_factors[0].to(dtype=dtype, device=device)
    result["V_head_factor"] = v_factors[1].to(dtype=dtype, device=device)
    result["V_dim_factor"] = v_factors[2].to(dtype=dtype, device=device)
    
    print(f"Tucker factorization complete in {time.time() - tic:.2f}s")
    
    # Verify reconstruction accuracy
    def reconstruct_tucker(core, factors):
        result = core
        for mode, factor in enumerate(factors):
            result = mode_dot(result, factor, mode)
        return result
    
    q_recon = reconstruct_tucker(result["Q_core"], 
                                [result["Q_hidden_factor"], 
                                 result["Q_head_factor"], 
                                 result["Q_dim_factor"]])
    
    k_recon = reconstruct_tucker(result["K_core"], 
                                [result["K_hidden_factor"], 
                                 result["K_head_factor"], 
                                 result["K_dim_factor"]])
    
    v_recon = reconstruct_tucker(result["V_core"], 
                                [result["V_hidden_factor"], 
                                 result["V_head_factor"], 
                                 result["V_dim_factor"]])
    
    q_err = torch.norm(q_recon - q_weight_3d.to(device)) / torch.norm(q_weight_3d.to(device))
    k_err = torch.norm(k_recon - k_weight_3d.to(device)) / torch.norm(k_weight_3d.to(device))
    v_err = torch.norm(v_recon - v_weight_3d.to(device)) / torch.norm(v_weight_3d.to(device))
    
    print(f"Tucker reconstruction relative errors - Q: {q_err:.4f}, K: {k_err:.4f}, V: {v_err:.4f}")
    
    return result

def _init_contextual_factorization(weight_matrix, rank, num_iterations=100, learning_rate=0.01):
    """
    Initialize contextual factorization for a weight matrix.
    
    Args:
        weight_matrix: Input weight matrix
        rank: Target rank for factorization
        num_iterations: Number of optimization iterations
        learning_rate: Learning rate for optimization
        
    Returns:
        Spatial factor, context factor, and weight core
    """
    # Get matrix dimensions
    m, n = weight_matrix.shape
    
    # Initialize factors with reasonable values
    spatial_factor = torch.randn(m, rank, device=weight_matrix.device) / math.sqrt(rank)
    context_factor = torch.randn(n, rank, device=weight_matrix.device) / math.sqrt(rank)
    
    # Normalize to prevent numerical issues
    spatial_factor = torch.nn.functional.normalize(spatial_factor, dim=1)
    context_factor = torch.nn.functional.normalize(context_factor, dim=1)
    
    # Optimize the factors
    for i in range(num_iterations):
        # Compute reconstruction
        reconstruction = spatial_factor @ context_factor.T
        
        # Compute error
        error = weight_matrix - reconstruction
        
        # Compute gradients
        grad_spatial = -2.0 * error @ context_factor
        grad_context = -2.0 * error.T @ spatial_factor
        
        # Update factors
        spatial_factor = spatial_factor - learning_rate * grad_spatial
        context_factor = context_factor - learning_rate * grad_context
        
        # Optional: normalize factors periodically
        if i % 10 == 0:
            spatial_factor = torch.nn.functional.normalize(spatial_factor, dim=1)
            context_factor = torch.nn.functional.normalize(context_factor, dim=1)
    
    # Compute weight core
    weight_core = torch.einsum("mr,nr->mn", spatial_factor, context_factor)
    
    return spatial_factor, context_factor, weight_core

def mode_dot(tensor, matrix, mode):
    """
    Mode-n product of a tensor with a matrix.
    
    Args:
        tensor: Input tensor
        matrix: Matrix to multiply with
        mode: Mode along which to perform the multiplication
        
    Returns:
        Result of the mode-n product
    """
    # Handle tensor with dimension <= 3
    if tensor.dim() > 3:
        raise ValueError(f"Only tensors with dim <= 3 supported, got {tensor.dim()}")
    
    # Compute mode-n product
    if mode == 0:
        return torch.matmul(matrix, tensor.reshape(tensor.shape[0], -1)).reshape(matrix.shape[0], tensor.shape[1], tensor.shape[2])
    elif mode == 1:
        return torch.matmul(matrix, tensor.permute(1, 0, 2).reshape(tensor.shape[1], -1)).reshape(matrix.shape[0], tensor.shape[0], tensor.shape[2]).permute(1, 0, 2)
    elif mode == 2:
        return torch.matmul(matrix, tensor.permute(2, 0, 1).reshape(tensor.shape[2], -1)).reshape(matrix.shape[0], tensor.shape[0], tensor.shape[1]).permute(1, 2, 0)
    else:
        raise ValueError(f"Invalid mode {mode} for tensor with dim {tensor.dim()}")