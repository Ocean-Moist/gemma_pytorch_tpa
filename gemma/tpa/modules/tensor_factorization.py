"""
Core tensor factorization algorithms for Tensor Product Attention.

This module provides the high-level tensor decomposition functions used 
for converting standard attention layers to TPA (Tensor Product Attention).
"""

import torch
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
    tl.set_backend('pytorch')

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

def shared_factors_tucker_decomposition(weight, num_heads, num_kv_heads, target_ranks, dtype=torch.float16, device="cuda"):
    """
    TensorLLM-style Tucker decomposition with shared factor matrices.
    
    This implements the approach described in the provided documentation,
    using shared factor matrices across attention heads.
    
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
        raise ImportError("TensorLy is required for shared_factors_tucker_decomposition but not available")
    
    tic = time.time()
    print("Using shared factor matrices approach for Tucker decomposition...")
    
    # Prepare target ranks
    hidden_rank = target_ranks.get("hidden_rank", 8)
    head_rank = target_ranks.get("head_rank", 4)
    dim_rank = target_ranks.get("dim_rank", 4)
    
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
    
    # Step 1: Multi-head Tensorisation
    # Reshape the weight matrix into the proper tensor format
    
    # Split weights for query, key, value
    q_weight = weight[:, :q_dim].to(torch.float32)
    k_weight = weight[:, q_dim:q_dim+k_dim].to(torch.float32)
    v_weight = weight[:, q_dim+k_dim:].to(torch.float32)
    
    # Reshape to 3D tensors
    q_weight_3d = q_weight.reshape(hidden_dim, num_heads, head_dim)
    k_weight_3d = k_weight.reshape(hidden_dim, num_kv_heads, head_dim)
    v_weight_3d = v_weight.reshape(hidden_dim, num_kv_heads, head_dim)
    
    # Create combined tensor for Q, K, V (all using same hidden dimension)
    # We'll stack them along a new dimension
    # Create a 4D tensor with dimensions (hidden_dim, max_heads, head_dim, 3)
    # where the last dimension distinguishes between Q, K, V
    
    max_heads = max(num_heads, num_kv_heads)
    combined_tensor = torch.zeros((hidden_dim, max_heads, head_dim, 3), dtype=torch.float32, device=device)
    
    # Fill in the values
    combined_tensor[:, :num_heads, :, 0] = q_weight_3d
    combined_tensor[:, :num_kv_heads, :, 1] = k_weight_3d
    combined_tensor[:, :num_kv_heads, :, 2] = v_weight_3d
    
    # Step 2: Apply Tucker decomposition with shared factor matrices
    print("Applying TensorLy Tucker decomposition with shared factors...")
    
    # Set the ranks for each mode
    ranks = [hidden_rank, head_rank, dim_rank, 3]  # Last mode has 3 components (Q,K,V)
    
    # Apply Tucker decomposition
    core_tensor, factors = tucker(combined_tensor, rank=ranks, init='random')
    
    # Extract the shared factors
    hidden_factor = factors[0]  # For the hidden dimension
    head_factor = factors[1]    # For the head dimension
    dim_factor = factors[2]     # For the embedding dimension
    qkv_factor = factors[3]     # For the QKV dimension
    
    # Step 3: Map Tucker factors to TPA parameters
    result = {}
    
    # Extract cores for Q, K, V by indexing the last dimension
    q_core = tl.tenalg.mode_dot(core_tensor, qkv_factor[0:1], mode=3).squeeze(3)
    k_core = tl.tenalg.mode_dot(core_tensor, qkv_factor[1:2], mode=3).squeeze(3)
    v_core = tl.tenalg.mode_dot(core_tensor, qkv_factor[2:3], mode=3).squeeze(3)
    
    # Store the results
    result["Q_core"] = q_core.to(dtype=dtype, device=device)
    result["Q_hidden_factor"] = hidden_factor.to(dtype=dtype, device=device)
    result["Q_head_factor"] = head_factor[:num_heads].to(dtype=dtype, device=device)
    result["Q_dim_factor"] = dim_factor.to(dtype=dtype, device=device)
    
    result["K_core"] = k_core.to(dtype=dtype, device=device)
    result["K_hidden_factor"] = hidden_factor.to(dtype=dtype, device=device)
    result["K_head_factor"] = head_factor[:num_kv_heads].to(dtype=dtype, device=device)
    result["K_dim_factor"] = dim_factor.to(dtype=dtype, device=device)
    
    result["V_core"] = v_core.to(dtype=dtype, device=device)
    result["V_hidden_factor"] = hidden_factor.to(dtype=dtype, device=device)
    result["V_head_factor"] = head_factor[:num_kv_heads].to(dtype=dtype, device=device)
    result["V_dim_factor"] = dim_factor.to(dtype=dtype, device=device)
    
    print(f"Shared factors Tucker decomposition complete in {time.time() - tic:.2f}s")
    
    # Check reconstruction accuracy for Q, K, V
    q_recon = tl.tenalg.multi_mode_dot(
        result["Q_core"], 
        [result["Q_hidden_factor"], result["Q_head_factor"], result["Q_dim_factor"]], 
        modes=[0, 1, 2]
    )
    
    k_recon = tl.tenalg.multi_mode_dot(
        result["K_core"], 
        [result["K_hidden_factor"], result["K_head_factor"], result["K_dim_factor"]], 
        modes=[0, 1, 2]
    )
    
    v_recon = tl.tenalg.multi_mode_dot(
        result["V_core"], 
        [result["V_hidden_factor"], result["V_head_factor"], result["V_dim_factor"]], 
        modes=[0, 1, 2]
    )
    
    # Calculate reconstruction errors
    q_err = torch.norm(q_recon[:, :num_heads] - q_weight_3d) / torch.norm(q_weight_3d)
    k_err = torch.norm(k_recon[:, :num_kv_heads] - k_weight_3d) / torch.norm(k_weight_3d)
    v_err = torch.norm(v_recon[:, :num_kv_heads] - v_weight_3d) / torch.norm(v_weight_3d)
    
    print(f"Shared factors reconstruction errors - Q: {q_err:.4f}, K: {k_err:.4f}, V: {v_err:.4f}")
    
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