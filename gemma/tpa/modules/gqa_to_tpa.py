"""
GQA to TPA conversion using Tucker decomposition.

This module provides functionality to convert Grouped Query Attention (GQA) weights
to Tensor Product Attention (TPA) format using TensorLLM-style Tucker decomposition.
"""

import torch
import torch.nn.functional as F
import math
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

from .svd_utils import HAS_TENSORLY

# Import TensorLy if available
if HAS_TENSORLY:
    import tensorly as tl
    from tensorly.decomposition import tucker
    tl.set_backend('pytorch')

def gqa_to_tpa_conversion(
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    o_weight: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    q_rank: int = 6,
    k_rank: int = 2,
    v_rank: int = 2,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    Convert GQA attention weights to TPA format using Tucker decomposition.
    
    This function implements the conversion approach described in the document,
    using TensorLLM-style Tucker decomposition while preserving the grouped structure.
    
    Args:
        q_weight: Query projection weight matrix [hidden_dim, num_heads * head_dim]
        k_weight: Key projection weight matrix [hidden_dim, num_kv_heads * head_dim]
        v_weight: Value projection weight matrix [hidden_dim, num_kv_heads * head_dim]
        o_weight: Output projection weight matrix [num_heads * head_dim, hidden_dim]
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads (groups)
        q_rank: Rank for query factorization
        k_rank: Rank for key factorization
        v_rank: Rank for value factorization
        dtype: Data type for the output tensors
        device: Device for computation
        
    Returns:
        Dictionary of factorized weights for TPA implementation
    """
    if not HAS_TENSORLY:
        raise ImportError("TensorLy is required for gqa_to_tpa_conversion")
    
    # Start timing
    tic = time.time()
    print("Starting GQA to TPA conversion using Tucker decomposition...")
    
    # Make sure tensors are on the right device and dtype for processing
    q_weight = q_weight.to(torch.float32)
    k_weight = k_weight.to(torch.float32)
    v_weight = v_weight.to(torch.float32)
    o_weight = o_weight.to(torch.float32).t()  # Transpose to match expected format
    
    # Get dimensions
    hidden_dim = q_weight.shape[0]
    head_dim = q_weight.shape[1] // num_heads
    
    # Check that dimensions are consistent with GQA
    assert head_dim * num_heads == q_weight.shape[1], "Query weight dimensions don't match num_heads"
    assert head_dim * num_kv_heads == k_weight.shape[1], "Key weight dimensions don't match num_kv_heads"
    assert head_dim * num_kv_heads == v_weight.shape[1], "Value weight dimensions don't match num_kv_heads"
    assert hidden_dim == o_weight.shape[0], "Output weight dimensions don't match hidden_dim"
    
    # STEP 1: Multi-head Tensorisation
    # Reshape the weights to better represent the heads
    q_weights_reshaped = q_weight.reshape(hidden_dim, num_heads, head_dim)
    k_weights_reshaped = k_weight.reshape(hidden_dim, num_kv_heads, head_dim)
    v_weights_reshaped = v_weight.reshape(hidden_dim, num_kv_heads, head_dim)
    o_weights_reshaped = o_weight.reshape(hidden_dim, num_heads, head_dim)
    
    # Create mapping from query heads to kv groups
    # For each query head, identify which kv head it should use
    heads_per_group = num_heads // num_kv_heads
    q_to_kv_mapping = torch.tensor([i // heads_per_group for i in range(num_heads)],
                                  device=q_weight.device)
    
    # Create the 4D tensor representation (hidden_dim, head_dim, 4, num_heads)
    # We'll store the query, key, value, and output weights for each head
    all_weights = []
    
    for h in range(num_heads):
        # For each query head, get the corresponding kv head
        kv_head = q_to_kv_mapping[h]
        
        # Stack the weights for this head
        head_tensor = torch.stack([
            q_weights_reshaped[:, h, :],                  # Query weights for this head
            k_weights_reshaped[:, kv_head, :],            # Key weights for the group
            v_weights_reshaped[:, kv_head, :],            # Value weights for the group
            o_weights_reshaped[:, h, :]                   # Output weights for this head
        ], dim=2)  # Stack along dimension 2
        
        all_weights.append(head_tensor)
    
    # Combine all heads into a single 4D tensor
    W_all = torch.stack(all_weights, dim=3)  # Shape: [hidden_dim, head_dim, 4, num_heads]
    
    # Ensure proper dimensions
    print(f"Tensorized weights shape: {W_all.shape}")
    
    # STEP 2: Tucker Decomposition with Shared Factor Matrices
    # Set target ranks
    R1 = q_rank  # Rank for the model dimension
    R2 = q_rank  # Rank for the head dimension
    R3 = q_rank  # Rank for the QKV distinction
    
    # Adjust ranks if needed to prevent issues
    R1 = min(R1, W_all.shape[0])
    R2 = min(R2, W_all.shape[1])
    R3 = min(R3, W_all.shape[2])
    
    print(f"Applying Tucker decomposition with ranks: ({R1}, {R2}, {R3})")
    
    try:
        # Apply Tucker decomposition to the full tensor
        # This enforces shared factor matrices across all heads
        core, factors = tucker(W_all, rank=[R1, R2, R3, num_heads], tol=1e-4)
        
        # Extract the factor matrices
        U1 = factors[0]  # For hidden dimension
        U2 = factors[1]  # For head dimension
        U3 = factors[2]  # For QKV distinction
        # No need for factor 3 since we'll keep each head separate
        
        print(f"Tucker decomposition complete. Core shape: {core.shape}")
        print(f"Factor shapes: {[f.shape for f in factors]}")
    
    except Exception as e:
        print(f"Tucker decomposition failed: {e}")
        print("Falling back to separate decompositions for Q, K, V")
        
        # Fallback: Decompose Q, K, V separately
        q_tensors = W_all[:, :, 0, :]  # Shape: [hidden_dim, head_dim, num_heads]
        k_tensors = W_all[:, :, 1, :]
        v_tensors = W_all[:, :, 2, :]
        o_tensors = W_all[:, :, 3, :]
        
        # Apply tucker to each separately
        q_core, q_factors = tucker(q_tensors, rank=[R1, R2, num_heads])
        k_core, k_factors = tucker(k_tensors, rank=[k_rank, k_rank, num_kv_heads])
        v_core, v_factors = tucker(v_tensors, rank=[v_rank, v_rank, num_kv_heads])
        o_core, o_factors = tucker(o_tensors, rank=[R1, R2, num_heads])
        
        # Create result using the separate decompositions
        result = {}
        result["Q_core"] = q_core.to(dtype=dtype, device=device)
        result["Q_hidden_factor"] = q_factors[0].to(dtype=dtype, device=device)
        result["Q_dim_factor"] = q_factors[1].to(dtype=dtype, device=device)
        result["Q_head_factor"] = torch.eye(num_heads, device=device).to(dtype=dtype)
        
        result["K_core"] = k_core.to(dtype=dtype, device=device)
        result["K_hidden_factor"] = k_factors[0].to(dtype=dtype, device=device)
        result["K_dim_factor"] = k_factors[1].to(dtype=dtype, device=device)
        result["K_head_factor"] = torch.eye(num_kv_heads, device=device).to(dtype=dtype)
        
        result["V_core"] = v_core.to(dtype=dtype, device=device)
        result["V_hidden_factor"] = v_factors[0].to(dtype=dtype, device=device)
        result["V_dim_factor"] = v_factors[1].to(dtype=dtype, device=device)
        result["V_head_factor"] = torch.eye(num_kv_heads, device=device).to(dtype=dtype)
        
        toc = time.time()
        print(f"Fallback conversion complete in {toc - tic:.2f} seconds")
        
        return result
    
    # STEP 3: Map Tucker Factors to TPA Parameters
    result = {}
    
    # Extract factor matrices for query, key, value parts
    # U3 contains factors for Q, K, V, O: shape [4, R3]
    q_factor = U3[0].reshape(-1, 1)  # [R3, 1]
    k_factor = U3[1].reshape(-1, 1)  # [R3, 1]
    v_factor = U3[2].reshape(-1, 1)  # [R3, 1]
    
    # For query:
    # Create the core tensor with QKV factors
    q_core_shape = (R1, R2, num_heads)
    q_core = torch.zeros(q_core_shape, dtype=core.dtype, device=core.device)
    
    # Project the core using q_factor
    for h in range(num_heads):
        q_core[:, :, h] = torch.tensordot(core[:, :, :, h], q_factor, dims=[[2], [0]]).squeeze(-1)
    
    # For keys and values, we need to handle the group structure
    k_core_shape = (R1, R2, num_kv_heads)
    v_core_shape = (R1, R2, num_kv_heads)
    
    k_core = torch.zeros(k_core_shape, dtype=core.dtype, device=core.device)
    v_core = torch.zeros(v_core_shape, dtype=core.dtype, device=core.device)
    
    # Project the core using k_factor and v_factor, preserving the group structure
    for g in range(num_kv_heads):
        # Find all query heads that map to this KV group
        group_q_heads = [h for h in range(num_heads) if q_to_kv_mapping[h] == g]
        
        # Average the cores from all heads in this group
        group_core = torch.zeros_like(core[:, :, :, 0])
        for h in group_q_heads:
            group_core += core[:, :, :, h]
        group_core /= len(group_q_heads)
        
        # Project using k_factor and v_factor
        k_core[:, :, g] = torch.tensordot(group_core, k_factor, dims=[[2], [0]]).squeeze(-1)
        v_core[:, :, g] = torch.tensordot(group_core, v_factor, dims=[[2], [0]]).squeeze(-1)
    
    # Store the results
    # We use tensordot instead of matmul to project from the higher-dimensional space
    
    # For queries:
    result["Q_core"] = q_core.to(dtype=dtype, device=device)
    result["Q_hidden_factor"] = U1.to(dtype=dtype, device=device)
    result["Q_dim_factor"] = U2.to(dtype=dtype, device=device)
    result["Q_head_factor"] = torch.eye(num_heads, device=device).to(dtype=dtype)
    
    # For keys:
    result["K_core"] = k_core.to(dtype=dtype, device=device)
    result["K_hidden_factor"] = U1.to(dtype=dtype, device=device)
    result["K_dim_factor"] = U2.to(dtype=dtype, device=device)
    result["K_head_factor"] = torch.eye(num_kv_heads, device=device).to(dtype=dtype)
    
    # For values:
    result["V_core"] = v_core.to(dtype=dtype, device=device)
    result["V_hidden_factor"] = U1.to(dtype=dtype, device=device)
    result["V_dim_factor"] = U2.to(dtype=dtype, device=device)
    result["V_head_factor"] = torch.eye(num_kv_heads, device=device).to(dtype=dtype)
    
    # Calculate the weight matrices for Wa and Wb in TPA format to verify correctness
    
    # Calculate the weight matrices for Wa and Wb in TPA format
    # For TPA implementation:
    # A_q = W_A_q(x) - shape [batch, seq, heads, q_rank]
    # B_q = W_B_q(x) - shape [batch, seq, q_rank, head_dim]
    
    # Expand dimensions for TPA format
    W_A_q_expanded = torch.zeros((hidden_dim, num_heads * q_rank), device=q_weight.device, dtype=torch.float32)
    W_A_k_expanded = torch.zeros((hidden_dim, num_kv_heads * k_rank), device=k_weight.device, dtype=torch.float32)
    W_A_v_expanded = torch.zeros((hidden_dim, num_kv_heads * v_rank), device=v_weight.device, dtype=torch.float32)
    
    # The proper way to create the expanded Wa is to repeat for each head
    for h in range(num_heads):
        W_A_q = U1 @ q_core[:, :, h]
        W_A_q_expanded[:, h*q_rank:(h+1)*q_rank] = W_A_q
        
    for g in range(num_kv_heads):
        W_A_k = U1 @ k_core[:, :, g]
        W_A_v = U1 @ v_core[:, :, g]
        W_A_k_expanded[:, g*k_rank:(g+1)*k_rank] = W_A_k
        W_A_v_expanded[:, g*v_rank:(g+1)*v_rank] = W_A_v
    
    # Create the B matrices which need reshaping for TPA format
    W_B_q = U2.t()  # [R2, head_dim]
    W_B_k = U2.t()  # [R2, head_dim]
    W_B_v = U2.t()  # [R2, head_dim]
    
    # Reshape B matrices for TPA format (head-interleaved)
    W_B_q_reshaped = W_B_q.repeat(num_heads, 1)  # [num_heads*R2, head_dim]
    W_B_k_reshaped = W_B_k.repeat(num_kv_heads, 1)  # [num_kv_heads*R2, head_dim]
    W_B_v_reshaped = W_B_v.repeat(num_kv_heads, 1)  # [num_kv_heads*R2, head_dim]
    
    # Add expanded matrices to result
    result["W_A_q"] = W_A_q_expanded.to(dtype=dtype, device=device)
    result["W_A_k"] = W_A_k_expanded.to(dtype=dtype, device=device)
    result["W_A_v"] = W_A_v_expanded.to(dtype=dtype, device=device)
    
    result["W_B_q"] = W_B_q_reshaped.to(dtype=dtype, device=device)
    result["W_B_k"] = W_B_k_reshaped.to(dtype=dtype, device=device)
    result["W_B_v"] = W_B_v_reshaped.to(dtype=dtype, device=device)
    
    # Verify reconstruction error
    # This is important to ensure the factorization is accurate
    
    # Reconstruct query weights
    q_recon = torch.zeros_like(q_weights_reshaped)
    for h in range(num_heads):
        q_head_A = W_A_q_expanded[:, h*q_rank:(h+1)*q_rank]
        q_head_B = W_B_q_reshaped[h*q_rank:(h+1)*q_rank, :]
        q_recon[:, h, :] = (q_head_A @ q_head_B).reshape(hidden_dim, head_dim)
    
    # Reconstruct key and value weights with the group structure
    k_recon = torch.zeros_like(k_weights_reshaped)
    v_recon = torch.zeros_like(v_weights_reshaped)
    
    for g in range(num_kv_heads):
        k_group_A = W_A_k_expanded[:, g*k_rank:(g+1)*k_rank]
        k_group_B = W_B_k_reshaped[g*k_rank:(g+1)*k_rank, :]
        k_recon[:, g, :] = (k_group_A @ k_group_B).reshape(hidden_dim, head_dim)
        
        v_group_A = W_A_v_expanded[:, g*v_rank:(g+1)*v_rank]
        v_group_B = W_B_v_reshaped[g*v_rank:(g+1)*v_rank, :]
        v_recon[:, g, :] = (v_group_A @ v_group_B).reshape(hidden_dim, head_dim)
    
    # Reshape for error calculation
    q_recon = q_recon.reshape(hidden_dim, -1)
    k_recon = k_recon.reshape(hidden_dim, -1)
    v_recon = v_recon.reshape(hidden_dim, -1)
    
    # Calculate relative errors
    q_err = torch.norm(q_recon - q_weight) / torch.norm(q_weight)
    k_err = torch.norm(k_recon - k_weight) / torch.norm(k_weight)
    v_err = torch.norm(v_recon - v_weight) / torch.norm(v_weight)
    
    print(f"Reconstruction relative errors - Q: {q_err:.4f}, K: {k_err:.4f}, V: {v_err:.4f}")
    
    toc = time.time()
    print(f"GQA to TPA conversion complete in {toc - tic:.2f} seconds")
    
    return result


def convert_gqa_model_to_tpa(model, q_rank=6, k_rank=2, v_rank=2, dtype=torch.float16, device="cuda"):
    """
    Convert a GQA model to TPA format by applying the conversion to each attention layer.
    
    Args:
        model: The input model with GQA
        q_rank: Rank for query factorization
        k_rank: Rank for key factorization
        v_rank: Rank for value factorization
        dtype: Data type for output tensors
        device: Device for computation
        
    Returns:
        Model with converted weights
    """
    if not HAS_TENSORLY:
        raise ImportError("TensorLy is required for convert_gqa_model_to_tpa")
    
    print("Converting GQA model to TPA format...")
    
    for name, module in model.named_modules():
        # Identify modules that have the query, key, value projections
        if hasattr(module, "q_proj") and hasattr(module, "k_proj") and hasattr(module, "v_proj"):
            print(f"Converting attention layer: {name}")
            
            # Get the number of heads and kv heads (groups)
            if hasattr(module, "num_heads") and hasattr(module, "num_key_value_heads"):
                num_heads = module.num_heads
                num_kv_heads = module.num_key_value_heads
                
                print(f"  Heads: {num_heads}, KV heads: {num_kv_heads}")
                
                # Extract weight matrices
                q_weight = module.q_proj.weight
                k_weight = module.k_proj.weight
                v_weight = module.v_proj.weight
                o_weight = module.o_proj.weight
                
                # Apply GQA to TPA conversion
                factorized_weights = gqa_to_tpa_conversion(
                    q_weight, k_weight, v_weight, o_weight,
                    num_heads, num_kv_heads,
                    q_rank, k_rank, v_rank,
                    dtype, device
                )
                
                # Apply factorized weights to the model
                if hasattr(module, "apply_factorized_weights"):
                    module.apply_factorized_weights(factorized_weights)
                else:
                    print(f"Warning: Module {name} does not support applying factorized weights")
    
    print("GQA to TPA conversion complete")
    return model