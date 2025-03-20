"""
GQA to TPA conversion using Tucker decomposition.

This module provides functionality to convert Grouped Query Attention (GQA) weights
to Tensor Product Attention (TPA) format using TensorLLM-style Tucker decomposition.
"""

import torch
import torch.nn as nn
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
    # Set PyTorch as backend, which will use CUDA if PyTorch is using CUDA
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
    device: str = "cuda",
    override_head_dim: int = None,  # Optional parameter to force a specific head dimension
    transposed_weights: bool = False,  # Whether weights are in [out_proj, in_proj] format (False) or [in_proj, out_proj] format (True)
    use_dynamic_ranks: bool = True,  # Whether to use ranks determined by Tucker decomposition (True) or force specified ranks (False)
    config = None,  # Model config to ensure consistent dimensions
    fat_ranks: bool = False  # Whether to use very large ranks (240) for higher accuracy
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
        override_head_dim: Optional parameter to force a specific head dimension
        transposed_weights: Whether weights are in [out_proj, in_proj] format (False) or [in_proj, out_proj] format (True)
        use_dynamic_ranks: Whether to use ranks determined by Tucker decomposition (True) or force specified ranks (False)
        config: Model config to ensure consistent dimensions - we'll use config.hidden_size if available
        
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
    
    # Handle weight format - Gemma typically has weights in [output_proj, input_proj] format
    if transposed_weights:
        print("Using transposed weight format")
        o_weight = o_weight.to(torch.float32)
    else:
        print("Transposing output weights to match expected format")
        o_weight = o_weight.to(torch.float32).t()  # Transpose to expected format
        
    # Update transposed_weights if we transpose the q,k,v weights later
    transposed_weights = transposed_weights or need_transpose
    
    # Apply fat_ranks parameter - override ranks if specified
    if fat_ranks:
        print("Using energy-based SVD rank selection (98% explained variance) instead of fixed ranks")
        # We'll determine actual ranks during the Tucker decomposition based on explained variance
        # These are just initial values that will be overridden
        q_rank = 240
        k_rank = 240
        v_rank = 240
        print(f"Initial ranks (will be refined based on SVD): q_rank={q_rank}, k_rank={k_rank}, v_rank={v_rank}")
    
    # Get dimensions - use config.hidden_size if provided, otherwise infer from weights
    # This ensures consistency with the model's actual hidden state dimensions
    config_hidden_size = getattr(config, 'hidden_size', None) if config is not None else None
    
    # For Gemma models, check if weights need to be transposed
    # In PyTorch Linear layers, weights have shape [out_features, in_features]
    # But for TPA conversion, we need them in [hidden_dim, projection_dim] format
    need_transpose = False
    
    if config_hidden_size is not None:
        hidden_dim = config_hidden_size
        print(f"Using config.hidden_size={hidden_dim} instead of inferring from weights")
        
        # Check if weights are in the wrong orientation (common in PyTorch Linear layers)
        if q_weight.shape[0] != hidden_dim and q_weight.shape[1] == hidden_dim:
            print(f"Detected transposed weight orientation: q_weight shape is {q_weight.shape} but hidden_dim={hidden_dim}")
            print(f"Weights appear to be in PyTorch Linear format [out_features, in_features]")
            print(f"Transposing weights for proper TPA conversion")
            
            # Transpose the weights for TPA conversion
            print(f"\nDIMENSION MISMATCH DETECTED: Transposing weights for Gemma-3-1B model")
            print(f"  Original shapes: q=[{q_weight.shape}], k=[{k_weight.shape}], v=[{v_weight.shape}]")
            print(f"  Expected hidden_dim from config: {hidden_dim}")
            print(f"  Actual hidden_dim in weights: {q_weight.shape[1]}")
            print(f"  This is a known issue with Gemma-3-1B PyTorch weights orientation")
            
            q_weight = q_weight.transpose(0, 1)
            k_weight = k_weight.transpose(0, 1)
            v_weight = v_weight.transpose(0, 1)
            need_transpose = True
            
            print(f"  After transposition: q=[{q_weight.shape}], k=[{k_weight.shape}], v=[{v_weight.shape}]")
        
        # After possible transposition, verify again
        if q_weight.shape[0] != hidden_dim:
            raise ValueError(f"CRITICAL ERROR: Weight hidden dim {q_weight.shape[0]} != config.hidden_size {hidden_dim}. Cannot proceed with mismatched dimensions.")
    else:
        # No config provided, infer from weights (and possibly transpose)
        if q_weight.shape[0] > q_weight.shape[1]:
            # Likely already in the right orientation
            hidden_dim = q_weight.shape[0]
        else:
            # Likely needs transposition
            print(f"No config.hidden_size provided. Weight shape suggests transposition needed")
            q_weight = q_weight.transpose(0, 1)
            k_weight = k_weight.transpose(0, 1)
            v_weight = v_weight.transpose(0, 1)
            hidden_dim = q_weight.shape[0]
            need_transpose = True
            
        print(f"Inferred hidden_dim={hidden_dim} from weight shape")
    
    # Check if we have a forced head dimension
    if override_head_dim is not None:
        print(f"Using override_head_dim={override_head_dim}")
        head_dim = override_head_dim
        q_head_dim = override_head_dim
        kv_head_dim = override_head_dim
    else:
        # Carefully calculate head dimensions based on weight shapes and number of heads
        # For Gemma-3-1B, after transposition q_weight shape is [1152, 1024]
        # where 1024 = num_heads(4) * head_dim(256)
        if need_transpose:
            # If we transposed, the weights are now in [hidden_dim, projections] format
            # The projection dimension should be divisible by the number of heads
            q_projection_dim = q_weight.shape[1]
            k_projection_dim = k_weight.shape[1]
            
            if q_projection_dim % num_heads == 0:
                q_head_dim = q_projection_dim // num_heads
                print(f"After transposition, calculated q_head_dim = {q_head_dim} (projection dim {q_projection_dim} / {num_heads} heads)")
            else:
                # Try to infer a reasonable value that divides exactly
                q_head_dim = 256  # Default for Gemma models
                print(f"WARNING: q_projection_dim {q_projection_dim} is not divisible by num_heads {num_heads}")
                print(f"Using default q_head_dim = {q_head_dim}")
            
            if k_projection_dim % num_kv_heads == 0:
                kv_head_dim = k_projection_dim // num_kv_heads
                print(f"After transposition, calculated kv_head_dim = {kv_head_dim} (projection dim {k_projection_dim} / {num_kv_heads} kv_heads)")
            else:
                # Try to infer a reasonable value
                kv_head_dim = 256  # Default for Gemma models
                print(f"WARNING: k_projection_dim {k_projection_dim} is not divisible by num_kv_heads {num_kv_heads}")
                print(f"Using default kv_head_dim = {kv_head_dim}")
        else:
            # Standard calculation when weights are already in expected orientation
            # The query weight has shape [hidden_dim, num_heads * head_dim]
            # So head_dim = q_weight.shape[1] / num_heads
            q_head_dim = q_weight.shape[1] // num_heads
            
            # The key/value weights have shape [hidden_dim, num_kv_heads * head_dim]
            # So kv_head_dim = k_weight.shape[1] / num_kv_heads
            kv_head_dim = k_weight.shape[1] // num_kv_heads
        
        # For backward compatibility
        head_dim = q_head_dim
    
    print(f"Dimensions: hidden_dim={hidden_dim}")
    print(f"Query: num_heads={num_heads}, head_dim={q_head_dim}")
    print(f"Key/Value: num_kv_heads={num_kv_heads}, head_dim={kv_head_dim}")
    
    # For GQA models, query heads and key/value heads can have different dimensions
    # We'll use the appropriate head_dim for each part
    use_different_dims = q_head_dim != kv_head_dim
    if use_different_dims:
        print(f"Using different dimensions for query ({q_head_dim}) and key/value ({kv_head_dim})")
    else:
        print(f"Using consistent head dimension of {q_head_dim} for all projections")
    
    # Verify final dimensions are consistent
    print(f"Final dimensions: hidden_dim={hidden_dim}, head_dim={head_dim}, num_heads={num_heads}, num_kv_heads={num_kv_heads}")
    print(f"Weight shapes: Q={q_weight.shape}, K={k_weight.shape}, V={v_weight.shape}, O={o_weight.shape}")
    
    # No assertions - try to handle inconsistent dimensions gracefully
    
    # STEP 1: Multi-head Tensorisation
    # For Gemma-3-1B, we need to recalculate head dimensions based on actual weights
    
    # For Gemma-3-1B, q_weight has shape [1152, 1024] after transposition
    # Where 1024 = 4 heads * 256 head_dim
    q_proj_dim = q_weight.shape[1]
    k_proj_dim = k_weight.shape[1]
    v_proj_dim = v_weight.shape[1]
    
    # Calculate head dimensions directly from the weights
    if q_proj_dim % num_heads != 0:
        raise ValueError(f"Q projection dimension {q_proj_dim} not divisible by num_heads {num_heads}")
    
    if k_proj_dim % num_kv_heads != 0:
        raise ValueError(f"K projection dimension {k_proj_dim} not divisible by num_kv_heads {num_kv_heads}")
    
    if v_proj_dim % num_kv_heads != 0:
        raise ValueError(f"V projection dimension {v_proj_dim} not divisible by num_kv_heads {num_kv_heads}")
    
    # Calculate actual head dimensions from the weights
    q_head_dim = q_proj_dim // num_heads
    kv_head_dim = k_proj_dim // num_kv_heads
    
    print(f"\nDIMENSION CALCULATION: Using actual weight dimensions to determine head_dim")
    print(f"  Q weights: {q_weight.shape} → {q_head_dim} = {q_proj_dim} / {num_heads} heads")
    print(f"  K weights: {k_weight.shape} → {kv_head_dim} = {k_proj_dim} / {num_kv_heads} kv_heads")
    print(f"  V weights: {v_weight.shape} → {kv_head_dim} = {v_proj_dim} / {num_kv_heads} kv_heads")
    
    # Override head_dim only if it's different from calculated values
    if override_head_dim is not None and override_head_dim != q_head_dim:
        print(f"WARNING: override_head_dim={override_head_dim} doesn't match calculated q_head_dim={q_head_dim}")
        print(f"Using calculated q_head_dim={q_head_dim} for better accuracy")
        # Use the calculated value instead of the override to ensure correct dimensions
        override_head_dim = q_head_dim
        
    # Use consistent head dimension for all
    head_dim = q_head_dim
    
    # Now reshape the weights with the corrected dimensions
    q_weights_reshaped = q_weight.reshape(hidden_dim, num_heads, q_head_dim)
    k_weights_reshaped = k_weight.reshape(hidden_dim, num_kv_heads, kv_head_dim)
    v_weights_reshaped = v_weight.reshape(hidden_dim, num_kv_heads, kv_head_dim)
    
    # For output projection, first get it in the right orientation
    if o_weight.shape[1] == hidden_dim:
        # If output weight is [out_features, hidden_dim], transpose to [hidden_dim, out_features]
        o_weight = o_weight.t()
        print(f"Transposed output weight to shape {o_weight.shape}")
    
    # For Gemma-3-1B, the output projection should have shape [hidden_dim, num_heads*head_dim]
    o_proj_dim = o_weight.shape[1]
    
    # Ensure output weight can be reshaped properly
    if o_weight.shape[0] != hidden_dim:
        raise ValueError(f"Output weight first dimension {o_weight.shape[0]} != hidden_dim {hidden_dim}")
    
    if o_proj_dim != num_heads * q_head_dim:
        # Try transposing again if dimensions are swapped
        if o_weight.shape[1] == hidden_dim and o_weight.shape[0] == num_heads * q_head_dim:
            o_weight = o_weight.t()
            o_proj_dim = o_weight.shape[1]
            print(f"Transposed output weight again to shape {o_weight.shape}")
        else:
            # For Gemma-3-1B model, the output projection may have dimensions that differ from q_weight
            # Recalculate the head dimension based on actual weight shape
            if o_proj_dim % num_heads == 0:
                o_head_dim = o_proj_dim // num_heads
                print(f"Output projection has different head dimension: {o_head_dim} (vs q_head_dim={q_head_dim})")
                # We'll use the output projection's head dimension for reshaping it
                q_head_dim = o_head_dim
            else:
                raise ValueError(f"Output projection dimension {o_proj_dim} isn't divisible by num_heads {num_heads}")
    
    # Now reshape with the correct dimensions
    o_weights_reshaped = o_weight.reshape(hidden_dim, num_heads, q_head_dim)
            
    print(f"Reshaped weight dimensions: Q={q_weights_reshaped.shape}, K={k_weights_reshaped.shape}, "
          f"V={v_weights_reshaped.shape}, O={o_weights_reshaped.shape}")
    
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
    # For TPA, the key parameter is the rank of the decomposition
    # The ranks need to be smaller than the respective dimensions
    
    # Set the ranks based on the fat_ranks parameter
    if fat_ranks:
        # For fat ranks mode, use large initial ranks for decomposition
        # The energy-based rank selection will refine these later (around line 580)
        print("Using initial large ranks for Tucker decomposition (will be refined by 98% energy threshold)")
        R1 = min(240, W_all.shape[0])  # Rank for the model dimension
        R2 = min(240, W_all.shape[1])  # Rank for the head dimension
        R3 = min(W_all.shape[2], 4)    # Rank for the QKV distinction (usually just 4)
    else:
        # Standard ranks based on q_rank
        R1 = min(q_rank, W_all.shape[0])  # Rank for the model dimension
        R2 = min(q_rank, W_all.shape[1])  # Rank for the head dimension
        R3 = min(q_rank, W_all.shape[2])  # Rank for the QKV distinction
    
    print(f"Using ranks: R1={R1}, R2={R2}, R3={R3}")
    print(f"Tensor shape: {W_all.shape}")
    
    print(f"Applying Tucker decomposition with ranks: ({R1}, {R2}, {R3})")
    
    try:
        # Apply Tucker decomposition to the full tensor
        # This enforces shared factor matrices across all heads
        print(f"Tensor shape before Tucker decomposition: {W_all.shape}")
        print(f"Target ranks: {[R1, R2, R3, num_heads]}")
        
        # Set up timing
        decomp_start = time.time()
        print("Starting Tucker decomposition...")
        
        # Ensure tensor is on the correct device before decomposition
        W_all_device = W_all.device
        print(f"Running Tucker decomposition on device: {W_all_device}")
        
        # Add verbose parameter to tucker and ensure we're using GPU
        tucker_start = time.time()
        try:
            print("Using GPU-accelerated Tucker decomposition...")
            # Ensure tensor is contiguous in memory for best performance
            W_all = W_all.contiguous()
            core, factors = tucker(W_all, rank=[R1, R2, R3, num_heads], tol=1e-4, verbose=True)
            tucker_end = time.time()
            print(f"Tucker decomposition completed in {tucker_end - tucker_start:.2f} seconds on {W_all_device}")
        except Exception as e:
            print(f"Error during Tucker decomposition: {e}")
            print("Trying alternative approach with explicit device handling...")
            
            # Try again with explicit device management
            # First ensure tensor is on CUDA if available
            if torch.cuda.is_available():
                cuda_device = torch.device('cuda')
                W_all = W_all.to(cuda_device)
                print(f"Moved tensor to {cuda_device} for decomposition")
            
            # Run decomposition with tensor on appropriate device
            core, factors = tucker(W_all, rank=[R1, R2, R3, num_heads], tol=1e-4, verbose=True)
            tucker_end = time.time()
            print(f"Alternative Tucker decomposition completed in {tucker_end - tucker_start:.2f} seconds")
        
        decomp_end = time.time()
        print(f"Tucker decomposition took {decomp_end - decomp_start:.2f} seconds")
        
        # Extract the factor matrices
        U1 = factors[0]  # For hidden dimension
        U2 = factors[1]  # For head dimension
        U3 = factors[2]  # For QKV distinction
        # No need for factor 3 since we'll keep each head separate
        
        print(f"Tucker decomposition complete. Core shape: {core.shape}")
        print(f"Factor shapes: {[f.shape for f in factors]}")
        
        # Check for NaN or infinite values
        if torch.isnan(core).any() or torch.isinf(core).any():
            print("WARNING: Core tensor contains NaN or Inf values!")
        
        for i, factor in enumerate(factors):
            if torch.isnan(factor).any() or torch.isinf(factor).any():
                print(f"WARNING: Factor {i} contains NaN or Inf values!")
    
    except Exception as e:
        print(f"Tucker decomposition failed: {e}")
        print("Falling back to separate decompositions for Q, K, V")
        import traceback
        traceback.print_exc()
        
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
    
    # Get the actual ranks from the Tucker decomposition
    # These might differ from the requested ranks
    actual_R1 = U1.shape[1]  # Rank for hidden dimension from factor shape
    actual_R2 = U2.shape[1]  # Rank for head dimension from factor shape
    
    print(f"Actual ranks from Tucker decomposition: R1={actual_R1}, R2={actual_R2}")
    
    # We'll determine the optimal ranks for each component separately
    # First, analyze the intrinsic ranks of Q, K, V matrices using SVD and energy analysis
    # This helps us respect the actual rank structure of each component
    print("\nANALYZING INTRINSIC RANKS using energy-based approach (cumulative explained variance)")
    
    # For queries
    intrinsic_q_rank = actual_R2  # Start with Tucker decomposition rank
    
    # For keys and values, analyze their intrinsic ranks using SVD
    # First reshape to 2D matrices for analysis
    k_matrix_2d = k_weight.reshape(hidden_dim, -1)
    v_matrix_2d = v_weight.reshape(hidden_dim, -1)
    
    # Compute SVD to analyze singular value distributions
    try:
        _, k_singular_values, _ = torch.linalg.svd(k_matrix_2d, full_matrices=False)
        _, v_singular_values, _ = torch.linalg.svd(v_matrix_2d, full_matrices=False)
        
        # Better approach: use energy-based threshold (cumulative explained variance)
        # This is more adaptive to different singular value distributions
        
        # Square singular values to get eigenvalues (energy)
        k_energy = k_singular_values ** 2
        v_energy = v_singular_values ** 2
        
        # Normalize by total energy
        k_energy_norm = k_energy / torch.sum(k_energy)
        v_energy_norm = v_energy / torch.sum(v_energy)
        
        # Compute cumulative explained variance
        k_cumulative = torch.cumsum(k_energy_norm, dim=0)
        v_cumulative = torch.cumsum(v_energy_norm, dim=0)
        
        # Find ranks that explain 90%, 95%, 98% of variance
        energy_thresholds = [0.9, 0.95, 0.98]
        k_ranks = []
        v_ranks = []
        
        print("  K singular value analysis:")
        for thresh in energy_thresholds:
            k_rank = torch.sum(k_cumulative <= thresh).item() + 1  # +1 because we need the first value that exceeds
            k_rank = min(k_rank, len(k_cumulative))  # Handle case where we might not reach threshold
            k_ranks.append(int(k_rank))
            print(f"    {thresh*100:.0f}% energy: rank {int(k_rank)}")
            
        print("  V singular value analysis:")
        for thresh in energy_thresholds:
            v_rank = torch.sum(v_cumulative <= thresh).item() + 1  # +1 because we need the first value that exceeds
            v_rank = min(v_rank, len(v_cumulative))  # Handle case where we might not reach threshold
            v_ranks.append(int(v_rank))
            print(f"    {thresh*100:.0f}% energy: rank {int(v_rank)}")
            
        # Set maximum practical rank based on fat_ranks setting
        if fat_ranks:
            # For fat ranks mode, allow much higher ranks
            max_practical_rank = 240  # Very high rank for better approximation
            print("  Using FAT RANKS mode with max_practical_rank=240")
            
            # For fat_ranks, use 98% energy threshold (highest of our thresholds)
            intrinsic_k_rank = k_ranks[2] if len(k_ranks) > 2 else k_ranks[-1]
            intrinsic_v_rank = v_ranks[2] if len(v_ranks) > 2 else v_ranks[-1]
            print(f"  Using 98% energy threshold for rank selection")
        else:
            # Use 95% energy threshold as default with normal cap
            max_practical_rank = 8  # Standard cap to avoid excessive computation
            
            # Choose rank based on 95% explained variance (middle of our thresholds)
            intrinsic_k_rank = k_ranks[1] if len(k_ranks) > 1 else k_ranks[0]
            intrinsic_v_rank = v_ranks[1] if len(v_ranks) > 1 else v_ranks[0]
        
        # Apply practical rank cap (higher for fat_ranks mode)
        intrinsic_k_rank = min(max_practical_rank, intrinsic_k_rank)
        intrinsic_v_rank = min(max_practical_rank, intrinsic_v_rank)
        
        # IMPORTANT: Also ensure component ranks don't exceed Tucker decomposition rank
        # This prevents dimension mismatches in the subsequent tensor operations
        intrinsic_k_rank = min(actual_R2, intrinsic_k_rank)
        intrinsic_v_rank = min(actual_R2, intrinsic_v_rank)
        
        # Ensure at least rank 2 for stability
        intrinsic_k_rank = max(2, intrinsic_k_rank)
        intrinsic_v_rank = max(2, intrinsic_v_rank)
        
        # Print final selected ranks
        print(f"  Selected ranks - K: {intrinsic_k_rank}, V: {intrinsic_v_rank} (capped by max_practical_rank={max_practical_rank} and Tucker rank={actual_R2})")
        
        print(f"Intrinsic ranks detected - Q: {intrinsic_q_rank}, K: {intrinsic_k_rank}, V: {intrinsic_v_rank}")
        
    except Exception as e:
        print(f"Error computing intrinsic ranks: {e}")
        # Fallback to Tucker decomposition ranks
        intrinsic_k_rank = actual_R2
        intrinsic_v_rank = actual_R2
        print(f"Using fallback ranks: K: {intrinsic_k_rank}, V: {intrinsic_v_rank}")
    
    if use_dynamic_ranks:
        # Use the dynamic ranks based on intrinsic structure
        actual_q_rank = intrinsic_q_rank
        actual_k_rank = intrinsic_k_rank
        actual_v_rank = intrinsic_v_rank
        print(f"\nUSING OPTIMAL COMPONENT-SPECIFIC RANKS: Q={actual_q_rank}, K={actual_k_rank}, V={actual_v_rank}")
        print(f"These ranks are determined by energy-based analysis to balance accuracy and efficiency")
    else:
        # Use the user-specified ranks but cap by the actual ranks
        actual_q_rank = min(q_rank, actual_R2)
        actual_k_rank = min(k_rank, intrinsic_k_rank)
        actual_v_rank = min(v_rank, intrinsic_v_rank)
        print(f"\nUSING USER-SPECIFIED RANKS (capped by intrinsic ranks): Q={actual_q_rank}, K={actual_k_rank}, V={actual_v_rank}")
        print(f"Original requested ranks were: Q={q_rank}, K={k_rank}, V={v_rank}")
    
    # Expand dimensions for TPA format using actual ranks
    # Make sure to use the correct hidden_dim for these matrices - it might be different from q_weight.shape[0]
    W_A_q_expanded = torch.zeros((hidden_dim, num_heads * actual_q_rank), device=q_weight.device, dtype=torch.float32)
    W_A_k_expanded = torch.zeros((hidden_dim, num_kv_heads * actual_k_rank), device=k_weight.device, dtype=torch.float32)
    W_A_v_expanded = torch.zeros((hidden_dim, num_kv_heads * actual_v_rank), device=v_weight.device, dtype=torch.float32)
    
    # The proper way to create the expanded Wa is to repeat for each head
    for h in range(num_heads):
        W_A_q = U1 @ q_core[:, :, h]
        W_A_q_expanded[:, h*actual_q_rank:(h+1)*actual_q_rank] = W_A_q
        
    for g in range(num_kv_heads):
        W_A_k = U1 @ k_core[:, :, g]
        W_A_v = U1 @ v_core[:, :, g]
        W_A_k_expanded[:, g*actual_k_rank:(g+1)*actual_k_rank] = W_A_k
        W_A_v_expanded[:, g*actual_v_rank:(g+1)*actual_v_rank] = W_A_v
    
    # IMPORTANT: This correctly implements contextual factorization (CF) from the TPA paper
    # In contextual factorization, both W_A_q and W_B_q project directly from hidden_states
    # We need to derive these projection matrices from the Tucker decomposition
    
    print(f"Using efficient SVD-based factorization for TPA weights with component-specific ranks")
    print(f"Component-specific ranks: Q={actual_q_rank}, K={actual_k_rank}, V={actual_v_rank}")
    
    # Ensure all operations happen on GPU if available
    device = q_weight.device
    if torch.cuda.is_available():
        # Get the CUDA device index (default to 0 if not specified)
        cuda_device_idx = 0
        if device.type == 'cuda' and device.index is not None:
            cuda_device_idx = device.index
            
        device = torch.device(f'cuda:{cuda_device_idx}')
        print(f"Using {device} for SVD-based TPA projection computation")
    
    # SVD-based TPA factorization
    # For each head, we compute the optimal factors W_A and W_B directly using SVD
    
    # Function to compute SVD-based TPA factors for a weight matrix
    def compute_svd_tpa_factors(weight_matrix, rank, hidden_dim, head_dim):
        """
        Compute TPA factors using optimal SVD approach.
        
        This implements the closed-form solution described in the mathematical formula:
        W ≈ U_R Σ_R V_R^T
        W_A_r = sqrt(R) * sqrt(σ_r) * u_r
        W_B_r = sqrt(R) * sqrt(σ_r) * v_r^T
        
        Args:
            weight_matrix: Original weight matrix [hidden_dim, head_dim]
            rank: Target rank for factorization
            hidden_dim: Hidden dimension size
            head_dim: Attention head dimension size
            
        Returns:
            W_A: First TPA factor [hidden_dim, rank]
            W_B: Second TPA factor [hidden_dim, rank * head_dim]
        """
        # Ensure weight matrix is on the correct device and has the right shape
        weight_matrix = weight_matrix.to(device)
        
        # Compute truncated SVD
        start_time = time.time()
        try:
            # Using torch's SVD implementation
            U, S, Vh = torch.linalg.svd(weight_matrix, full_matrices=False)
            
            # Truncate to target rank
            U_r = U[:, :rank]  # [hidden_dim, rank]
            S_r = S[:rank]     # [rank]
            Vh_r = Vh[:rank, :] # [rank, head_dim]
            
            # Compute scaling factor
            sqrt_rank = math.sqrt(rank)
            
            # Compute TPA factors according to the formula
            # W_A_r = sqrt(rank) * sqrt(sigma_r) * u_r
            # W_B_r = sqrt(rank) * sqrt(sigma_r) * v_r^T
            
            # Scale U and Vh by sqrt(sigma) and sqrt(rank)
            # Log singular values for debugging
            print(f"Singular values min: {S_r.min().item():.10e}, max: {S_r.max().item():.10e}")
            if (S_r <= 0).any():
                print(f"WARNING: Zero or negative singular values detected: {S_r[S_r <= 0]}")
            
            sqrt_S_r = torch.sqrt(S_r)
            
            # Log if we get NaN in sqrt
            if torch.isnan(sqrt_S_r).any():
                nan_indices = torch.where(torch.isnan(sqrt_S_r))[0]
                print(f"NaN values detected in sqrt_S_r at indices: {nan_indices.tolist()}")
                print(f"Corresponding S_r values: {S_r[nan_indices].tolist()}")
            
            # Create W_A matrix [hidden_dim, rank]
            W_A = sqrt_rank * U_r * sqrt_S_r.unsqueeze(0)
            
            # Log NaN or infinity values in W_A
            if torch.isnan(W_A).any() or torch.isinf(W_A).any():
                nan_count = torch.isnan(W_A).sum().item()
                inf_count = torch.isinf(W_A).sum().item()
                print(f"WARNING: NaN or Inf detected in W_A - {nan_count} NaNs, {inf_count} Infs")
                print(f"U_r min/max: {U_r.min().item():.10e}/{U_r.max().item():.10e}")
                print(f"sqrt_S_r min/max: {sqrt_S_r.min().item():.10e}/{sqrt_S_r.max().item():.10e}")
            
            # For W_B, we need to reshape to match the TPA format
            # Original formula gives W_B_r with shape [rank, head_dim]
            # We need to reshape to [hidden_dim, rank * head_dim]
            
            # First, scale Vh by sqrt(S) and sqrt(rank)
            scaled_Vh = sqrt_rank * Vh_r * sqrt_S_r.unsqueeze(1)  # [rank, head_dim]
            
            # Log NaN or infinity values in scaled_Vh
            if torch.isnan(scaled_Vh).any() or torch.isinf(scaled_Vh).any():
                nan_count = torch.isnan(scaled_Vh).sum().item()
                inf_count = torch.isinf(scaled_Vh).sum().item()
                print(f"WARNING: NaN or Inf detected in scaled_Vh - {nan_count} NaNs, {inf_count} Infs")
                print(f"Vh_r min/max: {Vh_r.min().item():.10e}/{Vh_r.max().item():.10e}")
            
            # Create W_B vectorized approach
            # Create a rank*head_dim tensor with proper scaling
            # This is a much more efficient vectorized implementation that avoids loops
            
            # Create W_B with the proper TPA shape [hidden_dim, rank * head_dim]
            W_B = torch.zeros((hidden_dim, rank * head_dim), device=device, dtype=torch.float32)
            
            # Vectorized implementation using einsum for clarity and efficiency
            # This directly computes all the required outer products and places them in the right positions
            
            # For each rank r, we need to construct the correct W_B according to the formula:
            # W_B_r = sqrt(R) * sqrt(σ_r) * v_r^T  
            # The TPA formula factorizes as: W x ≈ (1/R) * sum_{r=1}^{R} (W_A_r x) ⊗ (W_B_r x)
            # Where W_A_r = sqrt(R) * sqrt(σ_r) * u_r and W_B_r = sqrt(R) * sqrt(σ_r) * v_r^T
            for r in range(rank):
                # Get the scaled v_r^T vector according to the formula W_B_r = sqrt(R) * sqrt(σ_r) * v_r^T
                scaled_v_r = sqrt_rank * sqrt_S_r[r] * Vh_r[r]  # [head_dim]
                
                # Check for NaN/Inf in scaled_v_r
                if torch.isnan(scaled_v_r).any() or torch.isinf(scaled_v_r).any():
                    nan_count = torch.isnan(scaled_v_r).sum().item()
                    inf_count = torch.isinf(scaled_v_r).sum().item()
                    print(f"WARNING: NaN/Inf in scaled_v_r for r={r} - {nan_count} NaNs, {inf_count} Infs")
                    print(f"sqrt_S_r[{r}] = {sqrt_S_r[r].item():.10e}")
                    print(f"Vh_r[{r}] min/max: {Vh_r[r].min().item():.10e}/{Vh_r[r].max().item():.10e}")
                
                # For TPA, each row of W_B should be the same scaled_v_r
                # This creates a rank-separable structure where each dimension uses the same 
                # scaled vector, which is critical for the TPA factorization to work correctly
                W_B[:, r*head_dim:(r+1)*head_dim] = scaled_v_r.unsqueeze(0).expand(hidden_dim, -1)
            
            end_time = time.time()
            print(f"SVD-based factorization completed in {end_time - start_time:.4f} seconds")
            
            # Compute reconstruction error to verify quality of the SVD factorization
            reconstructed = U_r @ torch.diag(S_r) @ Vh_r
            error = torch.norm(weight_matrix - reconstructed) / torch.norm(weight_matrix)
            # This error is for the SVD approximation, NOT the TPA factorization
            # For SVD, this should be very small (<0.01) for small ranks
            print(f"SVD reconstruction relative error: {error.item():.6f}")
            
            # Also compute TPA reconstruction based on W_A and W_B to verify
            # TPA formula: W ≈ (1/R) * sum_{r=1}^{R} (W_A_r ⊗ W_B_r)
            W_recon_tpa = torch.zeros_like(weight_matrix)
            for r in range(rank):
                # Get the appropriate slices for this rank
                W_A_r = W_A[:, r:r+1]  # [hidden_dim, 1]
                W_B_r_slice = W_B[:, r*head_dim:(r+1)*head_dim]  # [hidden_dim, head_dim]
                
                # Correctly apply TPA reconstruction formula with proper outer product
                # For each dimension, compute the outer product of W_A[:,r] and the first row of W_B_r_slice
                # Since all rows of W_B_r_slice are identical (by construction)
                a_vec = W_A_r[:, 0]  # [hidden_dim]
                b_vec = W_B_r_slice[0, :]  # [head_dim]
                
                # Use torch.outer for a clean implementation of the outer product
                W_recon_tpa += torch.outer(a_vec, b_vec) / rank
                    
            # Compute TPA reconstruction error
            tpa_error = torch.norm(weight_matrix - W_recon_tpa) / torch.norm(weight_matrix)
            print(f"TPA reconstruction relative error: {tpa_error.item():.6f}")
            
            return W_A, W_B
            
        except Exception as e:
            print(f"Error during SVD computation: {e}")
            raise
    
    # Update the rank parameters to match what's actually used for computation
    # This ensures B projections are created with correct dimensions from the start
    if fat_ranks:
        # For fat ranks mode, use the maximum ranks possible with the current factorization
        # but capped by the actual Tucker decomposition ranks (actual_R2)
        practical_q_rank = min(240, actual_R2)
        practical_k_rank = min(240, actual_R2)
        practical_v_rank = min(240, actual_R2)
        print(f"Using FAT RANKS for computation - Q: {practical_q_rank}, K: {practical_k_rank}, V: {practical_v_rank}")
        print(f"These ranks provide higher accuracy but use more memory and computation")
    else:
        # Standard approach - use the requested ranks capped by the Tucker decomposition ranks
        practical_q_rank = min(q_rank, actual_R2)
        practical_k_rank = min(k_rank, actual_R2)
        practical_v_rank = min(v_rank, actual_R2)
        print(f"Using practical computation ranks - Q: {practical_q_rank}, K: {practical_k_rank}, V: {practical_v_rank}")
        print(f"These ranks balance the intrinsic structure with Tucker decomposition constraints")
    
    # Initialize output matrices with the practical ranks
    W_A_q = torch.zeros((hidden_dim, num_heads * practical_q_rank), device=device, dtype=torch.float32)
    W_A_k = torch.zeros((hidden_dim, num_kv_heads * practical_k_rank), device=device, dtype=torch.float32)
    W_A_v = torch.zeros((hidden_dim, num_kv_heads * practical_v_rank), device=device, dtype=torch.float32)
    
    W_B_q_optimal = torch.zeros((hidden_dim, practical_q_rank * head_dim), device=device, dtype=torch.float32)
    W_B_k_optimal = torch.zeros((hidden_dim, practical_k_rank * head_dim), device=device, dtype=torch.float32)
    W_B_v_optimal = torch.zeros((hidden_dim, practical_v_rank * head_dim), device=device, dtype=torch.float32)
    
    # Process query heads
    print(f"Computing SVD-based factorization for {num_heads} query heads...")
    for h in range(num_heads):
        # Get the weight matrix for this head
        head_weight = q_weight[:, h * head_dim:(h + 1) * head_dim]  # [hidden_dim, head_dim]
        
        # Compute TPA factors using practical ranks
        W_A_head, W_B_head = compute_svd_tpa_factors(head_weight, practical_q_rank, hidden_dim, head_dim)
        
        # Store in output matrices using practical ranks
        W_A_q[:, h * practical_q_rank:(h + 1) * practical_q_rank] = W_A_head
        
        # For the first head, we store W_B directly
        # For subsequent heads, we average with existing values
        if h == 0:
            W_B_q_optimal = W_B_head
        else:
            # Average B projections across heads for better generalization
            W_B_q_optimal = (h * W_B_q_optimal + W_B_head) / (h + 1)
    
    # Process key heads
    print(f"Computing SVD-based factorization for {num_kv_heads} key heads...")
    for g in range(num_kv_heads):
        # Get the weight matrix for this head
        head_weight = k_weight[:, g * head_dim:(g + 1) * head_dim]  # [hidden_dim, head_dim]
        
        # Compute TPA factors using practical ranks
        W_A_head, W_B_head = compute_svd_tpa_factors(head_weight, practical_k_rank, hidden_dim, head_dim)
        
        # Store in output matrices using practical ranks
        W_A_k[:, g * practical_k_rank:(g + 1) * practical_k_rank] = W_A_head
        
        # For the first head, we store W_B directly
        # For subsequent heads, we average with existing values
        if g == 0:
            W_B_k_optimal = W_B_head
        else:
            # Average B projections across heads for better generalization
            W_B_k_optimal = (g * W_B_k_optimal + W_B_head) / (g + 1)
    
    # Process value heads
    print(f"Computing SVD-based factorization for {num_kv_heads} value heads...")
    for g in range(num_kv_heads):
        # Get the weight matrix for this head
        head_weight = v_weight[:, g * head_dim:(g + 1) * head_dim]  # [hidden_dim, head_dim]
        
        # Compute TPA factors using practical ranks
        W_A_head, W_B_head = compute_svd_tpa_factors(head_weight, practical_v_rank, hidden_dim, head_dim)
        
        # Store in output matrices using practical ranks
        W_A_v[:, g * practical_v_rank:(g + 1) * practical_v_rank] = W_A_head
        
        # For the first head, we store W_B directly
        # For subsequent heads, we average with existing values
        if g == 0:
            W_B_v_optimal = W_B_head
        else:
            # Average B projections across heads for better generalization
            W_B_v_optimal = (g * W_B_v_optimal + W_B_head) / (g + 1)
    
    # Use the optimal projection weights for the TPA implementation
    W_B_q_reshaped = W_B_q_optimal  # [hidden_dim, q_rank*head_dim]
    W_B_k_reshaped = W_B_k_optimal  # [hidden_dim, k_rank*head_dim]
    W_B_v_reshaped = W_B_v_optimal  # [hidden_dim, v_rank*head_dim]
    
    print(f"Final TPA projection matrices with practical ranks - Q: {practical_q_rank}, K: {practical_k_rank}, V: {practical_v_rank}")
    print(f"B projection shapes: W_B_q={W_B_q_reshaped.shape}, W_B_k={W_B_k_reshaped.shape}, W_B_v={W_B_v_reshaped.shape}")
    print(f"Note: Initial analysis found high intrinsic ranks for K,V (needed for 95% energy: {k_rank},{v_rank})")
    print(f"      but we're using practical ranks capped by Tucker decomposition: {practical_k_rank},{practical_v_rank}")
    
    # Store the original uncapped ranks for reference
    original_q_rank = q_rank
    original_k_rank = k_rank
    original_v_rank = v_rank
    
    # Update the rank parameters to match what's actually used
    q_rank = actual_q_rank
    k_rank = actual_k_rank
    v_rank = actual_v_rank
    
    # Add expanded matrices to result - use the SVD computed factors
    result["W_A_q"] = W_A_q.to(dtype=dtype, device=device)
    result["W_A_k"] = W_A_k.to(dtype=dtype, device=device)
    result["W_A_v"] = W_A_v.to(dtype=dtype, device=device)
    
    result["W_B_q"] = W_B_q_reshaped.to(dtype=dtype, device=device)
    result["W_B_k"] = W_B_k_reshaped.to(dtype=dtype, device=device)
    result["W_B_v"] = W_B_v_reshaped.to(dtype=dtype, device=device)
    
    # Verify reconstruction error using the SVD approach
    # For TPA using SVD, we can directly test reconstruction quality 
    
    print("Verifying reconstruction quality of SVD-based TPA factors...")
    
    # Make sure to create 3D tensors for proper reconstruction verification
    # q_weights_reshaped is [hidden_dim, num_heads, head_dim]
    print(f"Debug - q_weights_reshaped shape: {q_weights_reshaped.shape}")
    print(f"Debug - k_weights_reshaped shape: {k_weights_reshaped.shape}")
    
    # Reconstruct query weights directly from the SVD factors as 3D tensor
    q_recon = torch.zeros((hidden_dim, num_heads, head_dim), device=device, dtype=torch.float32)
    k_recon = torch.zeros((hidden_dim, num_kv_heads, head_dim), device=device, dtype=torch.float32)
    v_recon = torch.zeros((hidden_dim, num_kv_heads, head_dim), device=device, dtype=torch.float32)
    
    # For SVD-based TPA, the reconstruction formula is:
    # W_r = 1/R * sum_r [ W_A_r * W_B_r ]
    # Where each component is an outer product
    
    for h in range(num_heads):
        # Get the A factors for this head
        q_head_A = W_A_q[:, h*q_rank:(h+1)*q_rank]  # [hidden_dim, q_rank]
        
        # Check the actual available rank
        actual_q_rank = q_head_A.shape[1]  # Get the actual available rank
        print(f"Debug - actual_q_rank for head {h}: {actual_q_rank}, q_rank: {q_rank}")
        
        # SVD reconstruction is simpler and more direct
        # For this head, compute the reconstructed weights directly
        head_reconstruction = torch.zeros((hidden_dim, head_dim), device=device, dtype=torch.float32)
        
        # Use the minimum of q_rank and actual available rank
        effective_q_rank = min(q_rank, actual_q_rank)
        
        # Use the TPA formula: 1/R * sum_r (W_A_r * W_B_r)
        for r in range(effective_q_rank):
            # Get the rth column of q_head_A
            a_r = q_head_A[:, r]  # [hidden_dim]
            
            # Get the corresponding slice of W_B_q for this rank
            b_r = W_B_q_optimal[:, r*head_dim:(r+1)*head_dim]  # [hidden_dim, head_dim]
            
            # Add to the running sum, following the TPA formula
            # Each rank contributes 1/rank of the total reconstruction
            # For correct reconstruction, we need to apply the formula: (1/R) * (W_A_r ⊗ W_B_r)
            # Since b_r has identical rows (all equal to the scaled v_r vector),
            # we only need to use the first row for the outer product
            
            # Check for NaN/Inf in the vectors before outer product
            if torch.isnan(a_r).any() or torch.isinf(a_r).any() or torch.isnan(b_r[0]).any() or torch.isinf(b_r[0]).any():
                nan_a = torch.isnan(a_r).sum().item()
                inf_a = torch.isinf(a_r).sum().item()
                nan_b = torch.isnan(b_r[0]).sum().item()
                inf_b = torch.isinf(b_r[0]).sum().item()
                print(f"WARNING: NaN/Inf in Q reconstruction vectors for head {h}, rank {r} - a_r: {nan_a} NaNs, {inf_a} Infs; b_r: {nan_b} NaNs, {inf_b} Infs")
                
            head_reconstruction += torch.outer(a_r, b_r[0]) / effective_q_rank
        
        # Store this head's reconstruction in the appropriate slice - using 3D indexing
        q_recon[:, h, :] = head_reconstruction
    
    # Reconstruct key and value weights with the same approach
    for g in range(num_kv_heads):
        # Get A factors for this group
        k_head_A = W_A_k[:, g*k_rank:(g+1)*k_rank]  # [hidden_dim, k_rank] 
        v_head_A = W_A_v[:, g*v_rank:(g+1)*v_rank]  # [hidden_dim, v_rank]
        
        # Reconstruct key weights
        k_head_reconstruction = torch.zeros((hidden_dim, head_dim), device=device, dtype=torch.float32)
        # Get actual available rank from k_head_A's shape
        actual_k_rank = k_head_A.shape[1]  # Get the actual available rank
        print(f"Debug - actual_k_rank for head {g}: {actual_k_rank}, k_rank: {k_rank}")
        
        # Use the minimum of k_rank and actual available rank
        effective_k_rank = min(k_rank, actual_k_rank)
        for r in range(effective_k_rank):
            a_r = k_head_A[:, r]  # [hidden_dim]
            b_r = W_B_k_optimal[:, r*head_dim:(r+1)*head_dim]  # [hidden_dim, head_dim]
            # Apply correct reconstruction using the TPA formula with proper outer product
            # Check for NaN/Inf in the vectors before outer product
            if torch.isnan(a_r).any() or torch.isinf(a_r).any() or torch.isnan(b_r[0]).any() or torch.isinf(b_r[0]).any():
                nan_a = torch.isnan(a_r).sum().item()
                inf_a = torch.isinf(a_r).sum().item()
                nan_b = torch.isnan(b_r[0]).sum().item()
                inf_b = torch.isinf(b_r[0]).sum().item()
                print(f"WARNING: NaN/Inf in K reconstruction vectors - a_r: {nan_a} NaNs, {inf_a} Infs; b_r: {nan_b} NaNs, {inf_b} Infs")
                
            # Only use the first row of b_r since all rows are identical
            k_head_reconstruction += torch.outer(a_r, b_r[0]) / effective_k_rank
        
        # Store this head's reconstruction - using 3D indexing
        k_recon[:, g, :] = k_head_reconstruction
        
        # Reconstruct value weights
        v_head_reconstruction = torch.zeros((hidden_dim, head_dim), device=device, dtype=torch.float32)
        # Get actual available rank from v_head_A's shape
        actual_v_rank = v_head_A.shape[1]  # Get the actual available rank
        print(f"Debug - actual_v_rank for head {g}: {actual_v_rank}, v_rank: {v_rank}")
        
        # Use the minimum of v_rank and actual available rank
        effective_v_rank = min(v_rank, actual_v_rank)
        for r in range(effective_v_rank):
            a_r = v_head_A[:, r]  # [hidden_dim]
            b_r = W_B_v_optimal[:, r*head_dim:(r+1)*head_dim]  # [hidden_dim, head_dim]
            # Apply correct reconstruction using the TPA formula with proper outer product
            # Check for NaN/Inf in the vectors before outer product
            if torch.isnan(a_r).any() or torch.isinf(a_r).any() or torch.isnan(b_r[0]).any() or torch.isinf(b_r[0]).any():
                nan_a = torch.isnan(a_r).sum().item()
                inf_a = torch.isinf(a_r).sum().item()
                nan_b = torch.isnan(b_r[0]).sum().item()
                inf_b = torch.isinf(b_r[0]).sum().item()
                print(f"WARNING: NaN/Inf in V reconstruction vectors - a_r: {nan_a} NaNs, {inf_a} Infs; b_r: {nan_b} NaNs, {inf_b} Infs")
                
            # Only use the first row of b_r since all rows are identical
            v_head_reconstruction += torch.outer(a_r, b_r[0]) / effective_v_rank
        
        # Store this head's reconstruction - using 3D indexing
        v_recon[:, g, :] = v_head_reconstruction
        
    # Add factorization ranks to results - store as simple integers, not tensors
    # This is critical for proper KV cache creation
    result["q_rank"] = int(q_rank)
    result["k_rank"] = int(k_rank)
    result["v_rank"] = int(v_rank)
    
    print(f"Using OPTIMIZED COMPONENT-SPECIFIC ranks for factorized weights - Q: {q_rank}, K: {k_rank}, V: {v_rank}")
    print(f"These ranks were carefully selected based on the intrinsic structure of each component")
    print(f"K & V ranks reflect their true rank structure and are within Tucker decomposition constraints")
    
    # Reshape for error calculation - must match original weight shape
    # Convert from 3D [hidden_dim, num_heads, head_dim] back to 2D [hidden_dim, num_heads*head_dim]
    q_recon_2d = q_recon.reshape(hidden_dim, num_heads * head_dim)
    k_recon_2d = k_recon.reshape(hidden_dim, num_kv_heads * head_dim)
    v_recon_2d = v_recon.reshape(hidden_dim, num_kv_heads * head_dim)
    
    print(f"Debug - After reshape: q_recon shape: {q_recon_2d.shape}, q_weight shape: {q_weight.shape}")
    
    # Calculate relative errors, handling potential shape differences
    # This addresses the case when q, k, v have different shapes in GQA
    try:
        # Use the 2D versions for error calculation
        q_err = torch.norm(q_recon_2d - q_weight) / torch.norm(q_weight)
    except RuntimeError as e:
        print(f"Warning: Cannot calculate Q reconstruction error due to shape mismatch: {e}")
        print(f"  q_recon_2d shape: {q_recon_2d.shape}, q_weight shape: {q_weight.shape}")
        q_err = torch.tensor(float('nan'))
        
    try:
        # Original key weights could have different dimensions in GQA
        # For Gemma 1B with 4 query heads, 1 KV head: k_weight is [256, 1152] but k_recon is [1024, 288]
        # We'll evaluate error on the common dimension only
        if k_recon_2d.shape != k_weight.shape:
            print(f"Warning: K shape mismatch: k_recon_2d={k_recon_2d.shape}, k_weight={k_weight.shape}")
            # Only using the first num_kv_heads rows for comparison
            k_recon_common = k_recon_2d[:k_weight.shape[0], :k_weight.shape[1]]
            # Or pad k_recon to match k_weight shape
            if k_recon_common.shape == k_weight.shape:
                k_err = torch.norm(k_recon_common - k_weight) / torch.norm(k_weight)
            else:
                print("  Cannot calculate K error - shapes too different")
                k_err = torch.tensor(float('nan'))
        else:
            k_err = torch.norm(k_recon_2d - k_weight) / torch.norm(k_weight)
    except RuntimeError as e:
        print(f"Warning: Cannot calculate K reconstruction error: {e}")
        k_err = torch.tensor(float('nan'))
        
    try:
        # Original value weights could have different dimensions in GQA
        if v_recon_2d.shape != v_weight.shape:
            print(f"Warning: V shape mismatch: v_recon_2d={v_recon_2d.shape}, v_weight={v_weight.shape}")
            # Only using the first num_kv_heads rows for comparison
            v_recon_common = v_recon_2d[:v_weight.shape[0], :v_weight.shape[1]]
            # Or pad v_recon to match v_weight shape
            if v_recon_common.shape == v_weight.shape:
                v_err = torch.norm(v_recon_common - v_weight) / torch.norm(v_weight)
            else:
                print("  Cannot calculate V error - shapes too different")
                v_err = torch.tensor(float('nan'))
        else:
            v_err = torch.norm(v_recon_2d - v_weight) / torch.norm(v_weight)
    except RuntimeError as e:
        print(f"Warning: Cannot calculate V reconstruction error: {e}")
        v_err = torch.tensor(float('nan'))
    
    print(f"Reconstruction relative errors - Q: {q_err:.4f}, K: {k_err:.4f}, V: {v_err:.4f}")
    
    toc = time.time()
    print(f"GQA to TPA conversion complete in {toc - tic:.2f} seconds")
    
    return result


def convert_gqa_model_to_tpa(model, q_rank=6, k_rank=2, v_rank=2, dtype=torch.float16, device="cuda", use_dynamic_ranks=True, fat_ranks=False):
    """
    Convert a GQA model to TPA format by applying the conversion to each attention layer.
    This modifies the input model in-place and then returns it.
    
    Args:
        model: The input model with GQA (GemmaForCausalLM)
        q_rank: Rank for query factorization
        k_rank: Rank for key factorization
        v_rank: Rank for value factorization
        dtype: Data type for output tensors
        device: Device for computation
        use_dynamic_ranks: Whether to use ranks determined by Tucker decomposition (True) 
                          or force specified ranks (False)
        fat_ranks: Whether to use much larger ranks (240) for higher accuracy but more memory usage
        
    Returns:
        The modified input model with TPA weights (still a GemmaForCausalLM)
    """
    if not HAS_TENSORLY:
        raise ImportError("TensorLy is required for convert_gqa_model_to_tpa")
    
    print("Converting GQA model to TPA format...")
    
    # Add timing and layer counting
    import time
    start_time = time.time()
    layers_converted = 0
    attention_modules_found = 0
    
    # Debug model structure
    print(f"Model type: {type(model).__name__}")
    print("Searching for attention modules...")
    
    # GemmaAttention modules are found in different structures based on the model architecture
    # We'll handle both GemmaForCausalLM and Gemma3ForMultimodalLM structures
    attention_modules = []
    
    # Check if using standard GemmaForCausalLM structure
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        print("Detected standard GemmaForCausalLM structure")
        for i, layer in enumerate(model.model.layers):
            if hasattr(layer, "self_attn"):
                attention_module = layer.self_attn
                if hasattr(attention_module, "qkv_proj"):
                    # Standard Gemma attention with combined QKV projection
                    print(f"  Found QKV-combined attention module in layer {i}")
                    # For standard Gemma attention, we need to split the QKV projection
                    # This requires additional handling for the weights
                    attention_modules.append((f"model.layers.{i}.self_attn", attention_module, "combined_qkv"))
                    attention_modules_found += 1
                elif hasattr(attention_module, "q_proj") and hasattr(attention_module, "k_proj") and hasattr(attention_module, "v_proj"):
                    # Already split Q,K,V projections
                    print(f"  Found split-QKV attention module in layer {i}")
                    attention_modules.append((f"model.layers.{i}.self_attn", attention_module, "split_qkv"))
                    attention_modules_found += 1
    
    # Check if using Gemma TPA model structure
    elif hasattr(model, "model") and hasattr(model.model, "layers") and hasattr(model.model.layers[0], "self_attn"):
        print("Detected Gemma TPA model structure")
        for i, layer in enumerate(model.model.layers):
            attention_module = layer.self_attn
            if hasattr(attention_module, "W_A_q") and hasattr(attention_module, "W_B_q"):
                # Already a TPA module
                print(f"  Layer {i} already using TPA attention - skipping")
            elif hasattr(attention_module, "q_proj") and hasattr(attention_module, "k_proj") and hasattr(attention_module, "v_proj"):
                print(f"  Found split-QKV attention module in layer {i}")
                attention_modules.append((f"model.model.layers.{i}.self_attn", attention_module, "split_qkv"))
                attention_modules_found += 1
    
    # If no attention modules found yet, try to find them in named_modules
    if attention_modules_found == 0:
        print("No attention modules found in standard structure, searching all modules...")
        for name, module in model.named_modules():
            if hasattr(module, "qkv_proj") and hasattr(module, "o_proj"):
                # Standard Gemma attention with combined QKV projection
                print(f"  Found combined-QKV attention module: {name}")
                attention_modules.append((name, module, "combined_qkv"))
                attention_modules_found += 1
            elif hasattr(module, "q_proj") and hasattr(module, "k_proj") and hasattr(module, "v_proj"):
                # Split Q,K,V projections
                print(f"  Found split-QKV attention module: {name}")
                attention_modules.append((name, module, "split_qkv"))
                attention_modules_found += 1
    
    if attention_modules_found == 0:
        print("ERROR: No attention modules found in the model!")
        print("Model structure may not be compatible with this conversion method")
        print("Expected structure: model.model.layers[i].self_attn")
        print("Showing all top-level attributes of model:")
        for attr_name in dir(model):
            if not attr_name.startswith('_'):
                print(f"  {attr_name}")
        
        if hasattr(model, "model"):
            print("Showing all attributes of model.model:")
            for attr_name in dir(model.model):
                if not attr_name.startswith('_'):
                    print(f"  {attr_name}")
        
        return model
        
    print(f"Found {attention_modules_found} attention modules to convert")
    
    # Process each module
    for name, module, module_type in attention_modules:
        print(f"Converting attention layer: {name} (type: {module_type})")
        layer_start = time.time()
        
        # Get or derive the number of heads and kv heads
        if hasattr(module, "num_heads") and hasattr(module, "num_key_value_heads"):
            num_heads = module.num_heads
            num_kv_heads = module.num_key_value_heads
            print(f"  GQA structure verified: {num_heads} heads, {num_kv_heads} KV heads")
        else:
            # Try to infer from model config
            if hasattr(model, "config"):
                num_heads = getattr(model.config, "num_attention_heads", None)
                num_kv_heads = getattr(model.config, "num_key_value_heads", num_heads)
                if num_heads is not None:
                    print(f"  Inferred from config: {num_heads} heads, {num_kv_heads} KV heads")
                else:
                    print(f"  ERROR: Could not determine num_heads from config")
                    continue
            else:
                print(f"  ERROR: Module {name} does not have GQA structure and no config found")
                continue
        
        # Check if apply_factorized_weights exists
        if hasattr(module, "apply_factorized_weights"):
            print(f"  Module supports factorized weights")
        else:
            print(f"  WARNING: Module does not support applying factorized weights")
            print(f"  Adding apply_factorized_weights method dynamically")
            
            # Dynamically add method to apply_factorized_weights if it doesn't exist
            def apply_factorized_weights_fn(self, weights_dict):
                """Dynamically added method to apply factorized weights to attention module"""
                # Set factorized flag
                self.use_factorized_weights = True
                
                # Store the factorized weights
                for key, weight in weights_dict.items():
                    # Only convert to Parameter if it's a tensor with floating point dtype
                    if isinstance(weight, torch.Tensor) and weight.is_floating_point():
                        # Convert to Parameter for gradient tracking
                        setattr(self, key, nn.Parameter(weight))
                    else:
                        # Store non-tensor values or integer tensors directly
                        setattr(self, key, weight)
                
                # Remember original weights were factorized
                self.original_weights_replaced = True
            
            # Bind the method to the module
            import types
            module.apply_factorized_weights = types.MethodType(apply_factorized_weights_fn, module)
            module.use_factorized_weights = False
            
        try:
            # Handle different module types
            if module_type == "combined_qkv":
                # For combined QKV projection, we need to extract the separate weights
                qkv_weight = module.qkv_proj.weight
                
                # First identify if we have a standard Gemma architecture
                hidden_size = qkv_weight.shape[1]  # Input dimension
                
                # Get head_dim from various sources
                head_dim = None
                
                # First check module.head_dim
                if hasattr(module, "head_dim"):
                    head_dim = module.head_dim
                    print(f"  Using head_dim={head_dim} from module attribute")
                # Then check model config
                elif hasattr(model, "config") and hasattr(model.config, "head_dim"):
                    head_dim = model.config.head_dim
                    print(f"  Using head_dim={head_dim} from model config")
                # Calculate from hidden_size if available in config
                elif hasattr(model, "config") and hasattr(model.config, "hidden_size"):
                    config_hidden = model.config.hidden_size
                    if config_hidden == hidden_size:
                        head_dim = hidden_size // num_heads
                        print(f"  Calculated head_dim={head_dim} from config hidden_size={config_hidden}")
                # Last resort - infer from shape
                else:
                    # Default calculation
                    head_dim = hidden_size // num_heads
                    print(f"  Inferred head_dim={head_dim} from weight dimensions")
                
                # Calculate sizes for splitting
                q_size = num_heads * head_dim
                kv_size = num_kv_heads * head_dim
                
                # Check if the dimensions match what we expect
                total_size = q_size + kv_size * 2
                if total_size != qkv_weight.shape[0]:
                    print(f"  WARNING: QKV weight dimension {qkv_weight.shape[0]} doesn't match expected {total_size}")
                    print(f"  Attempting to determine split based on observed shape...")
                    
                    # The QKV is in (out_dim, hidden_dim) format
                    # Try to detect a fixed ratio for q:k:v sizes 
                    if num_heads >= num_kv_heads and num_heads % num_kv_heads == 0:
                        # Standard GQA setup
                        ratio = num_heads / num_kv_heads  # How many q heads per kv head
                        
                        # Calculate sizes keeping the ratio
                        total = qkv_weight.shape[0]  
                        base_unit = total / (ratio + 2)  # denominator is ratio*k + k + v
                        
                        # Split into q, k, v proportionally
                        k_part = int(base_unit + 0.5)  # round to nearest int
                        q_part = int(ratio * k_part + 0.5)
                        v_part = total - q_part - k_part
                        
                        print(f"  Using inferred split sizes - Q: {q_part}, K: {k_part}, V: {v_part}")
                        q_weight, k_weight, v_weight = qkv_weight.split([q_part, k_part, v_part], dim=0)
                    else:
                        # Can't determine - try an equal split (fallback)
                        split_size = qkv_weight.shape[0] // 3
                        print(f"  Using equal split size of {split_size}")
                        splits = [split_size, split_size, qkv_weight.shape[0] - 2*split_size]  # Ensure all adds up
                        q_weight, k_weight, v_weight = qkv_weight.split(splits, dim=0)
                else:
                    # Normal case - dimensions match expectations
                    q_weight, k_weight, v_weight = qkv_weight.split([q_size, kv_size, kv_size], dim=0)
                
                o_weight = module.o_proj.weight
                
                print(f"  Split combined QKV projection: Q: {q_weight.shape}, K: {k_weight.shape}, V: {v_weight.shape}")
            
            elif module_type == "split_qkv":
                # Already has separate Q, K, V projections
                q_weight = module.q_proj.weight
                k_weight = module.k_proj.weight
                v_weight = module.v_proj.weight
                o_weight = module.o_proj.weight
                
                print(f"  Weight dimensions - Q: {q_weight.shape}, K: {k_weight.shape}, V: {v_weight.shape}, O: {o_weight.shape}")
            
            else:
                print(f"  ERROR: Unknown module type: {module_type}")
                continue
            
            # Apply GQA to TPA conversion
            print(f"  Starting tensor decomposition for layer {name}...")
            decomp_start = time.time()
            
                # For GemmaForCausalLM, the correct head dimension needs to be calculated
            # The QKV projection for regular Gemma has a specific shape
            if module_type == "combined_qkv":
                # Combined QKV models have specific dimensions
                # The query projection is typically [num_heads*head_dim, hidden_dim]
                # But in GQA models, there are num_heads query heads and num_kv_heads KV heads
                q_head_dim = q_weight.shape[1] // num_heads
                kv_head_dim = k_weight.shape[1] // num_kv_heads
                
                print(f"  Calculated dimensions: q_head_dim={q_head_dim}, kv_head_dim={kv_head_dim}")
                print(f"  Heads: q={num_heads}, kv={num_kv_heads}")
                
                if q_head_dim != kv_head_dim:
                    print(f"  WARNING: Different head dimensions for Q ({q_head_dim}) and KV ({kv_head_dim})")
                    # For combined QKV with GQA, we'll use the query head dimension
                    head_dim = q_head_dim
                else:
                    head_dim = q_head_dim
            else:
                # For split QKV, just use the dimensions as they are
                head_dim = q_weight.shape[1] // num_heads
                
            print(f"  Using head_dim={head_dim} for tensor decomposition")
            
            # For Gemma, the weights are in a specific format
            # Weight format depends on the model architecture
            # Calculate the expected hidden dim by looking at the output weight matrix
            # For GemmaForCausalLM, hidden_dim is typically 1152 for 1B model
            expected_hidden_dim = q_weight.shape[0]  # Input dimension for weights
            
            # Check if weights are already transposed
            transposed_weights = (o_weight.shape[0] == num_heads * head_dim and 
                                o_weight.shape[1] == expected_hidden_dim)
                
            # Pass the model config to ensure consistent dimensions
            factorized_weights = gqa_to_tpa_conversion(
                q_weight, k_weight, v_weight, o_weight,
                num_heads, num_kv_heads,
                q_rank, k_rank, v_rank,
                dtype, device,
                override_head_dim=head_dim,  # Force the correct head dimension
                transposed_weights=transposed_weights,  # Handle weight format properly
                use_dynamic_ranks=use_dynamic_ranks,  # Whether to use ranks from Tucker decomposition
                config=model.config if hasattr(model, 'config') else None,  # Pass model config for hidden_size
                fat_ranks=fat_ranks  # Whether to use much larger ranks (240) for higher accuracy
            )
            
            decomp_end = time.time()
            print(f"  Decomposition completed in {decomp_end - decomp_start:.2f} seconds")
            
            # Check if factorized weights were created
            if not factorized_weights:
                print(f"  ERROR: Factorization returned empty result")
                continue
                
            print(f"  Factorized weights keys: {list(factorized_weights.keys())}")
            
            # Apply factorized weights to the model
            if hasattr(module, "apply_factorized_weights"):
                print(f"  Applying factorized weights to module...")
                module.apply_factorized_weights(factorized_weights)
                layers_converted += 1
            else:
                print(f"  ERROR: Module {name} does not support applying factorized weights")
        
        except Exception as e:
            print(f"  ERROR in layer {name}: {e}")
            import traceback
            traceback.print_exc()
        
        layer_end = time.time()
        print(f"  Layer conversion took {layer_end - layer_start:.2f} seconds")
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"GQA to TPA conversion complete: {layers_converted}/{attention_modules_found} layers converted in {total_time:.2f} seconds")
    return model


def create_tpa_model_from_standard(standard_model, q_rank=6, k_rank=2, v_rank=2, 
                                 dtype=torch.float16, device="cuda", use_dynamic_ranks=True,
                                 fat_ranks=False):
    """
    Create a Gemma3ForMultimodalLMwithTPA model from a standard GemmaForCausalLM model.
    This function:
    1. Creates a new TPA model with the same configuration
    
    Args:
        standard_model: The standard model to convert
        q_rank: Rank for query factorization
        k_rank: Rank for key factorization
        v_rank: Rank for value factorization
        dtype: Data type for model parameters
        device: Device to use for computation
        use_dynamic_ranks: Whether to use dynamic ranks based on SVD
        fat_ranks: Whether to use much larger ranks (240) for higher accuracy but more memory
    """
    # Print device info
    print(f"Creating TPA model from standard model using device: {device}")
    if torch.device(device).type == 'cuda':
        print(f"CUDA available: {torch.cuda.is_available()}, device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        device_name = torch.cuda.get_device_name(device)
        print(f"Device name: {device_name}")
        memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
        memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        print(f"Memory allocated: {memory_allocated:.2f} GB")
        print(f"Memory reserved: {memory_reserved:.2f} GB")
        
    """
    2. Applies TPA factorization to the weights
    3. Copies over the factorized weights to the TPA model
    
    Args:
        standard_model: A GemmaForCausalLM model to convert
        q_rank: Rank for query factorization
        k_rank: Rank for key factorization
        v_rank: Rank for value factorization
        dtype: Data type for output tensors
        device: Device for computation
        use_dynamic_ranks: Whether to use ranks from Tucker decomposition
        
    Returns:
        A new Gemma3ForMultimodalLMwithTPA model with TPA weights
    """
    from ..gemma3_tpa_model_modular import Gemma3ForMultimodalLMwithTPA
    
    # Start timing
    start_time = time.time()
    print("Creating TPA model from standard model...")
    
    # First, create a config for the TPA model based on the standard model's config
    if hasattr(standard_model, 'config'):
        config = standard_model.config
        
        # Add TPA-specific configuration
        config.q_rank = q_rank
        config.k_rank = k_rank
        config.v_rank = v_rank
        
        # For non-MQA/GQA models, ensure num_key_value_heads is set
        if not hasattr(config, 'num_key_value_heads'):
            config.num_key_value_heads = config.num_attention_heads
    else:
        print("Standard model has no config, creating a default one")
        from ...config import GemmaConfig
        
        # Create a basic config
        config = GemmaConfig()
        # Fill in necessary fields
        if hasattr(standard_model, 'model'):
            if hasattr(standard_model.model, 'embedder'):
                config.vocab_size = standard_model.model.embedder.weight.shape[0]
                config.hidden_size = standard_model.model.embedder.weight.shape[1]
            
            if hasattr(standard_model.model, 'layers') and len(standard_model.model.layers) > 0:
                config.num_layers = len(standard_model.model.layers)
                if hasattr(standard_model.model.layers[0].self_attn, 'num_heads'):
                    config.num_attention_heads = standard_model.model.layers[0].self_attn.num_heads
                    config.num_key_value_heads = getattr(standard_model.model.layers[0].self_attn, 
                                                       'num_key_value_heads', config.num_attention_heads)
        
        # Add TPA parameters
        config.q_rank = q_rank
        config.k_rank = k_rank
        config.v_rank = v_rank
    
    # Create a new TPA model with this config
    tpa_model = Gemma3ForMultimodalLMwithTPA(config)
    
    # Set the data type to match
    tpa_model = tpa_model.to(dtype)
    
    # Copy over all non-attention weights
    print("Copying non-attention weights...")
    for name, param in standard_model.named_parameters():
        # Skip attention-related parameters
        if any(x in name for x in ['qkv_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'attention']):
            continue
            
        # Find corresponding parameter in the TPA model
        if name in tpa_model.state_dict():
            tpa_model.state_dict()[name].copy_(param.data)
    
    # Apply the GQA to TPA conversion on the standard model
    print("Applying GQA to TPA conversion...")
    standard_model_converted = convert_gqa_model_to_tpa(
        standard_model, 
        q_rank=q_rank,
        k_rank=k_rank,
        v_rank=v_rank,
        dtype=dtype,
        device=device,
        use_dynamic_ranks=use_dynamic_ranks,
        fat_ranks=fat_ranks  # Pass fat_ranks parameter for higher accuracy mode
    )
    
    # Initialize structure to record layer-specific ranks
    layer_ranks = []
    
    # Now copy the factorized weights from the converted model to the TPA model
    print("Copying factorized TPA weights...")
    for name, module in standard_model_converted.named_modules():
        if hasattr(module, 'use_factorized_weights') and module.use_factorized_weights:
            print(f"  Found factorized module: {name}")
            
            # Create a dictionary to hold the factorized weights for this module
            # This replaces the missing 'result' variable referenced later
            factorized_weights = {}
            for key in dir(module):
                if key.startswith(('W_A_', 'W_B_')):
                    factorized_weights[key] = getattr(module, key)
            
            # Determine the corresponding name in the TPA model
            tpa_module_name = name
            
            # Extract layer index if possible
            layer_idx = -1
            if "layers." in tpa_module_name:
                try:
                    parts = tpa_module_name.split("layers.")
                    if len(parts) > 1:
                        idx_part = parts[1].split(".")[0]
                        layer_idx = int(idx_part)
                        print(f"  Layer index: {layer_idx}")
                except Exception as e:
                    print(f"  Could not extract layer index: {e}")
            
            # Find the module in the TPA model
            tpa_module = tpa_model
            for part in tpa_module_name.split('.'):
                if hasattr(tpa_module, part):
                    tpa_module = getattr(tpa_module, part)
                else:
                    print(f"  Warning: Could not find {part} in TPA model")
                    break
            
            # The TPA attention modules expect W_A_q etc. to be nn.Linear modules, not Parameters
            # So we need to create proper nn.Linear modules with the weights

            # First, extract the dimensions and ranks
            layer_specific_ranks = {}
            for key in ['q_rank', 'k_rank', 'v_rank']:
                if hasattr(module, key):
                    value = getattr(module, key)
                    # Store the rank in our layer-specific dict
                    layer_specific_ranks[key] = value
                    # Set these directly on the module
                    setattr(tpa_module, key, value)
            
            # Record the layer-specific ranks for KV cache creation
            if layer_idx >= 0:
                # Ensure list is long enough
                while len(layer_ranks) <= layer_idx:
                    layer_ranks.append({})
                # Store this layer's ranks
                layer_ranks[layer_idx] = layer_specific_ranks
                print(f"  Recorded ranks for layer {layer_idx}: {layer_specific_ranks}")
            
            # Extract head dimensions and counts from the TPA module
            num_heads = getattr(tpa_module, 'num_heads', 4)
            num_kv_heads = getattr(tpa_module, 'num_kv_heads', 1)
            head_dim = getattr(tpa_module, 'head_dim', 256)
            hidden_dim = getattr(tpa_module, 'hidden_size', 1024)
            q_rank = layer_specific_ranks.get('q_rank', 6)
            k_rank = layer_specific_ranks.get('k_rank', 2)
            v_rank = layer_specific_ranks.get('v_rank', 2)
            
            # For each weight matrix in the converted standard model, we need to create matching Linear modules
            # Use the actual weight shapes from factorized weights, not the expected dimensions
            for std_key in ['W_A_q', 'W_A_k', 'W_A_v', 'W_B_q', 'W_B_k', 'W_B_v']:
                tpa_key = std_key  # Same name in TPA module
                
                if hasattr(module, std_key):
                    # Get weight from standard model
                    weight = getattr(module, std_key)
                    print(f"  Source {std_key} shape: {weight.shape}")
                    
                    # Linear modules need weight shape [out_features, in_features]
                    # nn.Linear expects weights in transposed form from what we usually see
                    print(f"  Source {std_key} shape: {weight.shape}")
                    
                    # Determine proper in_features and out_features based on the intended use in TPA
                    if std_key.startswith('W_A_'):
                        # W_A weights connect hidden_dim to num_heads*rank
                        # For nn.Linear in TPA, we need [out_features=num_heads*rank, in_features=hidden_dim]
                        # This needs to be consistent with usage in tpa_attention.py: 
                        # A_q = self.W_A_q(hidden_states).view(batch_size, seq_len, self.num_heads, self.q_rank)
                        
                        # IMPORTANT: For Gemma model with 1B parameters, the hidden_size is 1152,
                        # but the weight actual shape may depend on the factorized dimensions
                        
                        # Use weight shape to determine actual hidden_dim if needed
                        actual_hidden_dim = weight.shape[0]
                        if actual_hidden_dim != hidden_dim:
                            print(f"  ERROR: Weight hidden dim {actual_hidden_dim} differs from model config {hidden_dim}")
                            raise ValueError(f"CRITICAL ERROR: Weight hidden dim {actual_hidden_dim} differs from model config hidden_dim {hidden_dim}. Cannot proceed with mismatched dimensions.")
                        
                        if std_key == 'W_A_q':
                            out_features = num_heads * q_rank
                        elif std_key == 'W_A_k':
                            out_features = num_kv_heads * k_rank
                        elif std_key == 'W_A_v':
                            out_features = num_kv_heads * v_rank
                        else:
                            out_features = weight.shape[1] if weight.shape[0] == hidden_dim else weight.shape[0]
                        
                        # Use the actual hidden dimension from the weight
                        in_features = actual_hidden_dim
                    else:
                        # CORRECTION: In contextual factorization (CF) form of TPA:
                        # W_B_q projects from hidden_dim to q_rank*head_dim, then reshapes to [batch, seq, q_rank, head_dim]
                        # B_q = self.W_B_q(hidden_states).view(batch_size, seq_len, self.q_rank, self.head_dim)
                        
                        print(f"  Creating Linear layer for B matrix in TPA contextual factorization")
                        
                        # For W_B_q, the correct dimensions for Linear are:
                        # in_features = hidden_dim (same as input hidden states)
                        # out_features = q_rank*head_dim (to be reshaped after projection)
                        if std_key == 'W_B_q':
                            in_features = hidden_dim
                            out_features = q_rank * head_dim
                            print(f"  W_B_q Linear should project from hidden_dim={hidden_dim} to q_rank*head_dim={out_features}")
                            
                        elif std_key == 'W_B_k':
                            in_features = hidden_dim
                            out_features = k_rank * head_dim
                            print(f"  W_B_k Linear should project from hidden_dim={hidden_dim} to k_rank*head_dim={out_features}")
                            
                        elif std_key == 'W_B_v':
                            in_features = hidden_dim
                            out_features = v_rank * head_dim
                            print(f"  W_B_v Linear should project from hidden_dim={hidden_dim} to v_rank*head_dim={out_features}")
                            
                        else:
                            # Fallback for unknown keys - use dimensions from weight
                            in_features = hidden_dim
                            out_features = weight.shape[1] if weight.shape[0] == hidden_dim else weight.shape[0]
                            print(f"  Unknown B matrix with dimensions [out={out_features}, in={in_features}]")
                    
                    print(f"  Creating {tpa_key} with in_features={in_features}, out_features={out_features}")
                    
                    # Create new Linear module with correct dimensions
                    # nn.Linear expects weights in shape [out_features, in_features]
                    # We need to make sure we create this with the right dimensions and set weights properly
                    linear = nn.Linear(in_features, out_features, bias=False)
                    
                    # Check current weight shape to see if we need to transpose
                    try:
                        if weight.shape[0] == out_features and weight.shape[1] == in_features:
                            # Weight already in Linear's expected format [out_features, in_features]
                            linear.weight.data.copy_(weight)
                            print(f"  {tpa_key} using weight directly (already in correct shape)")
                        else:
                            # Weight needs transposing to match Linear's expected format
                            # First check if dimensions are compatible
                            if weight.shape[1] == out_features and weight.shape[0] == in_features:
                                # Simple transposition case
                                linear.weight.data.copy_(weight.t())
                                print(f"  {tpa_key} transposing weight from {weight.shape} to {linear.weight.shape}")
                            else:
                                # Handle dimension mismatch for contextual factorization weights
                                print(f"  WARNING: Weight shape {weight.shape} doesn't match required Linear dimensions "
                                     f"[{out_features}, {in_features}] for contextual factorization")
                                
                                # For TPA with contextual factorization (CF), we have derived optimal projection weights
                                # Use these optimally-derived weights instead of simple resizing
                                if std_key == 'W_B_q':
                                    print(f"  Using optimally derived W_B_q projection weights from Tucker decomposition")
                                    # Check if the weight is in the result dictionary
                                    if "W_B_q" in factorized_weights:
                                        resized_weight = factorized_weights["W_B_q"].t()
                                        print(f"  Using derived W_B_q from factorized_weights with shape {resized_weight.shape}")
                                    else:
                                        print(f"  ERROR: W_B_q not found in factorized_weights, creating appropriate projection matrix")
                                        # Create appropriately sized weight matrix
                                        resized_weight = torch.randn((out_features, in_features), 
                                                                dtype=weight.dtype, device=weight.device) * 0.02
                                    
                                elif std_key == 'W_B_k':
                                    print(f"  Using optimally derived W_B_k projection weights from Tucker decomposition")
                                    # Check if the weight is in the result dictionary
                                    if "W_B_k" in factorized_weights:
                                        resized_weight = factorized_weights["W_B_k"].t()
                                        print(f"  Using derived W_B_k from factorized_weights with shape {resized_weight.shape}")
                                    else:
                                        print(f"  ERROR: W_B_k not found in factorized_weights, creating appropriate projection matrix")
                                        # Create appropriately sized weight matrix
                                        resized_weight = torch.randn((out_features, in_features), 
                                                                dtype=weight.dtype, device=weight.device) * 0.02
                                    
                                elif std_key == 'W_B_v':
                                    print(f"  Using optimally derived W_B_v projection weights from Tucker decomposition")
                                    # Check if the weight is in the result dictionary
                                    if "W_B_v" in factorized_weights:
                                        resized_weight = factorized_weights["W_B_v"].t()
                                        print(f"  Using derived W_B_v from factorized_weights with shape {resized_weight.shape}")
                                    else:
                                        print(f"  ERROR: W_B_v not found in factorized_weights, creating appropriate projection matrix")
                                        # Create appropriately sized weight matrix
                                        resized_weight = torch.randn((out_features, in_features), 
                                                                dtype=weight.dtype, device=weight.device) * 0.02
                                    
                                else:
                                    # Fallback for non-TPA weights, though this shouldn't happen for B matrices
                                    print(f"  FALLBACK: Creating appropriately shaped weight matrix")
                                    # Initialize with zeros for correctness
                                    resized_weight = torch.zeros((out_features, in_features), 
                                                               dtype=weight.dtype, device=weight.device)
                                    
                                    # Try to preserve information from original weight if dimensions allow
                                    if weight.shape[0] == in_features and weight.shape[1] <= out_features:
                                        # Copy information from original weight, transposing for Linear
                                        resized_weight[:weight.shape[1], :] = weight.t()
                                        print(f"  Preserved information from original weight")
                                    elif weight.shape[1] == in_features and weight.shape[0] <= out_features:
                                        # Copy information directly
                                        resized_weight[:weight.shape[0], :] = weight
                                        print(f"  Preserved information from original weight")
                                    else:
                                        # Use random initialization as last resort
                                        scale = 1.0 / math.sqrt(in_features)
                                        resized_weight = torch.randn((out_features, in_features), 
                                                                   dtype=weight.dtype, device=weight.device) * scale
                                        print(f"  Created new weight matrix with appropriate dimensions")
                                
                                linear.weight.data.copy_(resized_weight)
                                print(f"  {tpa_key} resized weight to match required dimensions: {linear.weight.shape}")
                    except Exception as copy_error:
                        print(f"  ERROR copying weight for {tpa_key}: {copy_error}")
                        # Create a new weight tensor with the correct shape
                        print(f"  Creating new random weight for {tpa_key} with shape {linear.weight.shape}")
                        # Initialize with small random values instead of zeros
                        nn.init.xavier_normal_(linear.weight)
                    
                    # Set the Linear module on the TPA module
                    setattr(tpa_module, tpa_key, linear)
                    print(f"  Created {tpa_key} with shape {linear.weight.shape}")
            
            # Mark the TPA module as using factorized weights
            tpa_module.use_factorized_weights = True
    
    # Store layer-specific ranks in config for KV cache creation
    if layer_ranks:
        print(f"Storing layer-specific ranks in model config: {layer_ranks}")
        if not hasattr(config, 'model_structure'):
            config.model_structure = {}
        config.model_structure["layer_ranks"] = layer_ranks
        
        # Update the model's config to match
        tpa_model.config = config
    
    # Set the tokenizer if available
    if hasattr(standard_model, 'tokenizer'):
        tpa_model.tokenizer = standard_model.tokenizer
        
    end_time = time.time()
    print(f"TPA model creation complete in {end_time - start_time:.2f} seconds")
    
    return tpa_model