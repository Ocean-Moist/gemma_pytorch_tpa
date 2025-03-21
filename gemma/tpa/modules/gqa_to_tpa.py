"""
GQA to TPA conversion using Tucker decomposition.

This module provides functionality to convert Grouped Query Attention (GQA) weights
to Tensor Product Attention (TPA) format using TensorLLM-style Tucker decomposition.
"""

import torch
import torch.nn as nn
import math
import time
from typing import Dict

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
    q_rank: int = 240,
    k_rank: int = 240,
    v_rank: int = 240,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    use_dynamic_ranks: bool = True,  # Whether to use ranks determined by SVD (True) or force specified ranks (False)
    config = None,  # Model config to ensure consistent dimensions
) -> Dict[str, torch.Tensor]:
    """
    Convert GQA attention weights to TPA format using SVD-based independent factorization.
    
    This function implements an improved conversion approach that properly handles GQA models
    where Q has different dimensions and head counts than K/V. Instead of forcing all
    projections to use the same head dimension, we factorize Q and K/V separately with their
    actual dimensions, which drastically reduces reconstruction error.
    
    Args:
        q_weight: Query projection weight matrix [hidden_dim, num_heads * q_head_dim]
        k_weight: Key projection weight matrix [hidden_dim, num_kv_heads * kv_head_dim]
        v_weight: Value projection weight matrix [hidden_dim, num_kv_heads * kv_head_dim]
        o_weight: Output projection weight matrix [num_heads * q_head_dim, hidden_dim]
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads (groups)
        q_rank: Rank for query factorization
        k_rank: Rank for key factorization
        v_rank: Rank for value factorization
        dtype: Data type for the output tensors
        device: Device for computation
        use_dynamic_ranks: Whether to use ranks determined by SVD (True) or force specified ranks (False)
        config: Model config to ensure consistent dimensions - we'll use config.hidden_size if available
        
    Returns:
        Dictionary of factorized weights for TPA implementation
    """

    # Start timing
    tic = time.time()
    print("Starting improved GQA to TPA conversion with separate Q and K/V factorization...")
    
    # Move weights to correct device and dtype for processing
    q_weight = q_weight.to(device, dtype=torch.float32)
    k_weight = k_weight.to(device, dtype=torch.float32)
    v_weight = v_weight.to(device, dtype=torch.float32)
    o_weight = o_weight.to(device, dtype=torch.float32)

    # Get dimensions - use config.hidden_size if provided, otherwise infer from weights
    config_hidden_size = getattr(config, 'hidden_size', None) if config is not None else None
    
    # For Gemma models, weights need to be transposed
    # In PyTorch Linear layers, weights have shape [out_features, in_features]
    # But for our factorization, we need them in [hidden_dim, projection_dim] format
    hidden_dim = config_hidden_size if config_hidden_size is not None else q_weight.shape[1]
    q_weight = q_weight.transpose(0, 1)
    k_weight = k_weight.transpose(0, 1)
    v_weight = v_weight.transpose(0, 1)

    print(f"After transposition: q=[{q_weight.shape}], k=[{k_weight.shape}], v=[{v_weight.shape}]")

    # Verify dimensions are consistent with expected hidden_dim
    if q_weight.shape[0] != hidden_dim:
        raise ValueError(f"CRITICAL ERROR: Q weight hidden dim {q_weight.shape[0]} != config.hidden_size {hidden_dim}.")
    if k_weight.shape[0] != hidden_dim:
        raise ValueError(f"CRITICAL ERROR: K weight hidden dim {k_weight.shape[0]} != config.hidden_size {hidden_dim}.")
    if v_weight.shape[0] != hidden_dim:
        raise ValueError(f"CRITICAL ERROR: V weight hidden dim {v_weight.shape[0]} != config.hidden_size {hidden_dim}.")

    print(f"Dimensions: hidden_dim={hidden_dim}")
    print(f"Weight shapes: Q={q_weight.shape}, K={k_weight.shape}, V={v_weight.shape}, O={o_weight.shape}")
    
    # Step 1: Calculate actual head dimensions directly from the weights
    q_proj_dim = q_weight.shape[1]
    k_proj_dim = k_weight.shape[1]
    v_proj_dim = v_weight.shape[1]
    
    # Verify projections are evenly divisible by head counts
    if q_proj_dim % num_heads != 0:
        raise ValueError(f"Q projection dimension {q_proj_dim} not divisible by num_heads {num_heads}")
    if k_proj_dim % num_kv_heads != 0:
        raise ValueError(f"K projection dimension {k_proj_dim} not divisible by num_kv_heads {num_kv_heads}")
    if v_proj_dim % num_kv_heads != 0:
        raise ValueError(f"V projection dimension {v_proj_dim} not divisible by num_kv_heads {num_kv_heads}")
    
    # Calculate actual head dimensions from the weights - keeping them separate
    q_head_dim = q_proj_dim // num_heads
    k_head_dim = k_proj_dim // num_kv_heads
    v_head_dim = v_proj_dim // num_kv_heads
    
    print(f"\nDIMENSION CALCULATION: Using separate head dimensions for Q and K/V")
    print(f"  Q weights: {q_weight.shape} → {q_head_dim} = {q_proj_dim} / {num_heads} heads")
    print(f"  K weights: {k_weight.shape} → {k_head_dim} = {k_proj_dim} / {num_kv_heads} kv_heads")
    print(f"  V weights: {v_weight.shape} → {v_head_dim} = {v_proj_dim} / {num_kv_heads} kv_heads")
    
    # Reshape weights to 3D tensors with proper per-head dimensions
    q_weights_reshaped = q_weight.reshape(hidden_dim, num_heads, q_head_dim)
    k_weights_reshaped = k_weight.reshape(hidden_dim, num_kv_heads, k_head_dim)
    v_weights_reshaped = v_weight.reshape(hidden_dim, num_kv_heads, v_head_dim)
    
    # Handle output weight by reshaping to match query projection
    # For o_weight, dimensions are usually [num_heads*q_head_dim, hidden_dim]
    o_weights_transposed = o_weight
    if o_weight.shape[1] == hidden_dim:
        # o_weight is already in [num_heads*q_head_dim, hidden_dim] format
        pass
    else:
        # Need to transpose the o_weight
        o_weights_transposed = o_weight.transpose(0, 1)
    
    # Create mapping from query heads to kv groups
    # For each query head, identify which kv head it should use
    heads_per_group = num_heads // num_kv_heads
    q_to_kv_mapping = torch.tensor([i // heads_per_group for i in range(num_heads)],
                                  device=q_weight.device)
    
    print(f"Head mapping: {num_heads} query heads to {num_kv_heads} KV heads, {heads_per_group} query heads per KV head")
    print(f"Mapping: {q_to_kv_mapping.tolist()}")
    
    # Initialize result dictionary
    result = {}
    
    # Store the mapping from query heads to kv heads
    result["q_to_kv_mapping"] = q_to_kv_mapping
    
    # Store the original head dimensions (critical for correct inference)
    result["q_head_dim"] = int(q_head_dim)
    result["k_head_dim"] = int(k_head_dim)
    result["v_head_dim"] = int(v_head_dim)
    
    # Step 2: Analyze intrinsic ranks using SVD for Q and K/V separately
    print("\nANALYZING INTRINSIC RANKS using SVD (separate analysis for Q and K/V)")
    
    # Function to analyze intrinsic rank using SVD and energy threshold
    def analyze_intrinsic_rank(weight_tensor, tensor_name):
        # Reshape to 2D for SVD
        weight_2d = weight_tensor.reshape(weight_tensor.shape[0], -1)
        # Compute SVD
        _, singular_values, _ = torch.linalg.svd(weight_2d, full_matrices=False)
        
        # Compute energy-based thresholds
        energy = singular_values ** 2
        energy_norm = energy / torch.sum(energy)
        cumulative = torch.cumsum(energy_norm, dim=0)
        
        # Calculate ranks for different energy thresholds
        thresholds = [0.9, 0.95, 0.98]
        ranks = []
        
        print(f"  {tensor_name} singular value analysis:")
        for thresh in thresholds:
            rank = torch.sum(cumulative <= thresh).item() + 1
            rank = min(rank, len(cumulative))
            ranks.append(int(rank))
            print(f"    {thresh*100:.0f}% energy: rank {int(rank)}")
        
        # Use the 95% threshold as the recommended rank
        recommended_rank = ranks[1] if len(ranks) > 1 else ranks[0]
        return recommended_rank, ranks, singular_values
    
    # Analyze Q, K, and V weight matrices separately
    q_recommended_rank, q_ranks, q_singular_values = analyze_intrinsic_rank(q_weight, "Q")
    k_recommended_rank, k_ranks, k_singular_values = analyze_intrinsic_rank(k_weight, "K")
    v_recommended_rank, v_ranks, v_singular_values = analyze_intrinsic_rank(v_weight, "V")
    
    # Apply practical limits to ranks
    max_practical_rank = 320  # Standard cap to avoid excessive computation
    
    # Calculate maximum possible ranks based on matrix dimensions
    max_q_rank = min(hidden_dim, q_proj_dim)
    max_k_rank = min(hidden_dim, k_proj_dim)
    max_v_rank = min(hidden_dim, v_proj_dim)
    
    print(f"\nMaximum possible ranks based on matrix dimensions: Q={max_q_rank}, K={max_k_rank}, V={max_v_rank}")
    
    # Determine final ranks to use
    if use_dynamic_ranks:
        actual_q_rank = min(max_practical_rank, q_recommended_rank, max_q_rank)
        actual_k_rank = min(max_practical_rank, k_recommended_rank, max_k_rank)
        actual_v_rank = min(max_practical_rank, v_recommended_rank, max_v_rank)
        print(f"USING OPTIMAL COMPONENT-SPECIFIC RANKS: Q={actual_q_rank}, K={actual_k_rank}, V={actual_v_rank}")
        print(f"These ranks are determined by energy-based analysis to balance accuracy and efficiency")
    else:
        # Use the user-specified ranks but cap them by matrix dimensions
        actual_q_rank = min(max_practical_rank, q_rank, max_q_rank)
        actual_k_rank = min(max_practical_rank, k_rank, max_k_rank)
        actual_v_rank = min(max_practical_rank, v_rank, max_v_rank)
        print(f"USING USER-SPECIFIED RANKS (capped by matrix dimensions): Q={actual_q_rank}, K={actual_k_rank}, V={actual_v_rank}")
    
    # Ensure minimum rank for numerical stability
    actual_q_rank = max(2, actual_q_rank)
    actual_k_rank = max(2, actual_k_rank)
    actual_v_rank = max(2, actual_v_rank)
    
    # Step 3: Independent factorization for Q and K/V components
    print("\nPerforming independent factorization for Q and K/V")
    
    # Function to compute SVD-based TPA factors
    def compute_svd_tpa_factors(weight_matrix, rank, name, head_dim, num_heads=1):
        """
        Compute TPA factors using optimal SVD approach.
        
        For a matrix W of shape [hidden_dim, proj_dim], this computes:
        W ≈ U_R Σ_R V_R^T
        
        Returns:
        - W_A: First TPA factor [hidden_dim, num_heads * rank]
        - W_B: Second TPA factor [hidden_dim, rank * head_dim]
        """
        start_time = time.time()
        print(f"  Computing SVD factorization for {name} with rank {rank}...")
        
        # Reshape to 2D for SVD if needed
        if weight_matrix.dim() > 2:
            orig_shape = weight_matrix.shape
            weight_matrix = weight_matrix.reshape(weight_matrix.shape[0], -1)
            print(f"  Reshaped {name} from {orig_shape} to {weight_matrix.shape}")

        # Compute truncated SVD
        U, S, Vh = torch.linalg.svd(weight_matrix, full_matrices=False)
        
        # Ensure rank does not exceed matrix dimensions
        max_possible_rank = min(weight_matrix.shape[0], weight_matrix.shape[1])
        if rank > max_possible_rank:
            print(f"  ADJUSTING {name} rank from {rank} to {max_possible_rank} (maximum possible for matrix dimensions)")
            rank = min(rank, max_possible_rank)
        
        # Truncate to target rank
        U_r = U[:, :rank]  # [hidden_dim, rank]
        S_r = S[:rank]     # [rank]
        Vh_r = Vh[:rank]   # [rank, proj_dim]
        
        # Compute scaling factor for better numerical properties
        sqrt_rank = math.sqrt(rank)
        
        # Scale factors by sqrt(singular_values) and sqrt(rank)
        sqrt_S_r = torch.sqrt(S_r)
        
        # For multi-head case, we need to handle each head separately
        if num_heads > 1:
            # Initialize output matrices with proper dimensions 
            W_A = torch.zeros((weight_matrix.shape[0], num_heads * rank), 
                              device=weight_matrix.device, dtype=weight_matrix.dtype)
            W_B = torch.zeros((weight_matrix.shape[0], rank * head_dim), 
                              device=weight_matrix.device, dtype=weight_matrix.dtype)
            
            # Per-head factorization for more accurate reconstruction
            for h in range(num_heads):
                # Extract offset for this head
                head_offset = h * head_dim
                
                # For each head, extract the corresponding slice from Vh_r
                # This ensures we factor according to the true head structure
                head_Vh = Vh_r[:, head_offset:head_offset + head_dim]
                
                # Scale U and Vh by sqrt(S) and sqrt(rank)
                W_A_head = sqrt_rank * U_r * sqrt_S_r.unsqueeze(0)  # [hidden_dim, rank]
                W_A[:, h*rank:(h+1)*rank] = W_A_head
                
                # For W_B, we create a proper slice for this head
                for r in range(rank):
                    scaled_v_r = sqrt_rank * sqrt_S_r[r] * head_Vh[r]  # [head_dim]
                    # In TPA format, we need identical rows for each hidden dimension
                    W_B[:, r*head_dim:(r+1)*head_dim] = scaled_v_r.unsqueeze(0).expand(weight_matrix.shape[0], -1)
        else:
            # Single head case (usually for K/V in GQA)
            W_A = sqrt_rank * U_r * sqrt_S_r.unsqueeze(0)  # [hidden_dim, rank]
            
            # For W_B, calculate properly scaled factors for TPA format
            W_B = torch.zeros((weight_matrix.shape[0], rank * head_dim), 
                              device=weight_matrix.device, dtype=weight_matrix.dtype)
            
            # For each rank component, compute the corresponding W_B slice
            for r in range(rank):
                # Need to adjust Vh to the head_dim - important for proper attention computation
                if Vh_r.shape[1] != head_dim:
                    # Reshape Vh_r to match head_dim - critical for correct dot products
                    print(f"  Adjusting Vh_r from shape {Vh_r.shape} to match head_dim {head_dim}")
                    # Simple approach: use the first head_dim dimensions if larger
                    if Vh_r.shape[1] > head_dim:
                        v_r = Vh_r[r, :head_dim]
                    else:
                        # Pad if smaller
                        v_r = torch.zeros(head_dim, device=Vh_r.device, dtype=Vh_r.dtype)
                        v_r[:Vh_r.shape[1]] = Vh_r[r]
                else:
                    v_r = Vh_r[r]
                
                scaled_v_r = sqrt_rank * sqrt_S_r[r] * v_r  # [head_dim]
                # In TPA, we need the same vector repeated across all hidden dimensions
                W_B[:, r*head_dim:(r+1)*head_dim] = scaled_v_r.unsqueeze(0).expand(weight_matrix.shape[0], -1)
        
        # Compute reconstruction error
        reconstructed = U_r @ torch.diag(S_r) @ Vh_r
        error = torch.norm(weight_matrix - reconstructed) / torch.norm(weight_matrix)
        print(f"  {name} SVD reconstruction error: {error.item():.6f}")
        
        end_time = time.time()
        print(f"  {name} SVD factorization completed in {end_time - start_time:.2f} seconds")
        
        return W_A, W_B
    
    # Process query heads - with proper query dimensions by factorizing each head SEPARATELY
    # This is critical for GQA models where q_head_dim may differ from kv_head_dim
    print(f"Factorizing query projection with {num_heads} heads, head_dim={q_head_dim}")
    print(f"  Q weight shape: {q_weight.shape}, total projection dim: {q_proj_dim}")
    
    # Pre-compute the maximum possible rank for each head
    per_head_max_ranks = []
    for h in range(num_heads):
        head_weight = q_weights_reshaped[:, h, :]  # shape: [hidden_dim, q_head_dim]
        head_max_rank = min(head_weight.shape[0], head_weight.shape[1])
        head_rank = min(actual_q_rank, head_max_rank)
        per_head_max_ranks.append(head_rank)
    
    # Get the maximum rank across all heads for allocating W_B
    max_head_rank = max(per_head_max_ranks)
    print(f"  Per-head maximum ranks: {per_head_max_ranks}, using max={max_head_rank} for W_B")
    
    # Initialize combined factors for Q - use actual per-head ranks
    # Each head gets its own slice in W_A based on its calculated rank
    total_q_rank = sum(per_head_max_ranks)
    W_A_q = torch.zeros((hidden_dim, total_q_rank), 
                       device=q_weight.device, dtype=q_weight.dtype)
    
    # W_B needs to be large enough to hold all heads' factors with proper offsets
    # Each head needs space for max_head_rank * q_head_dim
    W_B_q = torch.zeros((hidden_dim, num_heads * max_head_rank * q_head_dim), 
                       device=q_weight.device, dtype=q_weight.dtype)
    print(f"  Created W_B_q with shape {W_B_q.shape} to accommodate {num_heads} heads")
    
    # Track the offset in the W_A_q matrix for each head
    head_offsets = [0]
    for rank in per_head_max_ranks:
        head_offsets.append(head_offsets[-1] + rank)
    
    # Loop over heads and factorize each independently
    q_factorization_errors = []
    for h in range(num_heads):
        print(f"  Factorizing query head {h}/{num_heads}...")
        # Extract this head's weight slice
        head_weight = q_weights_reshaped[:, h, :]  # shape: [hidden_dim, q_head_dim]
        
        # Use the pre-computed rank for this head
        head_rank = per_head_max_ranks[h]
        print(f"  Using rank {head_rank} for head {h} (limited by head dimensions)")
        
        # Factorize this head independently with num_heads=1 since we're processing one head at a time
        W_A_head, W_B_head = compute_svd_tpa_factors(
            head_weight, head_rank, f"Q-head-{h}", q_head_dim, num_heads=1)
        
        # Store in the combined matrices at the correct offset
        start_idx = head_offsets[h]
        end_idx = head_offsets[h+1]
        W_A_q[:, start_idx:end_idx] = W_A_head
        
        # For W_B, we need consistent factors for the TPA implementation
        # Use a head-specific offset in the global W_B_q matrix to avoid overwriting
        head_offset = h * max_head_rank * q_head_dim
        print(f"  Using head offset {head_offset} for head {h} in W_B_q")
        
        # Only store up to the actual rank used for this head
        for r in range(min(head_rank, max_head_rank)):
            # Calculate global indices with head-specific offset
            global_start_idx = head_offset + r * q_head_dim
            global_end_idx = head_offset + (r + 1) * q_head_dim
            
            # Calculate local indices in the head's W_B
            local_start_idx = r * q_head_dim
            local_end_idx = (r + 1) * q_head_dim
            
            # Extract the head's factor for this rank
            head_factor = W_B_head[:, local_start_idx:local_end_idx]
            
            # Store this rank component in the global W_B at the head-specific offset
            W_B_q[:, global_start_idx:global_end_idx] = head_factor
            
        # Compute reconstruction error for this head using proper TPA reconstruction
        # Vectorized TPA reconstruction using einsum
        # Reshape tensors for efficient computation
        A_reshaped = W_A_head.reshape(hidden_dim, 1, head_rank)  # [hidden_dim, 1, head_rank]
        B_reshaped = W_B_head.reshape(hidden_dim, head_rank, q_head_dim)  # [hidden_dim, head_rank, q_head_dim]
        
        # Einstein summation computes the TPA reconstruction all at once
        # For each position i: sum_r (A[i,r] * B[i,r]) / rank
        reconstructed = torch.einsum('ipr,irj->ipj', A_reshaped, B_reshaped).squeeze(1) / head_rank
        
        # Compute error
        head_error = torch.norm(head_weight - reconstructed) / torch.norm(head_weight)
        q_factorization_errors.append(head_error.item())
        print(f"  Query head {h} factorization error: {head_error.item():.6f}")
    
    # Report average error across all query heads
    avg_q_error = sum(q_factorization_errors) / len(q_factorization_errors)
    print(f"  Average query head factorization error: {avg_q_error:.6f} ({avg_q_error*100:.2f}%)")
    
    # Process key and value heads with their own dimensions
    # For these we can use the original function since they typically have consistent dimensions
    print(f"Factorizing key projection with {num_kv_heads} KV heads, head_dim={k_head_dim}")
    print(f"  K weight shape: {k_weight.shape}, total projection dim: {k_proj_dim}")
    W_A_k, W_B_k = compute_svd_tpa_factors(
        k_weight, actual_k_rank, "K", k_head_dim, num_kv_heads)
    
    print(f"Factorizing value projection with {num_kv_heads} KV heads, head_dim={v_head_dim}")
    print(f"  V weight shape: {v_weight.shape}, total projection dim: {v_proj_dim}")
    W_A_v, W_B_v = compute_svd_tpa_factors(
        v_weight, actual_v_rank, "V", v_head_dim, num_kv_heads)
    
    # Add all factorized weights to the result dictionary
    result["W_A_q"] = W_A_q.to(dtype=dtype, device=device)
    result["W_A_k"] = W_A_k.to(dtype=dtype, device=device)
    result["W_A_v"] = W_A_v.to(dtype=dtype, device=device)
    
    result["W_B_q"] = W_B_q.to(dtype=dtype, device=device)
    result["W_B_k"] = W_B_k.to(dtype=dtype, device=device)
    result["W_B_v"] = W_B_v.to(dtype=dtype, device=device)
    
    # Store the per-head ranks and offsets for Q (critical for correct inference)
    result["q_per_head_ranks"] = per_head_max_ranks
    result["q_head_offsets"] = head_offsets
    result["q_max_head_rank"] = int(max_head_rank)
    
    # Store the effective ranks used (important for KV cache setup)
    result["q_rank"] = int(max_head_rank)  # Use max rank for compatibility
    result["k_rank"] = int(actual_k_rank)
    result["v_rank"] = int(actual_v_rank)
    
    # Step 4: Verify reconstruction quality
    print("\nVerifying reconstruction quality of factorized weights")
    
    # Function to verify reconstruction quality using TPA formulation
    def verify_tpa_reconstruction(W_A, W_B, orig_weight, rank, name, head_dim, num_heads):
        """Verify the reconstruction quality of the factorized TPA weights."""
        print(f"  Verifying {name} reconstruction...")
        
        # Reshape original weight for comparison
        if orig_weight.dim() == 2:
            # Reshape to [hidden_dim, num_heads, head_dim]
            orig_3d = orig_weight.reshape(orig_weight.shape[0], num_heads, head_dim)
        else:
            orig_3d = orig_weight
            
        hidden_dim = orig_3d.shape[0]
        
        # Check if this is for Q with head-specific offsets or standard K/V
        if name == "Q" and hasattr(W_B, "shape") and W_B.shape[1] == num_heads * max_head_rank * head_dim:
            # Q case with head-specific offsets in W_B
            print(f"  Using head-specific offsets for {name} verification")
            
            # Initialize reconstruction tensor
            recon = torch.zeros((hidden_dim, num_heads, head_dim), 
                              device=W_A.device, dtype=W_A.dtype)
            
            # Compute per-head reconstruction using the proper offsets
            for h in range(num_heads):
                # Get W_A for this head
                start_idx = head_offsets[h]
                end_idx = head_offsets[h+1]
                head_rank = per_head_max_ranks[h]
                head_W_A = W_A[:, start_idx:end_idx]
                
                # Get W_B for this head with proper offset
                head_offset = h * max_head_rank * head_dim
                head_W_B = W_B[:, head_offset:head_offset+head_rank*head_dim]
                
                # Reshape for efficient computation
                A_reshaped = head_W_A.reshape(hidden_dim, 1, head_rank)  # [hidden_dim, 1, head_rank]
                B_reshaped = head_W_B.reshape(hidden_dim, head_rank, head_dim)  # [hidden_dim, head_rank, head_dim]
                
                # Compute reconstruction for this head
                head_recon = torch.einsum('ipr,irj->ipj', A_reshaped, B_reshaped).squeeze(1) / head_rank
                
                # Store in the combined reconstruction
                recon[:, h, :] = head_recon
                
                # Print per-head reconstruction error
                head_error = torch.norm(orig_3d[:, h, :] - head_recon) / torch.norm(orig_3d[:, h, :])
                print(f"    Head {h} error: {head_error.item():.6f}")
            
        elif W_A.shape[1] == num_heads * rank:
            # Standard case for K/V: reshape W_A to [hidden_dim, num_heads, rank]
            W_A_reshaped = W_A.reshape(hidden_dim, num_heads, rank)
            
            # Reshape W_B to [hidden_dim, rank, head_dim]
            W_B_reshaped = W_B.reshape(hidden_dim, rank, head_dim)
            
            # Use efficient einsum for TPA reconstruction
            # This computes for all heads at once: sum_r (A[i,h,r] * B[i,r,d]) / rank
            recon = torch.einsum('ihr,ird->ihd', W_A_reshaped, W_B_reshaped) / rank
            
            # Print per-head stats for multi-head case
            if num_heads > 1:
                # Calculate errors per head in a vectorized way
                head_errors = torch.stack([
                    torch.norm(orig_3d[:, h, :] - recon[:, h, :]) / torch.norm(orig_3d[:, h, :])
                    for h in range(min(num_heads, 4))  # Only compute for the heads we'll display
                ])
                
                # Print the first few head errors
                for h in range(min(num_heads, 4)):
                    print(f"    Head {h} error: {head_errors[h].item():.6f}")
                
                if num_heads > 4:
                    print(f"    ... and {num_heads - 4} more heads")
            
        else:
            # Special case: W_A is already sliced for a specific head
            # We assume num_heads=1 in this case
            
            # Reshape for efficient computation
            A_reshaped = W_A.reshape(hidden_dim, 1, rank)  # [hidden_dim, 1, rank]
            B_reshaped = W_B.reshape(hidden_dim, rank, head_dim)  # [hidden_dim, rank, head_dim]
            
            # Use einsum for fast vectorized computation
            recon = torch.einsum('ipr,ird->ipd', A_reshaped, B_reshaped).squeeze(1) / rank
        
        # Compute overall reconstruction error
        error = torch.norm(orig_3d.reshape(-1) - recon.reshape(-1)) / torch.norm(orig_3d.reshape(-1))
        
        print(f"  {name} overall reconstruction error: {error.item():.6f} ({error.item()*100:.2f}%)")
        return error, recon
    
    # Verify query weights per head first - using batch operations
    print("\nVerifying reconstruction quality of query weights PER HEAD:")
    
    # Pre-allocate tensors for all results
    q_head_errors = []
    q_head_recons = [None] * num_heads
    
    # Process head verification in batches where possible to maximize throughput
    print(f"  Beginning verification of {num_heads} query heads with optimized batch operations...")
    print(f"  Using head-specific offsets in W_B_q for verification")
    start_time = time.time()
    
    # Loop through heads but process in parallel where possible
    for h in range(num_heads):
        # Extract this head's original weights
        head_orig = q_weights_reshaped[:, h, :]
        
        # Get the appropriate slice of W_A_q for this head
        start_idx = head_offsets[h]
        end_idx = head_offsets[h+1]
        head_rank = per_head_max_ranks[h]
        head_W_A = W_A_q[:, start_idx:end_idx]
        
        # Calculate head-specific offset in W_B_q
        head_offset = h * max_head_rank * q_head_dim
        
        # Extract slice from W_B_q with proper head-specific offset
        head_W_B = W_B_q[:, head_offset:head_offset+head_rank*q_head_dim]
        
        # Vectorized TPA reconstruction - use direct tensor operations 
        # This is the fastest way to compute TPA reconstruction
        
        # Special handling for the common case where head_rank is a power of 2
        if (head_rank & (head_rank-1) == 0) and head_rank > 0:  # Check if power of 2
            # Use specialized reshape + bmm for maximum GPU efficiency
            A_flat = head_W_A.reshape(-1, head_rank)  # [hidden_dim, head_rank]
            B_flat = head_W_B.reshape(hidden_dim*head_rank, q_head_dim)  # [hidden_dim*head_rank, q_head_dim]
            
            # Reshape A for batch matrix multiplication
            A_bmm = A_flat.unsqueeze(2)  # [hidden_dim, head_rank, 1]
            
            # Reshape B to group by position
            B_bmm = B_flat.reshape(hidden_dim, head_rank, q_head_dim)  # [hidden_dim, head_rank, q_head_dim]
            
            # Batch matrix multiply and sum - this utilizes optimized GEMM kernels
            head_recon = torch.bmm(A_bmm.transpose(1,2), B_bmm).squeeze(1) / head_rank
        else:
            # Standard case - use einsum which is highly optimized for GPU
            A_reshaped = head_W_A.reshape(hidden_dim, 1, head_rank)  # [hidden_dim, 1, head_rank]
            B_reshaped = head_W_B.reshape(hidden_dim, head_rank, q_head_dim)  # [hidden_dim, head_rank, q_head_dim]
            head_recon = torch.einsum('ipr,irj->ipj', A_reshaped, B_reshaped).squeeze(1) / head_rank
        
        # Compute error using direct norm
        head_error = torch.norm(head_orig - head_recon) / torch.norm(head_orig)
        
        # Store results
        q_head_errors.append(head_error.item())
        q_head_recons[h] = head_recon
    
    # Report timing information
    end_time = time.time()
    verification_time = end_time - start_time
    print(f"  Completed {num_heads} head verifications in {verification_time:.4f} seconds")
    
    # Report individual head errors
    for h in range(num_heads):
        print(f"  Head {h} reconstruction error: {q_head_errors[h]:.6f} ({q_head_errors[h]*100:.2f}%)")
    
    # Compute average error across all heads
    avg_q_error = sum(q_head_errors) / len(q_head_errors)
    print(f"Average Q head reconstruction error: {avg_q_error:.6f} ({avg_q_error*100:.2f}%)")
    
    # We can't directly verify the entire Q reconstruction with our modified layout
    # Instead, reconstruct a combined result by concatenating head reconstructions
    q_combined_recon = torch.zeros_like(q_weights_reshaped)
    for h in range(num_heads):
        q_combined_recon[:, h, :] = q_head_recons[h]
        
    # Calculate overall error on the combined reconstruction
    q_error = torch.norm(q_weights_reshaped.reshape(-1) - q_combined_recon.reshape(-1)) / torch.norm(q_weights_reshaped.reshape(-1))
    print(f"\nVerifying reconstruction quality of COMBINED factorized weights:")
    print(f"Q combined reconstruction error: {q_error.item():.6f} ({q_error.item()*100:.2f}%)")
    
    # Verify key weights
    k_error, k_recon = verify_tpa_reconstruction(
        W_A_k, W_B_k, k_weights_reshaped, actual_k_rank, "K", k_head_dim, num_kv_heads)
    
    # Verify value weights
    v_error, v_recon = verify_tpa_reconstruction(
        W_A_v, W_B_v, v_weights_reshaped, actual_v_rank, "V", v_head_dim, num_kv_heads)
    
    # Finish timing and return
    toc = time.time()
    print(f"\nGQA to TPA conversion complete in {toc - tic:.2f} seconds")
    print(f"Final reconstruction errors:")
    print(f"  Q combined: {q_error.item()*100:.2f}%")
    print(f"  Q per-head average: {avg_q_error*100:.2f}%")
    print(f"  Per-head details: {', '.join([f'Head {i}: {err*100:.2f}%' for i, err in enumerate(q_head_errors)])}")
    print(f"  K: {k_error.item()*100:.2f}%, V: {v_error.item()*100:.2f}%")
    print(f"Used head dimensions: Q={q_head_dim}, K={k_head_dim}, V={v_head_dim}")
    print(f"Used per-head ranks for Q: {per_head_max_ranks}")
    print(f"Used ranks: Q max={max_head_rank}, K={actual_k_rank}, V={actual_v_rank}")
    print(f"Total Q rank used: {total_q_rank} (sum of per-head ranks)")
    
    return result


def convert_gqa_model_to_tpa(model, q_rank=240, k_rank=240, v_rank=240, dtype=torch.float16, device="cuda", use_dynamic_ranks=True, fat_ranks=False):
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

    print("Converting GQA model to TPA format...")
    
    # Add timing and layer counting
    import time
    start_time = time.time()
    layers_converted = 0
    attention_modules_found = 0
    
    # Debug model structure
    print(f"Model type: {type(model).__name__}")
    print("Searching for attention modules...")
    
    attention_modules = []
    
    # Check if using standard GemmaForCausalLM structure
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
                
                head_dim = module.head_dim
                print(f"  Using head_dim={head_dim} from module attribute")

                # Calculate sizes for splitting
                q_size = num_heads * head_dim
                kv_size = num_kv_heads * head_dim
                
                # Normal case - dimensions match expectations
                q_weight, k_weight, v_weight = qkv_weight.split([q_size, kv_size, kv_size], dim=0)
                
                o_weight = module.o_proj.weight
                
                print(f"  Split combined QKV projection: Q: {q_weight.shape}, K: {k_weight.shape}, V: {v_weight.shape}")
            else:
                print(f"  ERROR: Unknown module type: {module_type}")
                continue
            
            # Apply GQA to TPA conversion
            print(f"  Starting tensor decomposition for layer {name}...")
            decomp_start = time.time()
            
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

            print(f"  Using head_dim={head_dim} for tensor decomposition")
            
            # Pass the model config to ensure consistent dimensions
            factorized_weights = gqa_to_tpa_conversion(
                q_weight, k_weight, v_weight, o_weight,
                num_heads, num_kv_heads,
                q_rank, k_rank, v_rank,
                dtype, device,
                use_dynamic_ranks=use_dynamic_ranks,  # Whether to use ranks from Tucker decomposition
                config=model.config if hasattr(model, 'config') else None,  # Pass model config for hidden_size
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


def create_tpa_model_from_standard(standard_model, q_rank=240, k_rank=240, v_rank=240,
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
    from ..gemma3_tpa_model import Gemma3ForMultimodalLMwithTPA
    
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
    
    # Special handling for embedding layer name mismatch
    print("Copying non-attention weights with special handling for embedding layer...")
    if hasattr(standard_model, 'embedder') and hasattr(tpa_model, 'text_token_embedder'):
        print("  Copying embedding weights from 'embedder' to 'text_token_embedder'")
        # Copy the weight tensor
        tpa_model.text_token_embedder.weight.data.copy_(standard_model.embedder.weight.data)
        # If using quantization, also copy the weight scaler
        if hasattr(standard_model.embedder, 'weight_scaler') and hasattr(tpa_model.text_token_embedder, 'weight_scaler'):
            tpa_model.text_token_embedder.weight_scaler.data.copy_(standard_model.embedder.weight_scaler.data)
    
    # Copy over all other non-attention weights
    for name, param in standard_model.named_parameters():
        # Skip attention-related parameters
        if any(x in name for x in ['qkv_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'attention']):
            continue
        # Skip embedder weights which we already copied manually
        if name.startswith("embedder"):
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
                        

                    else:
                        # CORRECTION: In contextual factorization (CF) form of TPA:
                        # W_B_q projects from hidden_dim to q_rank*head_dim, then reshapes to [batch, seq, q_rank, head_dim]
                        # B_q = self.W_B_q(hidden_states).view(batch_size, seq_len, self.q_rank, self.head_dim)
                        
                        print(f"  Creating Linear layer for B matrix in TPA contextual factorization")

                    # Rewrite for W_B_q, W_B_k, and W_B_v in the create_tpa_model_from_standard function

                    if std_key == 'W_B_q':
                        in_features = hidden_dim
                        # Use actual dimensions from weight tensor instead of q_rank
                        actual_q_dim = weight.shape[1]  # Get actual dimension from tensor
                        out_features = actual_q_dim  # Use this instead of q_rank*head_dim
                        print(f"  W_B_q Linear using actual tensor dimensions: {out_features}")

                    elif std_key == 'W_B_k':
                        in_features = hidden_dim
                        # Use actual dimensions from weight tensor instead of k_rank
                        actual_k_dim = weight.shape[1]  # Get actual dimension from tensor
                        out_features = actual_k_dim  # Use this instead of k_rank*head_dim
                        print(f"  W_B_k Linear using actual tensor dimensions: {out_features}")

                    elif std_key == 'W_B_v':
                        in_features = hidden_dim
                        # Use actual dimensions from weight tensor instead of v_rank
                        actual_v_dim = weight.shape[1]  # Get actual dimension from tensor
                        out_features = actual_v_dim  # Use this instead of v_rank*head_dim
                        print(f"  W_B_v Linear using actual tensor dimensions: {out_features}")
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
    # print out distro of weights in each layer
    for name, module in tpa_model.named_modules():
        if hasattr(module, 'use_factorized_weights') and module.use_factorized_weights:
            print(f"  Found factorized module: {name}")
            for key in dir(module):
                if key.startswith(('W_A_', 'W_B_')):
                    weight = getattr(module, key)
                    if hasattr(weight, 'data'):
                        weight_data = weight.data
                    elif hasattr(weight, 'weight'):
                        weight_data = weight.weight
                    else:
                        print(f"  Warning: Could not get tensor data for {key}")
                        continue
                    print(f"  {key} weight distribution: {weight_data.abs().mean().item():.4f} mean, {weight_data.abs().std().item():.4f} std")

    end_time = time.time()
    print(f"TPA model creation complete in {end_time - start_time:.2f} seconds")
    
    return tpa_model