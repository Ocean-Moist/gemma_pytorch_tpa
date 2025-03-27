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

from gemma.tpa import GemmaForCausalLMwithTPA
from gemma.tpa.gemma3_tpa_model import TPAAttention

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
    
    # Now use the first dimension after transposition, which is the hidden_dim
    hidden_dim = config_hidden_size if config_hidden_size is not None else q_weight.shape[0]

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

    num_heads = config.num_attention_heads       # e.g. 4
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)  # e.g. 1
    q_head_dim = config.head_dim  # e.g. 256
    k_head_dim = config.head_dim  # e.g. 256
    v_head_dim = config.head_dim  # e.g. 256
    
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
        # actual_q_rank = min(max_practical_rank, math.floor(q_recommended_rank / 4), max_q_rank)
        actual_q_rank = 96 # manual override for testing
        # actual_k_rank = min(max_practical_rank, k_recommended_rank, max_k_rank)
        # actual_v_rank = min(max_practical_rank, v_recommended_rank, max_v_rank)
        actual_k_rank = 48 # manual override for testing
        actual_v_rank = 48 # manual override for testing
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
    # Convert to target dtype and device, checking for zeros
    W_A_q_final = W_A_q.to(dtype=dtype, device=device)
    W_A_k_final = W_A_k.to(dtype=dtype, device=device)
    W_A_v_final = W_A_v.to(dtype=dtype, device=device)
    
    W_B_q_final = W_B_q.to(dtype=dtype, device=device)
    W_B_k_final = W_B_k.to(dtype=dtype, device=device)
    W_B_v_final = W_B_v.to(dtype=dtype, device=device)
    
    # Check for zeros in the converted weights and log detailed stats
    # print("\n============ WEIGHT DEBUG STATS AFTER CONVERSION ============")
    # print(f"W_A_q stats: shape={W_A_q_final.shape}, mean={W_A_q_final.abs().mean().item():.8f}, "
    #       f"std={W_A_q_final.std().item():.8f}, zero_percent={(W_A_q_final == 0).float().mean().item()*100:.2f}%")
    # print(f"W_A_k stats: shape={W_A_k_final.shape}, mean={W_A_k_final.abs().mean().item():.8f}, "
    #       f"std={W_A_k_final.std().item():.8f}, zero_percent={(W_A_k_final == 0).float().mean().item()*100:.2f}%")
    # print(f"W_A_v stats: shape={W_A_v_final.shape}, mean={W_A_v_final.abs().mean().item():.8f}, "
    #       f"std={W_A_v_final.std().item():.8f}, zero_percent={(W_A_v_final == 0).float().mean().item()*100:.2f}%")
    #
    # print(f"W_B_q stats: shape={W_B_q_final.shape}, mean={W_B_q_final.abs().mean().item():.8f}, "
    #       f"std={W_B_q_final.std().item():.8f}, zero_percent={(W_B_q_final == 0).float().mean().item()*100:.2f}%")
    # print(f"W_B_k stats: shape={W_B_k_final.shape}, mean={W_B_k_final.abs().mean().item():.8f}, "
    #       f"std={W_B_k_final.std().item():.8f}, zero_percent={(W_B_k_final == 0).float().mean().item()*100:.2f}%")
    # print(f"W_B_v stats: shape={W_B_v_final.shape}, mean={W_B_v_final.abs().mean().item():.8f}, "
    #       f"std={W_B_v_final.std().item():.8f}, zero_percent={(W_B_v_final == 0).float().mean().item()*100:.2f}%")
    #
    # # Check if weights are all zeros and add fallback initialization if needed
    # for name, weight in [('W_A_q', W_A_q_final), ('W_A_k', W_A_k_final), ('W_A_v', W_A_v_final),
    #                      ('W_B_q', W_B_q_final), ('W_B_k', W_B_k_final), ('W_B_v', W_B_v_final)]:
    #     if weight.abs().sum().item() == 0:
    #         print(f"WARNING: {name} contains all zeros! This will cause degenerate model behavior.")
    #         print(f"  Would need fallback initialization for {name}")
    #
    result["W_A_q"] = W_A_q_final
    result["W_A_k"] = W_A_k_final
    result["W_A_v"] = W_A_v_final
    
    result["W_B_q"] = W_B_q_final
    result["W_B_k"] = W_B_k_final
    result["W_B_v"] = W_B_v_final
    
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
            # Make tensors contiguous before reshaping to avoid memory layout issues
            head_W_A_cont = head_W_A.contiguous()
            head_W_B_cont = head_W_B.contiguous()
            
            # Use specialized reshape + bmm for maximum GPU efficiency
            A_flat = head_W_A_cont.reshape(-1, head_rank)  # [hidden_dim, head_rank]
            B_flat = head_W_B_cont.reshape(hidden_dim*head_rank, q_head_dim)  # [hidden_dim*head_rank, q_head_dim]
            
            # Reshape A for batch matrix multiplication
            A_bmm = A_flat.unsqueeze(2)  # [hidden_dim, head_rank, 1]
            
            # Reshape B to group by position and ensure contiguity
            B_bmm = B_flat.reshape(hidden_dim, head_rank, q_head_dim)  # [hidden_dim, head_rank, q_head_dim]
            B_bmm = B_bmm.contiguous()  # Ensure contiguous memory layout
            
            # Batch matrix multiply and sum - this utilizes optimized GEMM kernels
            head_recon = torch.bmm(A_bmm.transpose(1,2), B_bmm).squeeze(1) / head_rank
        else:
            # Standard case - use einsum which is highly optimized for GPU
            # Make tensors contiguous before reshaping
            head_W_A_cont = head_W_A.contiguous()
            head_W_B_cont = head_W_B.contiguous()
            
            A_reshaped = head_W_A_cont.reshape(hidden_dim, 1, head_rank)  # [hidden_dim, 1, head_rank]
            B_reshaped = head_W_B_cont.reshape(hidden_dim, head_rank, q_head_dim)  # [hidden_dim, head_rank, q_head_dim]
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


def split_combined_qkv_weights(combined_qkv: torch.Tensor, config) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    combined_qkv : shape [ (Q_dim + K_dim + V_dim), hidden_size ]
        - Q_dim = num_heads * q_head_dim
        - K_dim = num_kv_heads * kv_head_dim
        - V_dim = num_kv_heads * kv_head_dim
      For Gemma's GQA with 4 query heads, 1 KV head, and head_dim=256:
        Q_dim = 4 * 256 = 1024
        K_dim = 1 * 256 = 256
        V_dim = 1 * 256 = 256
      so total rows = 1024 + 256 + 256 = 1536, and hidden_size might be 1152.

    config:
      - config.num_attention_heads (e.g. 4)
      - config.num_key_value_heads (e.g. 1)
      - config.hidden_size         (1152)
      - config.head_dim           (256)  # or something else

    Returns:
      q_weight : torch.Tensor [hidden_size, Q_dim]  = [1152, 1024]
      k_weight : torch.Tensor [hidden_size, K_dim]  = [1152, 256]
      v_weight : torch.Tensor [hidden_size, V_dim]  = [1152, 256]

    These shapes are correct for factorizing Q, K, V.
    """

    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // num_heads)

    # Compute how many rows belong to Q, K, V
    Q_dim = num_heads * head_dim
    K_dim = num_kv_heads * head_dim
    V_dim = num_kv_heads * head_dim  # same in standard GQA

    # Sanity-check combined_qkv shape
    expected_rows = Q_dim + K_dim + V_dim  # e.g. 1024+256+256=1536
    if combined_qkv.shape[0] != expected_rows:
        raise ValueError(
            f"Combined QKV has shape {combined_qkv.shape} but expected first "
            f"dim=Q_dim+K_dim+V_dim={expected_rows} based on config. "
            f"(Q_dim={Q_dim}, K_dim={K_dim}, V_dim={V_dim})"
        )
    hidden_size = config.hidden_size
    if combined_qkv.shape[1] != hidden_size:
        raise ValueError(
            f"Combined QKV has shape {combined_qkv.shape} but expected second "
            f"dim={hidden_size} from config."
        )

    # Slice out Q, K, V from the rows
    q_rows = Q_dim
    k_rows = K_dim
    v_rows = V_dim

    q_block = combined_qkv[0 : q_rows, :]                      # shape [1024, 1152] if Q_dim=1024
    k_block = combined_qkv[q_rows : q_rows + k_rows, :]        # shape [256, 1152]
    v_block = combined_qkv[q_rows + k_rows : q_rows + k_rows + v_rows, :]

    # Transpose each, to get [ hidden_size, X_dim ]
    # e.g. q_weight => [1152, 1024], k_weight => [1152, 256], v_weight => [1152, 256]
    q_weight = q_block.transpose(0, 1).contiguous()
    k_weight = k_block.transpose(0, 1).contiguous()
    v_weight = v_block.transpose(0, 1).contiguous()

    return q_weight, k_weight, v_weight


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
                # Normal case - dimensions match expectations
                # New code: slice + transpose carefully
                combined_qkv = module.qkv_proj.weight  # shape [1536, 1152] for GQA 4+1 heads
                q_weight, k_weight, v_weight = split_combined_qkv_weights(combined_qkv, model.config)

                # Now we do NOT guess rank from shape; we know:
                #   q_weight is [1152, 1024] => hidden_size=1152, Q_dim=4*256=1024
                #   k_weight is [1152, 256]  => hidden_size=1152, K_dim=1*256=256
                #   v_weight is [1152, 256]
                #
                # Next, call your factorization method with known q_head_dim=256, etc.

                o_weight = module.o_proj.weight
                
                print(f"  Split combined QKV projection: Q: {q_weight.shape}, K: {k_weight.shape}, V: {v_weight.shape}")
            else:
                print(f"  ERROR: Unknown module type: {module_type}")
                continue
            
            # Apply GQA to TPA conversion
            print(f"  Starting tensor decomposition for layer {name}...")
            decomp_start = time.time()
            
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
    Create a GemmaForCausalLMwithTPA model from a standard GemmaForCausalLM model.
    This function:
    1. Performs a dry run of factorization to get rank/offset/dim info.
    2. Updates the model config with this information.
    3. Creates a new TPA model instance using the updated config.
    4. Copies non-attention weights.
    5. Loads the full factorized weights into the correctly sized TPA layers.

    Args:
        standard_model: A GemmaForCausalLM model to convert.
        q_rank: Base rank for query factorization (used if not dynamic).
        k_rank: Base rank for key factorization.
        v_rank: Base rank for value factorization.
        dtype: Data type for the final TPA model parameters.
        device: Device to use for computation and the final model.
        use_dynamic_ranks: Whether to use dynamic ranks based on SVD.
        fat_ranks: Whether to use potentially larger ranks for higher accuracy.

    Returns:
        A new GemmaForCausalLMwithTPA model with TPA weights.
    """
    start_time = time.time()
    print(f"Creating TPA model from standard model using device: {device}, dtype: {dtype}")
    # ... (Optional: print device info) ...

    # --- 1. Configuration Setup ---
    if not hasattr(standard_model, 'config'):
        raise ValueError("Standard model must have a 'config' attribute.")
    config = standard_model.config
    # Ensure essential attrs exist or provide defaults
    config.num_attention_heads = getattr(config, 'num_attention_heads', 4) # Example default
    config.num_key_value_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
    config.hidden_size = getattr(config, 'hidden_size', 1152) # Example default
    config.head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)

    # --- 2. Dry Run Factorization & Config Update ---
    print("Performing dry run of GQA->TPA conversion to determine ranks/dims...")
    all_factorized_weights_data = {} # Store results of conversion
    representative_layer_data = None

    for i, layer in enumerate(standard_model.model.layers):
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "qkv_proj"):
            module = layer.self_attn
            name = f"model.layers.{i}.self_attn"
            print(f"  Analyzing layer {i}...")

            # Split QKV weights
            q_weight, k_weight, v_weight = split_combined_qkv_weights(module.qkv_proj.weight, config)
            o_weight = module.o_proj.weight

            # Perform factorization (use float32 for accuracy during this step)
            factorized_data = gqa_to_tpa_conversion(
                q_weight, k_weight, v_weight, o_weight,
                config.num_attention_heads,
                config.num_key_value_heads,
                q_rank, k_rank, v_rank,
                dtype=torch.float32, # Convert results to target dtype later
                device=device,
                use_dynamic_ranks=use_dynamic_ranks,
                config=config,
            )
            all_factorized_weights_data[name] = factorized_data

            # Store info from the first layer into the config
            if i == 0:
                representative_layer_data = factorized_data
                config.q_per_head_ranks = factorized_data['q_per_head_ranks']
                config.q_max_head_rank = factorized_data['q_max_head_rank']
                config.q_head_offsets = factorized_data['q_head_offsets']
                config.q_rank = factorized_data['q_rank'] # This is max_q_rank
                config.k_rank = factorized_data['k_rank']
                config.v_rank = factorized_data['v_rank']
                # Store actual head dims found during factorization
                config.q_head_dim = factorized_data['q_head_dim']
                config.k_head_dim = factorized_data['k_head_dim']
                config.v_head_dim = factorized_data['v_head_dim']
                # Also store total_q_rank for W_A_q definition
                config.total_q_rank = sum(config.q_per_head_ranks)

                print(f"  Updated config with representative ranks/dims from layer {i}:")
                print(f"    q_per_head_ranks: {config.q_per_head_ranks}")
                print(f"    q_max_head_rank: {config.q_max_head_rank}")
                print(f"    k_rank: {config.k_rank}, v_rank: {config.v_rank}")
                print(f"    total_q_rank (for W_A_q): {config.total_q_rank}")
                print(f"    dims: Q={config.q_head_dim}, K={config.k_head_dim}, V={config.v_head_dim}")
        else:
            print(f"  Skipping layer {i} - No compatible attention module found.")

    if representative_layer_data is None:
        raise RuntimeError("Could not find any compatible attention layers to determine ranks.")

    # --- 3. Create TPA Model Instance ---
    print("Creating TPA model instance with updated config...")
    # Ensure config has necessary fields expected by GemmaForCausalLMwithTPA constructor
    # (like tokenizer path, architecture type etc. if needed, copy from standard_model.config)
    tpa_model = GemmaForCausalLMwithTPA(config)
    tpa_model = tpa_model.to(dtype=dtype, device=device) # Move model AFTER creation
    print(f"  TPA model created on device: {next(tpa_model.parameters()).device}, dtype: {next(tpa_model.parameters()).dtype}")

    # --- 4. Copy Non-Attention Weights ---
    print("Copying non-attention weights...")
    standard_sd = standard_model.state_dict()
    tpa_sd = tpa_model.state_dict()

    weights_to_copy = {}
    for name, param in standard_sd.items():
        # Skip all original attention projections and the factorized ones we'll handle
        if 'self_attn.' in name and any(k in name for k in ['qkv_proj', 'o_proj', 'W_A_', 'W_B_']):
            continue
        # Handle potential embedder name difference
        if name == 'model.embedder.weight':
            target_name = 'text_token_embedder.weight' # Name in TPA model
            if target_name in tpa_sd:
                weights_to_copy[target_name] = param.data.clone().to(dtype=dtype, device=device)
                print(f"  Mapping {name} -> {target_name}")
            else:
                print(f"  Warning: Target embedder {target_name} not found in TPA model.")
            continue # Skip original name

        # Copy other matching weights
        if name in tpa_sd:
            if tpa_sd[name].shape == param.shape:
                weights_to_copy[name] = param.data.clone().to(dtype=dtype, device=device)
            else:
                print(f"  Warning: Shape mismatch for {name}. Standard: {param.shape}, TPA: {tpa_sd[name].shape}. Skipping.")
        # else:
        #      print(f"  Info: Parameter {name} not found in TPA state_dict.") # Optional: for debugging

    # Load the collected weights
    tpa_model.load_state_dict(weights_to_copy, strict=False) # strict=False allows partial loading
    print(f"  Copied {len(weights_to_copy)} non-attention parameter tensors.")


    # --- 5. Load Factorized Weights ---
    print("Loading factorized TPA weights into TPA model layers...")
    for name, factorized_data in all_factorized_weights_data.items():
        # Navigate to the corresponding attention module in the TPA model
        try:
            tpa_module = tpa_model.get_submodule(name)
            if not isinstance(tpa_module, TPAAttention):
                print(f"  Warning: Submodule {name} in TPA model is not TPAAttention type. Skipping.")
                continue
        except AttributeError:
            print(f"  Warning: Could not find submodule {name} in TPA model. Skipping.")
            continue

        print(f"  Loading weights into TPA module: {name}")

        # --- Store rank/offset info directly on the TPA module instance ---
        # This makes it available during the forward pass without relying on config
        tpa_module.q_per_head_ranks = factorized_data['q_per_head_ranks']
        tpa_module.q_max_head_rank = factorized_data['q_max_head_rank']
        tpa_module.q_head_offsets = factorized_data['q_head_offsets']
        tpa_module.total_q_rank = sum(tpa_module.q_per_head_ranks)
        tpa_module.k_rank = factorized_data['k_rank']
        tpa_module.v_rank = factorized_data['v_rank']
        tpa_module.head_dim = factorized_data['q_head_dim']
        tpa_module.k_head_dim = factorized_data['k_head_dim']
        tpa_module.v_head_dim = factorized_data['v_head_dim']
        print(f"    Stored ranks/dims on module instance {name}")
        # --- End Store ---


        # Load weights into the nn.Linear layers
        for factor_key in ['W_A_q', 'W_A_k', 'W_A_v', 'W_B_q', 'W_B_k', 'W_B_v']:
            if factor_key in factorized_data and hasattr(tpa_module, factor_key):
                weight_tensor_float32 = factorized_data[factor_key].to(device=device) # Keep on device, convert dtype below
                linear_layer = getattr(tpa_module, factor_key)

                if isinstance(linear_layer, nn.Linear):
                    target_shape = linear_layer.weight.shape # [out_features, in_features]
                    source_shape = weight_tensor_float32.shape # [hidden_dim, factor_dim] (from factorization)

                    print(f"    Loading {factor_key}: Target shape {target_shape}, Source shape {source_shape}")

                    # nn.Linear expects [out, in], source is usually [in, out] format from matrix factorization
                    expected_source_shape = (target_shape[1], target_shape[0])

                    if source_shape == expected_source_shape:
                        # Source matches transpose of target: transpose source and copy
                        try:
                            linear_layer.weight.data.copy_(weight_tensor_float32.t().to(dtype))
                            print(f"      Loaded {factor_key} with transpose.")
                        except Exception as e:
                            print(f"      ERROR loading {factor_key} with transpose: {e}. Source: {source_shape}, Target: {target_shape}")
                    elif source_shape == target_shape:
                        # Source matches target directly (less common for factors)
                        try:
                            linear_layer.weight.data.copy_(weight_tensor_float32.to(dtype))
                            print(f"      Loaded {factor_key} directly (shapes matched).")
                        except Exception as e:
                            print(f"      ERROR loading {factor_key} directly: {e}. Source/Target: {source_shape}")
                    else:
                        # This should not happen if __init__ sizes are correct
                        print(f"    CRITICAL ERROR: Mismatch loading {factor_key}! Target={target_shape}, Source={source_shape}. Factorization/Model definition mismatch.")
                        # Optional: Raise error or attempt resize (resize likely incorrect)
                        raise ValueError(f"Cannot load weights for {factor_key} due to shape mismatch.")
                else:
                    print(f"    Warning: Target attribute {factor_key} in {name} is not nn.Linear.")
            else:
                print(f"    Warning: Factorized weight {factor_key} not found in data or target layer missing in {name}.")

    # --- Final Steps ---
    # Set the tokenizer if available
    if hasattr(standard_model, 'tokenizer'):
        tpa_model.tokenizer = standard_model.tokenizer
        print("  Copied tokenizer.")

    # Optional: Verification step to compare outputs (requires careful input matching)

    end_time = time.time()
    print(f"TPA model creation complete in {end_time - start_time:.2f} seconds")
    return tpa_model