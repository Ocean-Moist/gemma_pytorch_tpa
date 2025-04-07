"""
GQA to TPA conversion using Tucker decomposition.

This module provides functionality to convert Grouped Query Attention (GQA) weights
to Tensor Product Attention (TPA) format using TensorLLM-style Tucker decomposition.
"""

import tensorly as tl

from gemma.model import GemmaForCausalLM
from gemma.tpa import GemmaForCausalLMwithSVDTPA
from gemma.tpa.gemma3_tpa_model import SVDTPAAttention

# Set PyTorch as backend, which will use CUDA if PyTorch is using CUDA
tl.set_backend('pytorch')

import torch
import math
import time
from typing import Dict, Tuple

# Assuming the TPA model and config definitions are available
# from gemma.tpa import GemmaForCausalLMwithTPA # Not needed here, but needed by caller
# from gemma.tpa.gemma3_tpa_model import TPAAttention # Not needed here

# --- Paste the previously generated compute_svd_tpa_factors here ---
def compute_svd_tpa_factors(
        weight_matrix: torch.Tensor,
        rank: int,
        name: str,
        head_dim: int,
        hidden_dim: int,
        device: torch.device = torch.device("cpu"), # Added device parameter
        dtype: torch.dtype = torch.float32 # Added dtype for consistency
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Constant B-Factor TPA factors using SVD.

    For a matrix W of shape [hidden_dim, proj_dim], this computes:
    W ≈ U_R Σ_R V_R^T

    Args:
        weight_matrix: The 2D weight matrix slice to factorize (e.g., a single head's W_Q^i).
                       Shape: [hidden_dim, proj_dim]. IMPORTANT: Assumes proj_dim == head_dim for this func.
        rank: The target rank R for the SVD truncation.
        name: String identifier (e.g., "K", "Q-head-0") for logging.
        head_dim: The dimension of the output head vectors.
        hidden_dim: The input hidden dimension (first dimension of weight_matrix).
        device: The torch device for computation.
        dtype: The torch dtype for computation.

    Returns:
        A tuple: (wa_weights_component, b_const_buffer_component)
        - wa_weights_component: torch.Tensor, shape [rank, hidden_dim]. Weights for the W_A linear layer component.
        - b_const_buffer_component: torch.Tensor, shape [rank, head_dim]. Constant buffer data for the B_const component.
    """
    start_time = time.time()
    # print(f"  Computing SVD factorization for {name} with target rank {rank}...") # Reduced verbosity

    # Ensure input is on the correct device and dtype for SVD
    weight_matrix_compute = weight_matrix.to(device=device, dtype=dtype)

    # --- 1. Validate Input Dimensions ---
    if weight_matrix_compute.shape[0] != hidden_dim:
        raise ValueError(f"[{name}] weight_matrix first dim {weight_matrix_compute.shape[0]} != expected hidden_dim {hidden_dim}")
    proj_dim = weight_matrix_compute.shape[1]
    if proj_dim != head_dim:
        print(f"  WARNING: [{name}] weight_matrix projection dim {proj_dim} != head_dim {head_dim}. "
              f"Ensure this function operates on per-head weight slices.")
        # We might need to adjust Vh later if the input wasn't per-head

    # --- 2. Perform Truncated SVD ---
    try:
        # Use float32 for SVD stability, convert back later if needed
        U, S, Vh = torch.linalg.svd(weight_matrix_compute.float(), full_matrices=False)
        # U: [hidden_dim, full_rank], S: [full_rank], Vh: [full_rank, proj_dim]
    except torch.linalg.LinAlgError as e:
        print(f"  ERROR: SVD failed for {name}: {e}. Returning zero tensors.")
        # Return zero tensors matching expected output shapes
        wa_weights_component = torch.zeros((rank, hidden_dim), device=device, dtype=dtype)
        b_const_buffer_component = torch.zeros((rank, head_dim), device=device, dtype=dtype)
        return wa_weights_component, b_const_buffer_component
    except RuntimeError as e:
        # Catch potential CUDA errors during SVD
        if "CUDA" in str(e):
            print(f"  ERROR: CUDA error during SVD for {name}: {e}. Trying on CPU.")
            try:
                U, S, Vh = torch.linalg.svd(weight_matrix_compute.cpu().float(), full_matrices=False)
                U, S, Vh = U.to(device), S.to(device), Vh.to(device) # Move back to original device
            except Exception as cpu_e:
                print(f"  ERROR: SVD failed on CPU as well for {name}: {cpu_e}. Returning zero tensors.")
                wa_weights_component = torch.zeros((rank, hidden_dim), device=device, dtype=dtype)
                b_const_buffer_component = torch.zeros((rank, head_dim), device=device, dtype=dtype)
                return wa_weights_component, b_const_buffer_component
        else:
            print(f"  ERROR: Unknown error during SVD for {name}: {e}. Returning zero tensors.")
            wa_weights_component = torch.zeros((rank, hidden_dim), device=device, dtype=dtype)
            b_const_buffer_component = torch.zeros((rank, head_dim), device=device, dtype=dtype)
            return wa_weights_component, b_const_buffer_component


    # --- 3. Ensure rank is valid ---
    max_possible_rank = min(weight_matrix_compute.shape[0], weight_matrix_compute.shape[1])
    effective_rank = min(rank, max_possible_rank, len(S)) # Also limited by actual singular values returned
    if effective_rank < rank:
        # print(f"  ADJUSTING {name} rank from {rank} to {effective_rank} (limited by matrix dimensions/SVD result)") # Reduced verbosity
        pass
    if effective_rank <= 0: # Use <= 0 for robustness
        print(f"  WARNING: [{name}] Effective rank is {effective_rank} after SVD. Returning zero tensors.")
        wa_weights_component = torch.zeros((rank, hidden_dim), device=device, dtype=dtype) # Use original target rank for shape
        b_const_buffer_component = torch.zeros((rank, head_dim), device=device, dtype=dtype)
        return wa_weights_component, b_const_buffer_component
    rank = effective_rank # Use the actual effective rank

    # --- 4. Truncate SVD components ---
    U_r = U[:, :rank]      # Shape: [hidden_dim, rank]
    S_r = S[:rank]         # Shape: [rank]
    Vh_r = Vh[:rank, :]    # Shape: [rank, proj_dim]

    # --- 5. Handle proj_dim vs head_dim in Vh_r ---
    if proj_dim != head_dim:
        # print(f"  Adjusting Vh_r projection dim {proj_dim} to match head_dim {head_dim} for {name}") # Reduced verbosity
        if proj_dim > head_dim:
            Vh_r = Vh_r[:, :head_dim] # Take the first 'head_dim' columns
            # print(f"    Truncated Vh_r to shape {Vh_r.shape}") # Reduced verbosity
        else:
            padding_size = head_dim - proj_dim
            padding = torch.zeros((rank, padding_size), device=device, dtype=dtype)
            Vh_r = torch.cat([Vh_r, padding], dim=1)
            # print(f"    Padded Vh_r to shape {Vh_r.shape}") # Reduced verbosity
    # Now Vh_r has shape [rank, head_dim]

    # --- 6. Calculate Scaling Factors ---
    sqrt_S_r = torch.sqrt(torch.clamp(S_r, min=1e-12)) # Shape: [rank]
    sqrt_rank_val = math.sqrt(rank) if rank > 0 else 1.0

    # --- 7. Compute W_A weights component (Corresponds to U) ---
    # Formula: W_A^T should be proportional to sqrt(R) * U_r
    # OLD: wa_untransposed = sqrt_rank_val * U_r * sqrt_S_r.unsqueeze(0) # Incorrect scaling
    wa_untransposed = sqrt_rank_val * U_r                     # Correct scaling based on derivation
    # Weights for nn.Linear(hidden_dim, rank_or_total_rank) should be [rank_or_total_rank, hidden_dim]
    wa_weights_component = wa_untransposed.t().contiguous() # Shape: [rank, hidden_dim]

    # --- 8. Compute B_const buffer component (Corresponds to Vh) ---
    # Formula: B_const proportional to sqrt(R) * S_r * Vh_r
    # ***** CORRECTION APPLIED HERE *****
    # Use S_r directly, not sqrt(S_r)
    b_const_buffer_component = sqrt_rank_val * S_r.unsqueeze(1) * Vh_r # Shape: [rank, head_dim]
    # Applying S_r.unsqueeze(1) to make S_r [rank, 1] broadcastable with Vh_r [rank, head_dim]
    b_const_buffer_component = b_const_buffer_component.contiguous()
    # ***** END OF CORRECTION *****

    # --- 9. (Optional) Compute and Log Reconstruction Error ---
    # This adds overhead, can be disabled for speed after verification
    verify_recon = True # Set to True to enable verification logging
    if verify_recon:
        try:
            reconstructed = U_r @ torch.diag(S_r) @ Vh_r
            target_weight = weight_matrix_compute # Assumes input was already [hidden_dim, head_dim]
            if reconstructed.shape == target_weight.shape:
                # Use float32 for error calculation stability
                error = torch.norm(target_weight.float() - reconstructed.float()) / torch.norm(target_weight.float())
                print(f"  {name} SVD component reconstruction error (rank {rank}): {error.item():.6f}") # Reduced verbosity
            else:
                print(f"  Skipping error calculation for {name} due to shape mismatch after adjustments.") # Reduced verbosity
                pass
        except Exception as e:
            print(f"  Error during reconstruction verification for {name}: {e}")

    end_time = time.time()
    # print(f"  {name} SVD factorization component completed in {end_time - start_time:.4f} seconds") # Reduced verbosity

    # --- 10. Return the distinct components ---
    # Convert back to the target dtype AFTER all computations
    return wa_weights_component.to(dtype=dtype), b_const_buffer_component.to(dtype=dtype)
# --- End of compute_svd_tpa_factors ---


def gqa_to_tpa_conversion(
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        v_weight: torch.Tensor,
        o_weight: torch.Tensor, # Keep o_weight for potential future use, though not factored here
        num_heads: int,
        num_kv_heads: int,
        q_rank: int = 240, # Target rank (may be reduced by SVD)
        k_rank: int = 12,  # Target rank
        v_rank: int = 12,  # Target rank
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        use_dynamic_ranks: bool = True, # Whether to use ranks determined by SVD (True) or force specified ranks (False)
        config = None, # Model config for dimensions
) -> Dict[str, torch.Tensor]:
    """
    Convert GQA attention weights to Constant B-Factor TPA format using SVD.

    Factorizes Q, K, V weights independently based on their head structures.
    Returns weights for W_A layers and constant data for B_const buffers.

    Args:
        q_weight: Query projection weight matrix [hidden_dim, num_heads * q_head_dim]
        k_weight: Key projection weight matrix [hidden_dim, num_kv_heads * kv_head_dim]
        v_weight: Value projection weight matrix [hidden_dim, num_kv_heads * kv_head_dim]
        o_weight: Output projection weight matrix [num_heads * q_head_dim, hidden_dim] (passed but not factored)
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads (groups)
        q_rank: Target rank for query factorization (per head)
        k_rank: Target rank for key factorization (per group)
        v_rank: Target rank for value factorization (per group)
        dtype: Data type for the output tensors (factors).
        device: Device for computation ('cuda' or 'cpu').
        use_dynamic_ranks: If True, attempts to find optimal ranks via SVD energy.
                           If False, uses the provided q_rank, k_rank, v_rank (capped by dims).
        config: Model config object (expected to have hidden_size, head_dim etc.).

    Returns:
        Dictionary containing:
        - 'W_A_weights': {'q': tensor, 'k': tensor, 'v': tensor} - Weights for nn.Linear layers.
        - 'B_const_buffers': {'q': tensor, 'k': tensor, 'v': tensor} - Constant buffer data.
        - 'Config_Updates': Dictionary with ranks, dims, offsets needed for model config.
    """
    tic = time.time()
    print("Starting GQA to Constant B-Factor TPA conversion...")
    torch_device = torch.device(device)

    # --- 1. Get Dimensions and Reshape ---
    if config is None:
        raise ValueError("Model config must be provided for dimensions.")

    hidden_dim = config.hidden_size
    q_head_dim = getattr(config, 'q_head_dim', config.head_dim) # Allow override from config if already set
    k_head_dim = getattr(config, 'k_head_dim', config.head_dim)
    v_head_dim = getattr(config, 'v_head_dim', config.head_dim)

    print(f"Using dimensions: hidden={hidden_dim}, q_head={q_head_dim}, k_head={k_head_dim}, v_head={v_head_dim}")
    print(f"Heads: Q={num_heads}, KV={num_kv_heads}")

    # Move original weights to compute device (use float32 for SVD precision)
    q_weight_f32 = q_weight.to(torch_device, dtype=torch.float32)
    k_weight_f32 = k_weight.to(torch_device, dtype=torch.float32)
    v_weight_f32 = v_weight.to(torch_device, dtype=torch.float32)

    # Reshape weights to [hidden_dim, num_heads/groups, head_dim]
    try:
        q_weights_reshaped = q_weight_f32.reshape(hidden_dim, num_heads, q_head_dim)
        k_weights_reshaped = k_weight_f32.reshape(hidden_dim, num_kv_heads, k_head_dim)
        v_weights_reshaped = v_weight_f32.reshape(hidden_dim, num_kv_heads, v_head_dim)
    except Exception as e:
        print(f"ERROR reshaping weights: {e}")
        print(f"Shapes - Q: {q_weight_f32.shape} (expected hidden={hidden_dim}, heads={num_heads}, dim={q_head_dim})")
        print(f"Shapes - K: {k_weight_f32.shape} (expected hidden={hidden_dim}, heads={num_kv_heads}, dim={k_head_dim})")
        print(f"Shapes - V: {v_weight_f32.shape} (expected hidden={hidden_dim}, heads={num_kv_heads}, dim={v_head_dim})")
        raise

    # --- 2. Determine Factorization Ranks ---
    # Placeholder for dynamic rank calculation (optional, based on use_dynamic_ranks)
    # For simplicity in this refactor, we'll primarily use the provided ranks,
    # capped by matrix dimensions in compute_svd_tpa_factors.
    # A full implementation of use_dynamic_ranks would involve SVD analysis here.
    actual_q_rank_target = q_rank
    actual_k_rank_target = k_rank
    actual_v_rank_target = v_rank
    if use_dynamic_ranks:
        print("Dynamic rank selection not fully implemented in this refactor - using provided target ranks.")
        # TODO: Implement SVD energy analysis here if use_dynamic_ranks is True
        pass

    print(f"Target ranks: Q={actual_q_rank_target}, K={actual_k_rank_target}, V={actual_v_rank_target}")

    # --- 3. Factorize Query Weights (Per Head) ---
    print("Factorizing Query weights per head...")
    all_wa_q_comps = []
    all_b_const_q_comps = []
    per_head_q_ranks_used = []

    for h in range(num_heads):
        head_weight = q_weights_reshaped[:, h, :] # Shape: [hidden_dim, q_head_dim]
        wa_comp, b_const_comp = compute_svd_tpa_factors(
            head_weight, actual_q_rank_target, f"Q-head-{h}", q_head_dim, hidden_dim, torch_device, torch.float32
        )
        # wa_comp shape: [head_rank_used, hidden_dim]
        # b_const_comp shape: [head_rank_used, q_head_dim]
        head_rank_used = wa_comp.shape[0] # Get actual rank used after SVD capping
        per_head_q_ranks_used.append(head_rank_used)
        all_wa_q_comps.append(wa_comp)
        all_b_const_q_comps.append(b_const_comp)

    # Assemble final W_A_q weights and B_const_q buffer
    total_q_rank = sum(per_head_q_ranks_used)
    q_max_head_rank = max(per_head_q_ranks_used) if per_head_q_ranks_used else 0
    q_head_offsets = [0] + torch.cumsum(torch.tensor(per_head_q_ranks_used), dim=0).tolist()

    final_W_A_q_weights = torch.cat(all_wa_q_comps, dim=0) # Shape: [total_q_rank, hidden_dim]

    # Pad B_const components to max rank before stacking for the buffer
    final_B_const_q_buffer = torch.zeros((num_heads, q_max_head_rank, q_head_dim), device=torch_device, dtype=torch.float32)
    for h in range(num_heads):
        head_rank_used = per_head_q_ranks_used[h]
        if head_rank_used > 0:
            final_B_const_q_buffer[h, :head_rank_used, :] = all_b_const_q_comps[h]

    print(f"Query factorization complete. Total Q rank: {total_q_rank}, Max per-head Q rank: {q_max_head_rank}")
    print(f"  Per-head ranks used: {per_head_q_ranks_used}")

    # --- 4. Factorize Key Weights (Per KV Group) ---
    print("Factorizing Key weights per KV group...")
    all_wa_k_comps = []
    all_b_const_k_comps = []
    per_group_k_ranks_used = []

    for g in range(num_kv_heads):
        group_weight = k_weights_reshaped[:, g, :] # Shape: [hidden_dim, k_head_dim]
        wa_comp, b_const_comp = compute_svd_tpa_factors(
            group_weight, actual_k_rank_target, f"K-group-{g}", k_head_dim, hidden_dim, torch_device, torch.float32
        )
        group_rank_used = wa_comp.shape[0]
        per_group_k_ranks_used.append(group_rank_used)
        all_wa_k_comps.append(wa_comp)
        all_b_const_k_comps.append(b_const_comp)

    # Assemble K factors - Assume all groups use the same max rank for simplicity in layer definition
    # If ranks differ significantly, padding or more complex handling might be needed.
    k_max_group_rank = max(per_group_k_ranks_used) if per_group_k_ranks_used else 0
    print(f"Max K group rank used: {k_max_group_rank}")

    # Stack W_A components, padding if necessary (though less likely needed for W_A)
    # We need W_A_k shape [num_kv_heads * k_max_group_rank, hidden_dim] for the Linear layer
    padded_wa_k_comps = []
    for wa_comp in all_wa_k_comps:
        rank_used = wa_comp.shape[0]
        if rank_used < k_max_group_rank:
            padding = torch.zeros((k_max_group_rank - rank_used, hidden_dim), device=torch_device, dtype=torch.float32)
            padded_wa_k_comps.append(torch.cat([wa_comp, padding], dim=0))
        else:
            padded_wa_k_comps.append(wa_comp)
    final_W_A_k_weights = torch.cat(padded_wa_k_comps, dim=0) # Shape: [num_kv_heads * k_max_group_rank, hidden_dim]

    # Stack B_const components into buffer [num_kv_heads, k_max_group_rank, k_head_dim]
    final_B_const_k_buffer = torch.zeros((num_kv_heads, k_max_group_rank, k_head_dim), device=torch_device, dtype=torch.float32)
    for g in range(num_kv_heads):
        group_rank_used = per_group_k_ranks_used[g]
        if group_rank_used > 0:
            final_B_const_k_buffer[g, :group_rank_used, :] = all_b_const_k_comps[g]

    # --- 5. Factorize Value Weights (Per KV Group) ---
    print("Factorizing Value weights per KV group...")
    all_wa_v_comps = []
    all_b_const_v_comps = []
    per_group_v_ranks_used = []

    for g in range(num_kv_heads):
        group_weight = v_weights_reshaped[:, g, :] # Shape: [hidden_dim, v_head_dim]
        wa_comp, b_const_comp = compute_svd_tpa_factors(
            group_weight, actual_v_rank_target, f"V-group-{g}", v_head_dim, hidden_dim, torch_device, torch.float32
        )
        group_rank_used = wa_comp.shape[0]
        per_group_v_ranks_used.append(group_rank_used)
        all_wa_v_comps.append(wa_comp)
        all_b_const_v_comps.append(b_const_comp)

    # Assemble V factors
    v_max_group_rank = max(per_group_v_ranks_used) if per_group_v_ranks_used else 0
    print(f"Max V group rank used: {v_max_group_rank}")

    padded_wa_v_comps = []
    for wa_comp in all_wa_v_comps:
        rank_used = wa_comp.shape[0]
        if rank_used < v_max_group_rank:
            padding = torch.zeros((v_max_group_rank - rank_used, hidden_dim), device=torch_device, dtype=torch.float32)
            padded_wa_v_comps.append(torch.cat([wa_comp, padding], dim=0))
        else:
            padded_wa_v_comps.append(wa_comp)
    final_W_A_v_weights = torch.cat(padded_wa_v_comps, dim=0) # Shape: [num_kv_heads * v_max_group_rank, hidden_dim]

    final_B_const_v_buffer = torch.zeros((num_kv_heads, v_max_group_rank, v_head_dim), device=torch_device, dtype=torch.float32)
    for g in range(num_kv_heads):
        group_rank_used = per_group_v_ranks_used[g]
        if group_rank_used > 0:
            final_B_const_v_buffer[g, :group_rank_used, :] = all_b_const_v_comps[g]

    # --- 6. Assemble Result Dictionary ---
    result = {
        'W_A_weights': {
            'q': final_W_A_q_weights.to(dtype=dtype),
            'k': final_W_A_k_weights.to(dtype=dtype),
            'v': final_W_A_v_weights.to(dtype=dtype),
        },
        'B_const_buffers': {
            'q': final_B_const_q_buffer.to(dtype=dtype),
            'k': final_B_const_k_buffer.to(dtype=dtype),
            'v': final_B_const_v_buffer.to(dtype=dtype),
        },
        'Config_Updates': {
            'q_per_head_ranks': per_head_q_ranks_used,
            'q_max_head_rank': q_max_head_rank,
            'q_head_offsets': q_head_offsets,
            'total_q_rank': total_q_rank, # Sum of actual ranks used
            'k_rank': k_max_group_rank, # Use max rank for consistency
            'v_rank': v_max_group_rank, # Use max rank for consistency
            'q_head_dim': q_head_dim,
            'k_head_dim': k_head_dim,
            'v_head_dim': v_head_dim,
        }
    }

    # --- 7. Optional: Verification (Skipped for brevity in this refactor) ---
    # print("\nVerification step skipped in this version.")

    toc = time.time()
    print(f"\nGQA to Constant B-Factor TPA conversion complete in {toc - tic:.2f} seconds")
    print(f"  Final effective ranks: Q_max={q_max_head_rank}, K={k_max_group_rank}, V={v_max_group_rank}")
    print(f"  Total Q rank (for W_A_q): {total_q_rank}")

    return result


# ----- Helper to split combined QKV weights -----
def split_combined_qkv_weights(combined_qkv: torch.Tensor, config) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Splits the combined QKV weight tensor from standard Gemma GQA format.

    Args:
        combined_qkv: Tensor of shape [(Q_rows + K_rows + V_rows), hidden_size]
        config: Model config containing head counts, head_dim, hidden_size.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: q_weight, k_weight, v_weight
        Each with shape [hidden_size, specific_proj_dim].
    """
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    # Use potentially different head dimensions from config if available
    q_head_dim = getattr(config, 'q_head_dim', config.head_dim)
    k_head_dim = getattr(config, 'k_head_dim', config.head_dim)
    v_head_dim = getattr(config, 'v_head_dim', config.head_dim)
    hidden_size = config.hidden_size

    # Compute rows for Q, K, V based on potentially different head dims
    Q_rows = num_heads * q_head_dim
    K_rows = num_kv_heads * k_head_dim
    V_rows = num_kv_heads * v_head_dim

    # Sanity-check combined_qkv shape
    expected_rows = Q_rows + K_rows + V_rows
    if combined_qkv.shape[0] != expected_rows:
        raise ValueError(
            f"Combined QKV has shape {combined_qkv.shape} but expected first "
            f"dim={expected_rows} based on config head dims. "
            f"(Q_rows={Q_rows}, K_rows={K_rows}, V_rows={V_rows})"
        )
    if combined_qkv.shape[1] != hidden_size:
        raise ValueError(
            f"Combined QKV has shape {combined_qkv.shape} but expected second "
            f"dim={hidden_size} from config."
        )

    # Slice out Q, K, V blocks from the rows
    q_block = combined_qkv[0 : Q_rows, :]                      # Shape [Q_rows, hidden_size]
    k_block = combined_qkv[Q_rows : Q_rows + K_rows, :]        # Shape [K_rows, hidden_size]
    v_block = combined_qkv[Q_rows + K_rows : Q_rows + K_rows + V_rows, :] # Shape [V_rows, hidden_size]

    # Transpose each to get [hidden_size, specific_proj_dim] as needed by factorization
    q_weight = q_block.t().contiguous() # Shape [hidden_size, Q_rows]
    k_weight = k_block.t().contiguous() # Shape [hidden_size, K_rows]
    v_weight = v_block.t().contiguous() # Shape [hidden_size, V_rows]

    return q_weight, k_weight, v_weight


# ----- Main function to create TPA model from standard -----
def create_tpa_model_from_standard(
        standard_model: GemmaForCausalLM,
        q_rank: int = 6,
        k_rank: int = 2,
        v_rank: int = 2,
        dtype: torch.dtype = torch.float16, # Use torch.dtype directly
        device: str = "cuda",
        use_dynamic_ranks: bool = True,
        fat_ranks: bool = False
) -> GemmaForCausalLMwithSVDTPA:
    """
    Creates a GemmaForCausalLMwithSVDTPA model from a standard GemmaForCausalLM.

    Performs factorization using gqa_to_tpa_conversion, updates the config,
    instantiates the SVDTPA model, copies non-attention weights (incl. o_proj),
    and loads the factorized weights (W_A weights and B_const buffers) along
    with layer-specific rank info onto each TPA module instance.

    Args:
        standard_model: A GemmaForCausalLM model to convert.
        q_rank: Base rank for query factorization.
        k_rank: Base rank for key factorization.
        v_rank: Base rank for value factorization.
        dtype: Data type for the final TPA model parameters.
        device: Device for computation and the final model.
        use_dynamic_ranks: Allow conversion to dynamically determine ranks via SVD.
        fat_ranks: Use larger fixed ranks (e.g., 240) during conversion.

    Returns:
        A new GemmaForCausalLMwithSVDTPA model instance.
    """
    start_time = time.time()
    print(f"Creating SVD-TPA model from standard model...")
    torch_device = torch.device(device)

    if not hasattr(standard_model, 'config'):
        raise ValueError("Standard model must have a 'config' attribute.")

    # --- 0. Clone Config and Prepare ---
    # Clone the config to avoid modifying the original standard_model's config
    config = standard_model.config.__class__(**vars(standard_model.config))

    # Ensure essential attrs exist
    config.num_attention_heads = getattr(config, 'num_attention_heads', 4)
    config.num_key_value_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
    config.hidden_size = getattr(config, 'hidden_size', 1152)
    config.head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
    # Make sure tokenizer path is present
    config.tokenizer = getattr(config, 'tokenizer', 'tokenizer/tokenizer.model')

    # Determine target ranks
    if fat_ranks:
        print("Using FAT ranks (e.g., 240) for factorization.")
        target_q_rank, target_k_rank, target_v_rank = 240, 240, 240
    else:
        target_q_rank, target_k_rank, target_v_rank = q_rank, k_rank, v_rank

    print("\n--- Standard Model State Dictionary Keys ---")
    standard_sd = standard_model.state_dict()
    # ... (key printing logic - unchanged) ...
    print("--- End Standard Model Keys ---\n")

    # --- 1. Factorize Weights (All Layers) & Collect Layer-Specific Data ---
    print("Factorizing weights for all layers...")
    all_factorized_weights_data = {} # Store factorized data {attn_module_name: factor_data}
    first_layer_config_updates = None

    num_layers = config.num_hidden_layers
    for i in range(num_layers):
        layer_name = f"model.layers.{i}"
        attn_module_name = f"{layer_name}.self_attn"
        qkv_proj_key = f"{attn_module_name}.qkv_proj.weight"
        o_proj_key = f"{attn_module_name}.o_proj.weight"

        # Check if weights exist in standard model state dict
        if qkv_proj_key not in standard_sd or o_proj_key not in standard_sd:
            print(f" Skipping layer {i}: Weights not found in standard_model state_dict.")
            continue

        print(f"  Factorizing layer {i}...")
        qkv_weight = standard_sd[qkv_proj_key]
        o_weight = standard_sd[o_proj_key] # Needed for factor_data but not factored itself

        try:
            q_weight, k_weight, v_weight = split_combined_qkv_weights(qkv_weight, config)
        except Exception as e:
            print(f"  ERROR splitting QKV for layer {i}: {e}")
            continue

        # Perform factorization (compute in float32 for precision)
        factor_data = gqa_to_tpa_conversion(
            q_weight, k_weight, v_weight, o_weight,
            config.num_attention_heads,
            config.num_key_value_heads,
            target_q_rank, target_k_rank, target_v_rank,
            dtype=torch.float32, # Compute factors in float32
            device=device,
            use_dynamic_ranks=use_dynamic_ranks,
            config=config,
        )
        all_factorized_weights_data[attn_module_name] = factor_data

        # Store config updates from the first layer for model instantiation
        if i == 0:
            first_layer_config_updates = factor_data['Config_Updates']
            print(f"  Using Layer 0 factorization results for global config:")
            for key, value in first_layer_config_updates.items():
                setattr(config, key, value) # Update the main config object
                # ... (config printing logic - unchanged) ...

    if first_layer_config_updates is None:
        raise RuntimeError("Could not factorize any attention layers to update config.")

    # --- 2. Create SVD-TPA Model Instance ---
    print("\nCreating SVD-TPA model instance with updated global config...")
    # The config object now contains ranks/dims derived from layer 0
    # Ensure QK Norm is explicitly handled if needed (e.g., disable it)
    # config.use_qk_norm = False # Example: Disable QK Norm
    # print(f"  SVD-TPA model will use QK Norm: {config.use_qk_norm}")

    tpa_model = GemmaForCausalLMwithSVDTPA(config)
    # Move model first, then load weights
    tpa_model = tpa_model.to(device=torch_device, dtype=dtype).eval()
    print(f"  SVD-TPA model created on device: {torch_device}, dtype: {dtype}")

    # --- 3. Prepare New State Dict for TPA Model ---
    print("\nPreparing state dict for TPA model...")
    tpa_new_sd = {}
    tpa_target_keys = set(tpa_model.state_dict().keys()) # Keys expected by the TPA model

    # Copy non-attention weights (MLP, Norms, Embeddings) using ORIGINAL standard_sd
    copied_keys_count = 0
    for name, param in standard_sd.items():
        # Skip QKV (handled by factors) but KEEP O_PROJ for now
        if 'self_attn.' in name and 'qkv_proj.' in name:
            continue
        # Skip RoPE buffers (they are registered automatically)
        if name in ['local_freqs_cis', 'global_freqs_cis', 'freqs_cis']:
            continue

        # Handle embedder name difference
        target_name = name
        # Standard Gemma uses 'embedder.weight', TPA model uses 'text_token_embedder.weight'
        if name == 'embedder.weight':
            target_name = 'text_token_embedder.weight'

        # Check if the target key exists in the TPA model structure
        if target_name in tpa_target_keys:
            # Check shape match before adding to the new state dict
            target_shape = tpa_model.get_parameter(target_name).shape if target_name in dict(tpa_model.named_parameters()) else tpa_model.get_buffer(target_name).shape
            if target_shape == param.shape:
                tpa_new_sd[target_name] = param.data.clone().to(dtype=dtype, device=torch_device)
                copied_keys_count += 1
            else:
                print(f"  Warning: Shape mismatch for non-attention weight {name} -> {target_name}. Skipping. "
                      f"Standard: {param.shape}, TPA: {target_shape}")
        else:
            print(f"  Info: Weight {name} from standard model not found in TPA model structure.")

    print(f"Prepared {copied_keys_count} non-attention weights for TPA state dict.")

    # Add factorized weights (W_A, B_const) to the new state dict
    factor_keys_added = 0
    for attn_module_name, factor_data in all_factorized_weights_data.items():
        layer_index = int(attn_module_name.split('.')[2]) # Get layer index

        # Add W_A weights
        for factor_key_part in ['q', 'k', 'v']:
            wa_key = f"{attn_module_name}.W_A_{factor_key_part}.weight"
            if wa_key in tpa_target_keys:
                source_weights = factor_data['W_A_weights'][factor_key_part]
                target_shape = tpa_model.get_parameter(wa_key).shape
                if target_shape == source_weights.shape:
                    tpa_new_sd[wa_key] = source_weights.to(dtype=dtype, device=torch_device)
                    factor_keys_added += 1
                else:
                    print(f"  ERROR: Shape mismatch for {wa_key} in layer {layer_index}! "
                          f"Target: {target_shape}, Source: {source_weights.shape}. Skipping.")
            else:
                print(f"  Warning: Target key {wa_key} not found in TPA model state dict.")

        # Add B_const buffers (Buffers are part of state_dict)
        for factor_key_part in ['q', 'k', 'v']:
            b_key = f"{attn_module_name}.B_const_{factor_key_part}"
            if b_key in tpa_target_keys:
                source_buffer_data = factor_data['B_const_buffers'][factor_key_part]
                target_shape = tpa_model.get_buffer(b_key).shape
                if target_shape == source_buffer_data.shape:
                    tpa_new_sd[b_key] = source_buffer_data.to(dtype=dtype, device=torch_device)
                    factor_keys_added += 1 # Technically buffers, but count them
                else:
                    print(f"  ERROR: Shape mismatch for {b_key} buffer in layer {layer_index}! "
                          f"Target: {target_shape}, Source: {source_buffer_data.shape}. Skipping.")
            else:
                print(f"  Warning: Target buffer key {b_key} not found in TPA model state dict.")

    print(f"Added {factor_keys_added} factor weight/buffer tensors to TPA state dict.")

    # --- 4. Load Final State Dict into TPA Model ---
    print("\nLoading final state dict into TPA model...")
    load_result = tpa_model.load_state_dict(tpa_new_sd, strict=True) # Use strict=True now
    # If strict=True passes, all expected keys were present and loaded.
    print("State dict loaded successfully into TPA model.")
    # Print any unexpected messages from strict loading if they occur
    if load_result.missing_keys: print(f"  Strict Load Warning: Missing keys: {load_result.missing_keys}")
    if load_result.unexpected_keys: print(f"  Strict Load Warning: Unexpected keys: {load_result.unexpected_keys}")


    # --- 5. Store Layer-Specific Rank Info on Modules ---
    # This step is crucial for the forward pass to use the correct ranks/offsets
    # It needs to happen AFTER the model is created and state_dict loaded (though state_dict loading doesn't affect these attributes)
    print("\nStoring layer-specific rank info onto TPA attention modules...")
    with torch.no_grad():
        for attn_module_name, factor_data in all_factorized_weights_data.items():
            try:
                tpa_module = tpa_model.get_submodule(attn_module_name)
                if not isinstance(tpa_module, SVDTPAAttention): continue

                config_updates = factor_data['Config_Updates']
                tpa_module.q_per_head_ranks = config_updates['q_per_head_ranks']
                tpa_module.q_max_head_rank = config_updates['q_max_head_rank']
                tpa_module.q_head_offsets = config_updates['q_head_offsets']
                tpa_module.total_q_rank = config_updates['total_q_rank']
                tpa_module.k_rank = config_updates['k_rank']
                tpa_module.v_rank = config_updates['v_rank']
                tpa_module.head_dim = config_updates['q_head_dim']
                tpa_module.k_head_dim = config_updates['k_head_dim']
                tpa_module.v_head_dim = config_updates['v_head_dim']
            except AttributeError:
                print(f"  Warning: Could not find or set attributes on submodule {attn_module_name}.")
    print("Layer-specific rank info stored.")


    # --- Final Steps ---
    if hasattr(standard_model, 'tokenizer'):
        tpa_model.tokenizer = standard_model.tokenizer
        print("Copied tokenizer.")

    tpa_model.eval() # Ensure eval mode

    end_time = time.time()
    print(f"\nSVD-TPA model creation complete in {end_time - start_time:.2f} seconds")
    return tpa_model