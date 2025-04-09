"""
GQA to TPA conversion using Tucker decomposition.

This module provides functionality to convert Grouped Query Attention (GQA) weights
to Tensor Product Attention (TPA) format using TensorLLM-style Tucker decomposition.
"""

import tensorly as tl

from gemma.config import GemmaConfig
from time import time
from gemma.model import GemmaForCausalLM

# Set PyTorch as backend, which will use CUDA if PyTorch is using CUDA
tl.set_backend('pytorch')

import torch
import math
import time
from typing import Dict, Tuple, Any


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

def prepare_isp_kv_components(
        gqa_state_dict: Dict[str, torch.Tensor],
        config: GemmaConfig,
        r_k: int,
        r_v: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        qkv_proj_key_pattern: str = "model.layers.{}.self_attn.qkv_proj.weight",
        o_proj_key_pattern: str = "model.layers.{}.self_attn.o_proj.weight",
) -> Dict[str, Any]:
    """
    Prepares components for ISP-KV attention from standard GQA weights.

    Calculates the Key projection basis (V_r) based on QK interaction SVD,
    and the Value projection basis (Z_v) based on W_v SVD. Returns these
    bases along with the original projection weights needed for ISP-KV inference.

    Args:
        gqa_state_dict: State dictionary of the pre-trained GQA model.
        config: The GemmaConfig object for the model.
        r_k: Target rank for the Key interaction subspace.
        r_v: Target rank for the Value output subspace.
        device: The device ('cuda' or 'cpu') for computation.
        dtype: The target data type for the output weights and buffers.
        qkv_proj_key_pattern: Format string for QKV projection weight keys.
        o_proj_key_pattern: Format string for Output projection weight keys.

    Returns:
        A dictionary containing:
        - 'original_weights': {
              'qkv_proj.weight': Combined QKV weight tensor,
              'o_proj.weight': Output projection weight tensor
          } (Keys match potential layer structure, adjust if using split Q/K/V layers)
        - 'basis_buffers': {
              'V_r_basis': Stacked Key projection basis (shape [N_h, Dk, max_r_k]),
              'Z_v_basis': Stacked Value projection basis (shape [N_kv, Dv, max_r_v])
          }
        - 'config_updates': {
              'r_k': The maximum effective key rank used (max_r_k),
              'r_v': The maximum effective value rank used (max_r_v),
              'per_head_r_k': List of actual r_k used per query head,
              'per_group_r_v': List of actual r_v used per value group
          }
        - 'layer_specific_data': { # Data per original attention layer
             layer_idx (int): {
                 'original_weights': {'qkv': tensor, 'o': tensor}, # Original weights for *this* layer
                 'basis_buffers': {'V_r': tensor, 'Z_v': tensor}, # Bases for *this* layer
                 'config_updates': {'r_k': int, 'r_v': int, ...} # Actual ranks for *this* layer
             }
          }
    """
    print(f"--- Starting ISP-KV Component Preparation ---")
    print(f"Target ranks: r_k={r_k}, r_v={r_v}")
    start_time = time.time()
    torch_device = torch.device(device)
    compute_dtype = torch.float32 # Use float32 for SVD stability

    # --- Get model parameters from config ---
    hidden_dim = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim # Assume Dq = Dk = Dv = head_dim
    num_layers = config.num_hidden_layers
    heads_per_group = num_heads // num_kv_heads

    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")

    # --- Store results per layer ---
    all_layer_data = {}

    # --- Process Each Layer ---
    for layer_idx in range(num_layers):
        print(f"\nProcessing Layer {layer_idx}...")
        layer_start_time = time.time()

        # --- 1. Load/Extract Original Weights for this Layer ---
        qkv_key = qkv_proj_key_pattern.format(layer_idx)
        o_key = o_proj_key_pattern.format(layer_idx)

        if qkv_key not in gqa_state_dict or o_key not in gqa_state_dict:
            print(f"  Warning: Weights for layer {layer_idx} not found in state_dict. Skipping.")
            continue

        qkv_weight = gqa_state_dict[qkv_key].to(device=torch_device, dtype=compute_dtype)
        o_weight = gqa_state_dict[o_key].to(device=torch_device, dtype=compute_dtype)

        try:
            q_weight, k_weight, v_weight = split_combined_qkv_weights(qkv_weight, config)
            # q_weight: [hidden, N_h * Dq] -> Reshape later
            # k_weight: [hidden, N_kv * Dk] -> Reshape later
            # v_weight: [hidden, N_kv * Dv] -> Reshape later
        except Exception as e:
            print(f"  Error splitting QKV weights for layer {layer_idx}: {e}. Skipping layer.")
            continue

        # --- 2. Calculate Value Basis (Z_v) ---
        print("  Calculating Z_v basis (from W_v SVD - using RIGHT singular vectors)...") # Corrected comment
        all_Z_v_bases_layer = []
        actual_r_v_layer = []
        # Ensure v_weight is [hidden_dim, num_kv_heads * head_dim] before reshaping
        if v_weight.shape[0] != hidden_dim:
            raise ValueError(f"v_weight has unexpected shape {v_weight.shape}, expected first dim {hidden_dim}")
        try:
            v_weight_reshaped = v_weight.view(hidden_dim, num_kv_heads, head_dim)
        except RuntimeError as e:
            raise ValueError(f"Cannot reshape v_weight {v_weight.shape} into ({hidden_dim}, {num_kv_heads}, {head_dim})") from e


        for g in range(num_kv_heads):
            W_v_g = v_weight_reshaped[:, g, :] # Shape [hidden, Dv=head_dim]
            try:
                # W_v = U S V^T. Z_v should be V[:, :r_v].
                U_v, S_v, Vv_T = torch.linalg.svd(W_v_g, full_matrices=False)
                # Vv_T shape [min(hidden, Dv), Dv]. Vv = Vv_T.T shape [Dv, min(hidden, Dv)]
                effective_rank_limit = Vv_T.shape[0] # Number of singular values/vectors returned
                current_r_v = min(r_v, effective_rank_limit)

                if current_r_v <= 0:
                    print(f"    Warning: Effective r_v for group {g} is <= 0. Using zeros.")
                    # Create Z_v_g with correct target shape [Dv, r_v]
                    Z_v_g = torch.zeros((head_dim, r_v), device=torch_device, dtype=compute_dtype)
                    current_r_v = 0 # Record actual rank used
                else:
                    # Select the first current_r_v columns from Vv = Vv_T.T
                    Z_v_g = Vv_T.t()[:, :current_r_v] # Shape [Dv=head_dim, current_r_v]

                all_Z_v_bases_layer.append(Z_v_g)
                actual_r_v_layer.append(current_r_v)

            except torch.linalg.LinAlgError as e:
                print(f"    Error: SVD failed for W_v group {g}: {e}. Using zeros.")
                all_Z_v_bases_layer.append(torch.zeros((head_dim, r_v), device=torch_device, dtype=compute_dtype))
                actual_r_v_layer.append(0)

        max_r_v_layer = max(actual_r_v_layer) if actual_r_v_layer else 0
        print(f"    Max effective r_v for layer {layer_idx}: {max_r_v_layer} (Target: {r_v})")

        # Pad and Stack Z_v bases - Target shape [N_kv, Dv, max_r_v_layer]
        final_Z_v_basis_layer = torch.zeros((num_kv_heads, head_dim, max_r_v_layer), device=torch_device, dtype=compute_dtype)
        for g, Z_v_g in enumerate(all_Z_v_bases_layer):
            rank_used = actual_r_v_layer[g]
            if rank_used > 0:
                # Assign the [Dv, rank_used] tensor Z_v_g to the slice
                final_Z_v_basis_layer[g, :, :rank_used] = Z_v_g
        # Shape check: LHS slice is [head_dim, rank_used]. RHS Z_v_g is [head_dim, rank_used]. Matches.

        # --- 3. Calculate Key Basis (V_r) ---
        print("  Calculating V_r basis (from Wq^T Wk SVD)...")
        all_V_r_bases_layer = []
        actual_r_k_layer = []
        q_weight_reshaped = q_weight.view(hidden_dim, num_heads, head_dim)
        k_weight_reshaped = k_weight.view(hidden_dim, num_kv_heads, head_dim)

        for h in range(num_heads):
            W_q_h = q_weight_reshaped[:, h, :] # Shape [hidden, Dq]
            g = h // heads_per_group
            W_k_g = k_weight_reshaped[:, g, :] # Shape [hidden, Dk]

            # Assume Dq=Dk=head_dim
            if W_q_h.shape[1] != W_k_g.shape[1]:
                print(f"    Warning: Head dimensions mismatch Q({W_q_h.shape[1]}) vs K({W_k_g.shape[1]}) for head {h}. Skipping.")
                all_V_r_bases_layer.append(torch.zeros((head_dim, r_k), device=torch_device, dtype=compute_dtype))
                actual_r_k_layer.append(0)
                continue

            C_h = W_q_h.t() @ W_k_g # Shape [Dq, Dk]
            try:
                # C = U S V^T. V_r = V[:, :r_k]
                Uc_h, Sc_h, Vc_h_T = torch.linalg.svd(C_h, full_matrices=False)
                # Vc_h_T shape [min(Dq,Dk), Dk]. We need Vc_h = Vc_h_T.T shape [Dk, min(Dq,Dk)]
                effective_rank_limit = Vc_h_T.shape[0]
                current_r_k = min(r_k, effective_rank_limit)

                if current_r_k <= 0:
                    print(f"    Warning: Effective r_k for head {h} is <= 0. Using zeros.")
                    V_r_h = torch.zeros((head_dim, r_k), device=torch_device, dtype=compute_dtype) # Use target r_k for shape
                    current_r_k = 0
                else:
                    V_r_h = Vc_h_T.t()[:, :current_r_k] # Shape [Dk, current_r_k]

                all_V_r_bases_layer.append(V_r_h)
                actual_r_k_layer.append(current_r_k)

            except torch.linalg.LinAlgError as e:
                print(f"    Error: SVD failed for interaction matrix C head {h}: {e}. Using zeros.")
                all_V_r_bases_layer.append(torch.zeros((head_dim, r_k), device=torch_device, dtype=compute_dtype))
                actual_r_k_layer.append(0)

        max_r_k_layer = max(actual_r_k_layer) if actual_r_k_layer else 0
        print(f"    Max effective r_k for layer {layer_idx}: {max_r_k_layer} (Target: {r_k})")

        # Pad and Stack V_r bases
        final_V_r_basis_layer = torch.zeros((num_heads, head_dim, max_r_k_layer), device=torch_device, dtype=compute_dtype)
        for h, V_r_h in enumerate(all_V_r_bases_layer):
            rank_used = actual_r_k_layer[h]
            if rank_used > 0:
                final_V_r_basis_layer[h, :, :rank_used] = V_r_h

        # --- 4. Store Layer Results ---
        all_layer_data[layer_idx] = {
            'original_weights': {
                'qkv_proj.weight': gqa_state_dict[qkv_key].clone(), # Store original from state dict
                'o_proj.weight': gqa_state_dict[o_key].clone()
            },
            'basis_buffers': {
                # Convert final buffers to target dtype
                'V_r_basis': final_V_r_basis_layer.to(dtype=dtype),
                'Z_v_basis': final_Z_v_basis_layer.to(dtype=dtype)
            },
            'config_updates': {
                'r_k': max_r_k_layer, # Max effective rank used for this layer
                'r_v': max_r_v_layer, # Max effective rank used for this layer
                'per_head_r_k': actual_r_k_layer,
                'per_group_r_v': actual_r_v_layer,
            }
        }
        print(f"  Layer {layer_idx} processing time: {time() - layer_start_time:.2f}s")

    # --- 5. Aggregate Results (Optional but useful for global config) ---
    # Determine the overall max ranks across all layers for potentially consistent
    # buffer sizing in the final model, although layer-specific ranks are more precise.
    overall_max_r_k = 0
    overall_max_r_v = 0
    for idx, data in all_layer_data.items():
        overall_max_r_k = max(overall_max_r_k, data['config_updates']['r_k'])
        overall_max_r_v = max(overall_max_r_v, data['config_updates']['r_v'])

    print(f"\nOverall Max Ranks Found: r_k={overall_max_r_k}, r_v={overall_max_r_v}")

    # --- 6. Final Output Structure ---
    # Return the per-layer data directly. The loading script will need to iterate
    # through this dictionary and place the weights/buffers into the correct layers
    # of the instantiated ISP-KV model.

    total_time = time() - start_time
    print(f"--- ISP-KV Component Preparation Finished ({total_time:.2f}s) ---")

    # Add the global max ranks to the output for convenience
    final_output = {
        'layer_specific_data': all_layer_data,
        'global_max_ranks': {
            'r_k': overall_max_r_k,
            'r_v': overall_max_r_v
        }
    }

    return final_output