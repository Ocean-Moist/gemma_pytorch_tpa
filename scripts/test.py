# test_isp_kv_reconstruction.py
import argparse
import math
import os
import sys

import torch

# Ensure the project root is in the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    # If running from scripts/ directory, go up one level
    project_root_alt = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if os.path.isdir(os.path.join(project_root_alt, 'gemma')):
        sys.path.insert(0, project_root_alt)
    else: # If running from root, use current dir
        if os.path.isdir('gemma'):
            sys.path.insert(0, '.')
        else:
            print("Warning: Cannot determine project root. Imports might fail.")

# --- Imports ---
# Standard Gemma modules
from gemma import config as gemma_config

from gemma.tpa.modules.gqa_to_tpa import prepare_isp_kv_components, split_combined_qkv_weights  # Use GemmaConfig placeholder if needed
isp_kv_modules_available = True

# --- Helper Functions ---
def get_base_config(variant: str) -> gemma_config.GemmaConfig:
    """Gets the base Gemma configuration for a given variant."""
    variant = variant.lower()
    # USE FLOAT32 FOR THIS TEST SCRIPT
    dtype = 'float32'

    try:
        config_getter = getattr(gemma_config, f'get_config_for_{variant}')
        return config_getter(dtype=dtype)
    except AttributeError:
        print(f"Warning: Config function 'get_config_for_{variant}' not found in gemma.config. Using defaults.")
        return gemma_config.GemmaConfig( # Example defaults for ~1b/2b
            num_hidden_layers=18,
            num_attention_heads=8,
            num_key_value_heads=1,
            hidden_size=2048,
            intermediate_size=16384,
            head_dim=256, # Assuming 256 for 1b/2b
            dtype=dtype
        )
    except Exception as e:
        print(f"Error getting config for variant {variant}: {e}")
        raise

# --- Main Test Function ---
def test_reconstruction(args):
    print("--- Starting ISP-KV Reconstruction Test ---")
    device = torch.device(args.device)
    # CRITICAL: Use float32 for numerical verification
    test_dtype = torch.float32
    print(f"Using device: {device}, Test dtype: {test_dtype}")

    # 1. Load GQA State Dict
    print(f"Loading GQA state dict from: {args.ckpt}")
    if not os.path.exists(args.ckpt):
        print(f"ERROR: Checkpoint not found at {args.ckpt}")
        return
    try:
        if os.path.isfile(args.ckpt):
            full_state_dict = torch.load(args.ckpt, map_location="cpu")
            if 'model_state_dict' in full_state_dict:
                gqa_state_dict = full_state_dict['model_state_dict']
            else:
                gqa_state_dict = full_state_dict # Assume it's just the state dict
        else: # Assume directory for sharded
            print("Loading from directory (experimental, assumes single .bin or requires sharded logic)")
            # Basic loading for single file in dir, adapt if truly sharded
            bin_files = [f for f in os.listdir(args.ckpt) if f.endswith('.bin') or f.endswith('.pt')]
            if len(bin_files) == 1:
                ckpt_path = os.path.join(args.ckpt, bin_files[0])
                full_state_dict = torch.load(ckpt_path, map_location="cpu")
                if 'model_state_dict' in full_state_dict:
                    gqa_state_dict = full_state_dict['model_state_dict']
                else:
                    gqa_state_dict = full_state_dict
            else:
                print("ERROR: Cannot load from directory - expected single .bin/.pt or add sharded logic.")
                return
    except Exception as e:
        print(f"ERROR loading state dict: {e}")
        return
    print("GQA state dict loaded.")

    # 2. Get Config
    config = get_base_config(args.variant)
    # Override ranks to full dimension
    full_rank = config.head_dim
    target_r_k = full_rank
    target_r_v = full_rank
    print(f"Using Full Rank for test: r_k = {target_r_k}, r_v = {target_r_v}")

    # 3. Run ISP-KV Component Preparation
    print("Running prepare_isp_kv_components...")
    try:
        isp_kv_prep_data = prepare_isp_kv_components(
            gqa_state_dict=gqa_state_dict,
            config=config,
            r_k=target_r_k,
            r_v=target_r_v,
            device=args.device, # Use specified device for SVD
            dtype=test_dtype,  # Prepare components in float32
        )
    except Exception as e:
        print(f"ERROR during component preparation: {e}")
        import traceback
        traceback.print_exc()
        return
    print("ISP-KV components prepared.")

    # 4. Select Layer and Extract Data
    layer_idx = args.layer_idx
    if layer_idx not in isp_kv_prep_data['layer_specific_data']:
        print(f"ERROR: Data for layer {layer_idx} not found in preparation results.")
        return

    layer_data = isp_kv_prep_data['layer_specific_data'][layer_idx]
    layer_config_updates = layer_data['config_updates']
    effective_r_k = layer_config_updates['r_k']
    effective_r_v = layer_config_updates['r_v']

    # Check if full rank was actually achieved
    if effective_r_k != full_rank or effective_r_v != full_rank:
        print(f"Warning: Full rank not achieved for layer {layer_idx}. Effective r_k={effective_r_k}, r_v={effective_r_v}. Reconstruction won't be exact.")
        # Continue test, but expect non-zero reconstruction error

    # Extract weights (move to test device and dtype)
    qkv_weight_orig = layer_data['original_weights']['qkv_proj.weight'].to(device=device, dtype=test_dtype)
    try:
        q_weight, k_weight, v_weight = split_combined_qkv_weights(qkv_weight_orig, config)
        # Shapes are [hidden, proj_dim]
    except Exception as e:
        print(f"ERROR splitting QKV weights for testing layer {layer_idx}: {e}")
        return

    # Extract bases (should already be float32 and on correct device from prep)
    V_r_basis = layer_data['basis_buffers']['V_r_basis'].to(device=device, dtype=test_dtype) # [N_h, Dk, r_k]
    Z_v_basis = layer_data['basis_buffers']['Z_v_basis'].to(device=device, dtype=test_dtype) # [N_kv, Dv, r_v]
    print(f"Extracted data for layer {layer_idx}.")
    print(f"  V_r shape: {V_r_basis.shape}")
    print(f"  Z_v shape: {Z_v_basis.shape}")

    # 5. Check Basis Orthonormality
    print("\n--- Checking Basis Orthonormality (using float32) ---")
    with torch.no_grad():
        Dk = config.head_dim
        Dv = config.head_dim
        N_h = config.num_attention_heads
        N_kv = config.num_key_value_heads
        heads_per_group = N_h // N_kv

        # Check K basis
        if V_r_basis.shape[-1] == Dk: # Only if full rank was achieved
            I_k = torch.eye(Dk, device=device, dtype=test_dtype)
            # V_r @ V_r.T should be Identity
            P_k = torch.matmul(V_r_basis, V_r_basis.transpose(-1, -2)) # [N_h, Dk, Dk]
            # --- FIX: Calculate norm correctly for batch of matrices ---
            # Calculate Frobenius norm for each matrix's difference from Identity
            error_matrices_k = P_k - I_k.unsqueeze(0) # [N_h, Dk, Dk]
            # Compute squared Frobenius norm for each matrix in the batch
            squared_fro_norms_k = torch.sum(error_matrices_k**2, dim=(-1, -2)) # [N_h]
            # Sum of squared norms, then sqrt == Overall Frobenius norm of the "flattened" error tensor
            total_fro_norm_k = torch.sqrt(torch.sum(squared_fro_norms_k))
            # Normalize by sqrt of total number of elements N_h * Dk * Dk
            identity_error_k = total_fro_norm_k / math.sqrt(N_h * Dk * Dk)
            # --- End Fix ---

            print(f"  K Basis Check (||V_r @ V_r.T - I||_F / sqrt(N*D*D)): {identity_error_k.item():.6e}")
            if identity_error_k > 1e-5: print("    WARNING: K Basis orthonormality error is high!")
        else:
            print(f"  Skipping K Basis orthonormality check (effective r_k {V_r_basis.shape[-1]} != Dk {Dk})")

        # Check V basis
        if Z_v_basis.shape[-1] == Dv: # Only if full rank was achieved
            I_v = torch.eye(Dv, device=device, dtype=test_dtype)
            # Z_v @ Z_v.T should be Identity
            P_v = torch.matmul(Z_v_basis, Z_v_basis.transpose(-1, -2)) # [N_kv, Dv, Dv]

            # --- FIX: Calculate norm correctly for batch of matrices ---
            error_matrices_v = P_v - I_v.unsqueeze(0) # [N_kv, Dv, Dv]
            squared_fro_norms_v = torch.sum(error_matrices_v**2, dim=(-1, -2)) # [N_kv]
            total_fro_norm_v = torch.sqrt(torch.sum(squared_fro_norms_v))
            identity_error_v = total_fro_norm_v / math.sqrt(N_kv * Dv * Dv)
            # --- End Fix ---
            print(f"  V Basis Check (||Z_v @ Z_v.T - I||_F / sqrt(N*D*D)): {identity_error_v.item():.6e}")
        if identity_error_v > 1e-5: print("    WARNING: V Basis orthonormality error is high!")
        else:
            print(f"  Skipping V Basis orthonormality check (effective r_v {Z_v_basis.shape[-1]} != Dv {Dv})")

    # 6. Simulate Projection & Reconstruction
    print("\n--- Simulating Projection and Reconstruction ---")
    # Create dummy input
    B = 2 # Batch size
    S = 10 # Sequence length
    dummy_input = torch.randn(B, S, config.hidden_size, device=device, dtype=test_dtype)

    with torch.no_grad():
        # Calculate original K, V
        k_orig = torch.matmul(dummy_input, k_weight).view(B, S, N_kv, Dk) # W_k is [hidden, N_kv*Dk]
        v_orig = torch.matmul(dummy_input, v_weight).view(B, S, N_kv, Dv) # W_v is [hidden, N_kv*Dv]

        # Simulate ISP-KV Projection
        k_rep = k_orig.repeat_interleave(heads_per_group, dim=2) # [B, S, N_h, Dk]
        pk = torch.einsum('bshd,hdr->bshr', k_rep, V_r_basis) # [B, S, N_h, r_k]
        pv = torch.einsum('bsgd,gdr->bsgr', v_orig, Z_v_basis) # [B, S, N_kv, r_v]

        # Simulate ISP-KV Reconstruction
        k_hat = torch.einsum('bshr,hrd->bshd', pk, V_r_basis.transpose(-1, -2)) # [B, S, N_h, Dk]
        v_hat = torch.einsum('bsgr,grd->bsgd', pv, Z_v_basis.transpose(-1, -2)) # [B, S, N_kv, Dv]

        # 7. Compare Original vs. Reconstructed
        print("\n--- Comparing Original vs. Reconstructed ---")

        # Compare K
        # Need to compare k_hat ([B, S, N_h, Dk]) with k_rep ([B, S, N_h, Dk])
        k_diff = k_rep - k_hat
        relative_error_k = torch.linalg.norm(k_diff) / torch.linalg.norm(k_rep)
        print(f"  Relative L2 Error (K vs K_hat): {relative_error_k.item():.6e}")
        if relative_error_k > 1e-5: print("    WARNING: K reconstruction error is high!")

        # Compare V
        # Need to compare v_hat ([B, S, N_kv, Dv]) with v_orig ([B, S, N_kv, Dv])
        v_diff = v_orig - v_hat
        relative_error_v = torch.linalg.norm(v_diff) / torch.linalg.norm(v_orig)
        print(f"  Relative L2 Error (V vs V_hat): {relative_error_v.item():.6e}")
        if relative_error_v > 1e-5: print("    WARNING: V reconstruction error is high!")

    print("\n--- Test Complete ---")
    if relative_error_k < 1e-5 and relative_error_v < 1e-5:
        print("SUCCESS: Reconstruction errors are low, ISP-KV projection/reconstruction cycle appears numerically stable in float32.")
    else:
        print("FAILURE: High reconstruction errors detected. Check basis calculation or projection/reconstruction logic.")


# --- Argparse ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ISP-KV Reconstruction Accuracy")
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the original GQA checkpoint file or directory.')
    parser.add_argument('--variant', type=str, default='1b', help='Model variant (e.g., 1b, 2b) to load config.')
    parser.add_argument('--layer_idx', type=int, default=0, help='Index of the decoder layer to test.')
    # Ranks are implicitly set to full head_dim by get_config_for_variant + test logic
    # parser.add_argument('--r_k', type=int, default=256, help='Target rank r_k (set to head_dim for full rank test).')
    # parser.add_argument('--r_v', type=int, default=256, help='Target rank r_v (set to head_dim for full rank test).')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda or cpu).')

    args = parser.parse_args()
    test_reconstruction(args)