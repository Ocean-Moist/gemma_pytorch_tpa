# test_isp_kv_reconstruction_v2.py
import torch
import time
import math
import os
import sys
import argparse
import gc
import json
from typing import Dict, Tuple, Any

# --- Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    project_root_alt = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if os.path.isdir(os.path.join(project_root_alt, 'gemma')):
        sys.path.insert(0, project_root_alt)
    else:
        if os.path.isdir('gemma'): sys.path.insert(0, '.')
        else: print("Warning: Cannot determine project root. Imports might fail.")

# --- Imports ---
from gemma import config as gemma_config
from gemma import model as gemma_model
try:
    from gemma.tpa.modules.gqa_to_tpa import prepare_isp_kv_components, split_combined_qkv_weights, GemmaConfig
except ImportError as e:
    print(f"ERROR: Could not import required functions: {e}")
    sys.exit(1)

# --- Helper Functions ---
def get_base_config(variant: str, dtype_str: str) -> gemma_config.GemmaConfig:
    """Gets the base Gemma configuration with specified dtype string."""
    variant = variant.lower()
    try:
        config_getter = getattr(gemma_config, f'get_config_for_{variant}')
        # Pass the dtype string directly if the getter supports it
        # Otherwise, create config and manually set dtype if possible
        config = config_getter(dtype=dtype_str) # Assumes getter accepts dtype string
        # Ensure head_dim exists, default if necessary (e.g., for 1b/2b)
        if not hasattr(config, 'head_dim') or config.head_dim is None:
            config.head_dim = 256 # Default for 1b/2b
        return config
    except AttributeError:
        print(f"Warning: Config function 'get_config_for_{variant}' not found. Using defaults.")
        return gemma_config.GemmaConfig(
            num_hidden_layers=18, num_attention_heads=8, num_key_value_heads=1,
            hidden_size=2048, intermediate_size=16384, head_dim=256, dtype=dtype_str
        )
    except Exception as e: print(f"Error getting config: {e}"); raise

def run_simulation_and_compare(
        label: str,
        dummy_input: torch.Tensor,
        k_weight: torch.Tensor,
        v_weight: torch.Tensor,
        V_r_basis: torch.Tensor,
        Z_v_basis: torch.Tensor,
        config: gemma_config.GemmaConfig,
        sim_dtype: torch.dtype,
        device: torch.device
):
    """Runs the projection/reconstruction cycle and prints errors."""
    print(f"\n--- Running Simulation ({label} / {sim_dtype}) ---")

    N_h = config.num_attention_heads
    N_kv = config.num_key_value_heads
    Dk = config.head_dim
    Dv = config.head_dim
    heads_per_group = N_h // N_kv
    r_k = V_r_basis.shape[-1]
    r_v = Z_v_basis.shape[-1]

    # Ensure inputs are on correct device and dtype for this simulation run
    dummy_input = dummy_input.to(device=device, dtype=sim_dtype)
    k_weight = k_weight.to(device=device, dtype=sim_dtype)
    v_weight = v_weight.to(device=device, dtype=sim_dtype)
    V_r_basis = V_r_basis.to(device=device, dtype=sim_dtype)
    Z_v_basis = Z_v_basis.to(device=device, dtype=sim_dtype)

    with torch.no_grad():
        # Calculate original K, V
        k_orig = torch.matmul(dummy_input, k_weight).view(dummy_input.shape[0], dummy_input.shape[1], N_kv, Dk)
        v_orig = torch.matmul(dummy_input, v_weight).view(dummy_input.shape[0], dummy_input.shape[1], N_kv, Dv)
        assert k_orig.dtype == sim_dtype
        assert v_orig.dtype == sim_dtype

        # Simulate ISP-KV Projection
        k_rep = k_orig.repeat_interleave(heads_per_group, dim=2) # [B, S, N_h, Dk]
        pk = torch.einsum('bshd,hdr->bshr', k_rep, V_r_basis) # [B, S, N_h, r_k]
        pv = torch.einsum('bsgd,gdr->bsgr', v_orig, Z_v_basis) # [B, S, N_kv, r_v]
        assert pk.dtype == sim_dtype
        assert pv.dtype == sim_dtype

        # Simulate ISP-KV Reconstruction
        k_hat = torch.einsum('bshr,hrd->bshd', pk, V_r_basis.transpose(-1, -2)) # [B, S, N_h, Dk]
        v_hat = torch.einsum('bsgr,grd->bsgd', pv, Z_v_basis.transpose(-1, -2)) # [B, S, N_kv, Dv]
        assert k_hat.dtype == sim_dtype
        assert v_hat.dtype == sim_dtype

        # Compare Original vs. Reconstructed
        print(f"\n--- Comparing Original vs. Reconstructed ({label} / {sim_dtype}) ---")

        # Compare K
        k_diff = k_rep - k_hat
        norm_k_rep = torch.linalg.norm(k_rep.float()) # Use float for norm stability
        norm_k_diff = torch.linalg.norm(k_diff.float())
        relative_error_k = norm_k_diff / norm_k_rep if norm_k_rep > 1e-9 else torch.tensor(0.0)
        print(f"  Relative L2 Error (K vs K_hat): {relative_error_k.item():.6e}")
        if relative_error_k > 1e-4: print(f"    WARNING: K reconstruction error is high for {sim_dtype}!")

        # Compare V
        v_diff = v_orig - v_hat
        norm_v_orig = torch.linalg.norm(v_orig.float())
        norm_v_diff = torch.linalg.norm(v_diff.float())
        relative_error_v = norm_v_diff / norm_v_orig if norm_v_orig > 1e-9 else torch.tensor(0.0)
        print(f"  Relative L2 Error (V vs V_hat): {relative_error_v.item():.6e}")
        if relative_error_v > 1e-4: print(f"    WARNING: V reconstruction error is high for {sim_dtype}!")

        return relative_error_k.item(), relative_error_v.item()


# --- Main Test Function ---
def test_reconstruction(args):
    print("--- Starting ISP-KV Reconstruction Test V2 ---")
    device = torch.device(args.device)
    # Basis calculation always uses float32
    basis_calc_dtype = torch.float32
    print(f"Using device: {device}, Basis calculation dtype: {basis_calc_dtype}")

    # 1. Load GQA State Dict (to CPU)
    print(f"Loading GQA state dict from: {args.ckpt}")
    # (Loading logic - same as before)
    if not os.path.exists(args.ckpt): print(f"ERROR: Ckpt not found: {args.ckpt}"); return
    try:
        if os.path.isfile(args.ckpt):
            full_state_dict = torch.load(args.ckpt, map_location="cpu"); gqa_state_dict = full_state_dict.get('model_state_dict', full_state_dict)
        else:
            bin_files = [f for f in os.listdir(args.ckpt) if f.endswith('.bin') or f.endswith('.pt')]; assert len(bin_files)==1, "Dir must contain exactly one .bin/.pt"
            ckpt_path = os.path.join(args.ckpt, bin_files[0]); full_state_dict = torch.load(ckpt_path, map_location="cpu"); gqa_state_dict = full_state_dict.get('model_state_dict', full_state_dict)
    except Exception as e: print(f"ERROR loading state dict: {e}"); return
    print("GQA state dict loaded to CPU.")

    # 2. Get Config
    config = get_base_config(args.variant, 'float32') # Get config with float32 default
    full_rank = config.head_dim
    target_r_k = full_rank
    target_r_v = full_rank
    print(f"Using Full Rank for test: r_k = {target_r_k}, r_v = {target_r_v}")

    # 3. Run ISP-KV Component Preparation (in float32)
    print("Running prepare_isp_kv_components (in float32)...")
    try:
        isp_kv_prep_data = prepare_isp_kv_components(
            gqa_state_dict=gqa_state_dict,
            config=config,
            r_k=target_r_k,
            r_v=target_r_v,
            device=args.device, # Use target device for SVD computation
            dtype=basis_calc_dtype, # Ensure bases are prepared in float32
        )
    except Exception as e: print(f"ERROR during component prep: {e}"); import traceback; traceback.print_exc(); return
    print("ISP-KV components prepared.")

    # 4. Select Layer and Extract Data
    layer_idx = args.layer_idx
    if layer_idx not in isp_kv_prep_data['layer_specific_data']: print(f"ERROR: Data for layer {layer_idx} not found."); return

    layer_data = isp_kv_prep_data['layer_specific_data'][layer_idx]
    # Extract weights (keep on CPU for now, convert dtype later)
    qkv_weight_orig_cpu = layer_data['original_weights']['qkv_proj.weight']
    k_weight_cpu = None
    v_weight_cpu = None
    try:
        _, k_weight_cpu, v_weight_cpu = split_combined_qkv_weights(qkv_weight_orig_cpu, config)
    except Exception as e: print(f"ERROR splitting QKV weights for layer {layer_idx}: {e}"); return

    # Extract bases (should be float32, move to target device)
    V_r_basis = layer_data['basis_buffers']['V_r_basis'].to(device=device)
    Z_v_basis = layer_data['basis_buffers']['Z_v_basis'].to(device=device)
    print(f"Extracted data for layer {layer_idx}.")

    # 5. Check Basis Orthonormality (float32)
    print("\n--- Checking Basis Orthonormality (float32) ---")
    # (Orthonormality check logic - same as previous corrected version)
    with torch.no_grad():
        Dk = config.head_dim; Dv = config.head_dim; N_h = config.num_attention_heads; N_kv = config.num_key_value_heads
        if V_r_basis.shape[-1] == Dk:
            I_k = torch.eye(Dk, device=device, dtype=basis_calc_dtype)
            P_k = torch.matmul(V_r_basis, V_r_basis.transpose(-1, -2))
            error_matrices_k = P_k - I_k.unsqueeze(0); squared_fro_norms_k = torch.sum(error_matrices_k**2, dim=(-1, -2))
            identity_error_k = torch.sqrt(torch.sum(squared_fro_norms_k)) / math.sqrt(N_h * Dk * Dk)
            print(f"  K Basis Check (float32): {identity_error_k.item():.6e}")
            if identity_error_k > 1e-5: print("    WARNING: K Basis orthonormality error HIGH!")
        if Z_v_basis.shape[-1] == Dv:
            I_v = torch.eye(Dv, device=device, dtype=basis_calc_dtype)
            P_v = torch.matmul(Z_v_basis, Z_v_basis.transpose(-1, -2))
            error_matrices_v = P_v - I_v.unsqueeze(0); squared_fro_norms_v = torch.sum(error_matrices_v**2, dim=(-1, -2))
            identity_error_v = torch.sqrt(torch.sum(squared_fro_norms_v)) / math.sqrt(N_kv * Dv * Dv)
            print(f"  V Basis Check (float32): {identity_error_v.item():.6e}")
            if identity_error_v > 1e-5: print("    WARNING: V Basis orthonormality error HIGH!")

    # 6. Simulate Projection & Reconstruction in float32
    B = 2; S = 10
    dummy_input_cpu = torch.randn(B, S, config.hidden_size, dtype=torch.float32) # Keep input on CPU for now
    err_k_f32, err_v_f32 = run_simulation_and_compare(
        "float32", dummy_input_cpu, k_weight_cpu, v_weight_cpu,
        V_r_basis, Z_v_basis, config, torch.float32, device
    )

    # 7. Simulate Projection & Reconstruction in float64 (Optional, for comparison)
    if args.test_float64:
        try:
            err_k_f64, err_v_f64 = run_simulation_and_compare(
                "float64", dummy_input_cpu, k_weight_cpu, v_weight_cpu,
                V_r_basis, Z_v_basis, config, torch.float64, device
            )
        except Exception as e:
            print(f"\nWarning: float64 simulation failed (likely unsupported op or OOM): {e}")

    print("\n--- Test Summary ---")
    if err_k_f32 < 1e-5 and err_v_f32 < 1e-5:
        print("SUCCESS: Reconstruction errors in float32 are low (<1e-5).")
        print("The persistent garbled output in full inference is likely due to:")
        print("  1. Issues in the attention score/softmax calculation in ISP_KVAttention.forward.")
        print("  2. Issues in how Q is calculated or used.")
        print("  3. Issues with GQA grouping logic (v_hat_grouped).")
        print("  4. Potential subtle dtype mismatches during the full inference forward pass.")
        print("  5. Errors in loading the *original* Wq/Wk/Wv/Wo weights in run_isp_kv.py.")
    else:
        print("FAILURE: Reconstruction errors in float32 are still high (>1e-5).")
        print("This suggests potential issues in:")
        print("  1. The einsum/matmul logic for projection (pk, pv) or reconstruction (k_hat, v_hat).")
        print("  2. The basis calculation SVD itself (less likely if ortho check passed).")
        print("  3. Subtle dtype issues even within the float32 simulation.")

# --- Argparse ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ISP-KV Reconstruction Accuracy V2")
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the original GQA checkpoint file or directory.')
    parser.add_argument('--variant', type=str, default='1b', help='Model variant (e.g., 1b, 2b) to load config.')
    parser.add_argument('--layer_idx', type=int, default=0, help='Index of the decoder layer to test.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda or cpu).')
    parser.add_argument('--test_float64', action='store_true', help='Also run simulation in float64 for comparison.')

    args = parser.parse_args()
    test_reconstruction(args)