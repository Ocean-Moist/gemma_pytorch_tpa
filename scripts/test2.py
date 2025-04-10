# test_effective_gqa.py
import torch
import time
import math
import os
import sys
import argparse
import gc
import json
from typing import Dict, Tuple, Any

# Ensure the project root is in the path
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
from gemma import tokenizer as gemma_tokenizer

# Import necessary functions from the conversion script
try:
    # Need split_qkv and the config placeholder (or actual config class)
    from gemma.tpa.modules.gqa_to_tpa import split_combined_qkv_weights, GemmaConfig
except ImportError as e:
    print(f"ERROR: Could not import required functions from conversion script: {e}")
    sys.exit(1)

# --- Helper Functions ---
@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    orig = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try: yield
    finally: torch.set_default_dtype(orig)

def get_base_config(variant: str) -> gemma_config.GemmaConfig:
    """Gets the base Gemma configuration for a given variant."""
    variant = variant.lower()
    # USE FLOAT32 FOR THIS TEST SCRIPT
    dtype = 'float32'
    try:
        config_getter = getattr(gemma_config, f'get_config_for_{variant}')
        return config_getter(dtype=dtype)
    except AttributeError:
        print(f"Warning: Config function 'get_config_for_{variant}' not found. Using defaults.")
        return gemma_config.GemmaConfig(
            num_hidden_layers=18, num_attention_heads=8, num_key_value_heads=1,
            hidden_size=2048, intermediate_size=16384, head_dim=256, dtype=dtype
        )
    except Exception as e: print(f"Error getting config: {e}"); raise

def calculate_layer_z_v_basis(
        v_weight_layer: torch.Tensor, # Shape [hidden, N_kv * Dv]
        config: gemma_config.GemmaConfig,
        r_v: int,
        device: torch.device,
) -> Tuple[torch.Tensor, int]:
    """Calculates the Z_v basis for a single layer using W_v SVD."""
    hidden_dim = config.hidden_size
    num_kv_heads = config.num_key_value_heads
    Dv = config.head_dim # Assume Dv = head_dim
    compute_dtype = torch.float32 # Use float32 for SVD

    v_weight_reshaped = v_weight_layer.view(hidden_dim, num_kv_heads, Dv).to(device=device, dtype=compute_dtype)
    all_Z_v_bases_layer = []
    actual_r_v_layer_list = []

    for g in range(num_kv_heads):
        W_v_g = v_weight_reshaped[:, g, :] # Shape [hidden, Dv]
        try:
            # W_v = U S V^T. Z_v = U[:, :r_v]. Basis columns span output space.
            # Need U from SVD(W_v_g)
            U_v, S_v, Vv_T = torch.linalg.svd(W_v_g, full_matrices=False)
            # U_v shape [hidden, min(hidden, Dv)]
            effective_rank_limit = U_v.shape[1] # Check rank based on output dim Dv
            current_r_v = min(r_v, effective_rank_limit, Dv) # Cannot exceed Dv

            if current_r_v <= 0:
                print(f"    Warning: Effective r_v for group {g} is <= 0. Using zeros.")
                Z_v_g = torch.zeros((Dv, r_v), device=device, dtype=compute_dtype)
                current_r_v = 0
            else:
                # We need basis vectors operating on v (Dv). U's columns span the output space
                # Projection is p_v = v @ Z_v. Reconstruction v_hat = p_v @ Z_v.T
                # So Z_v must be [Dv, r_v]. The left singular vectors U_v give the basis for range(W_v).
                # BUT wait, W_v maps hidden->Dv. U_v maps hidden->hidden basis. V_v maps Dv->Dv basis.
                # We need the basis for the Dv space related to the *output* of W_v.
                # Let's re-read ISP-KV paper: "Value Output Subspace Basis: Compute SVD of W_v = U_v S_v V_v^T. Select top rv *left* singular vectors: Zv = U_v[:, 1:rv]".
                # This implies Z_v comes from U_v, shape [hidden, r_v].
                # BUT the projection is pv = v @ Zv. v is [B,S,Dv]. Zv must be [Dv, r_v].
                # This contradiction suggests either my understanding or the paper's notation needs clarification.
                # Let's trust the projection/reconstruction math: We need Z_v of shape [Dv, r_v].
                # The principal components/directions *in the Dv output space* are given by the *right* singular vectors V_v.
                # Let's try Z_v = V_v[:, :r_v] = Vv_T.T[:, :r_v]
                Z_v_g = Vv_T.t()[:, :current_r_v] # Shape [Dv, current_r_v]

            all_Z_v_bases_layer.append(Z_v_g)
            actual_r_v_layer_list.append(current_r_v)

        except torch.linalg.LinAlgError as e:
            print(f"    Error: SVD failed for W_v group {g}: {e}. Using zeros.")
            all_Z_v_bases_layer.append(torch.zeros((Dv, r_v), device=device, dtype=compute_dtype))
            actual_r_v_layer_list.append(0)

    max_r_v_layer = max(actual_r_v_layer_list) if actual_r_v_layer_list else 0

    # Pad and Stack Z_v bases
    final_Z_v_basis_layer = torch.zeros((num_kv_heads, Dv, max_r_v_layer), device=device, dtype=compute_dtype)
    for g, Z_v_g in enumerate(all_Z_v_bases_layer):
        rank_used = actual_r_v_layer_list[g]
        if rank_used > 0:
            final_Z_v_basis_layer[g, :, :rank_used] = Z_v_g

    return final_Z_v_basis_layer, max_r_v_layer

def calculate_effective_v_weight(
        v_weight_orig_layer: torch.Tensor, # Shape [hidden, N_kv * Dv]
        Z_v_basis_layer: torch.Tensor,     # Shape [N_kv, Dv, r_v]
        config: gemma_config.GemmaConfig,
        device: torch.device,
) -> torch.Tensor:
    """Calculates the effective W_v by simulating projection and reconstruction."""
    hidden_dim = config.hidden_size
    num_kv_heads = config.num_key_value_heads
    Dv = config.head_dim
    compute_dtype = torch.float32 # Use float32 for calculation

    v_weight_orig_reshaped = v_weight_orig_layer.view(hidden_dim, num_kv_heads, Dv).to(device=device, dtype=compute_dtype)
    Z_v_basis = Z_v_basis_layer.to(device=device, dtype=compute_dtype) # [N_kv, Dv, r_v]

    # Reconstruction matrix P_v = Z_v @ Z_v.T
    P_v = torch.matmul(Z_v_basis, Z_v_basis.transpose(-1, -2)) # [N_kv, Dv, Dv]

    # Effective Weight W_v_eff = W_v @ P_v (element-wise per group)
    # W_v is [hidden, N_kv, Dv]. Need W_v[h,:,g] @ P_v[g,:,:] ? No.
    # Need W_v_g[hidden, Dv] @ P_v_g[Dv, Dv] -> W_v_eff_g[hidden, Dv]
    W_v_eff_list = []
    for g in range(num_kv_heads):
        W_v_g = v_weight_orig_reshaped[:, g, :] # [hidden, Dv]
        P_v_g = P_v[g, :, :]                   # [Dv, Dv]
        W_v_eff_g = torch.matmul(W_v_g, P_v_g) # [hidden, Dv]
        W_v_eff_list.append(W_v_eff_g)

    # Stack back and reshape
    W_v_eff = torch.stack(W_v_eff_list, dim=1) # [hidden, N_kv, Dv]
    W_v_eff = W_v_eff.view(hidden_dim, num_kv_heads * Dv) # [hidden, N_kv * Dv]

    return W_v_eff.to(dtype=v_weight_orig_layer.dtype) # Return in original dtype


# --- Main Test Function ---
def test_effective_weights(args):
    print("--- Starting Effective GQA Weights Test (V-path only) ---")
    device = torch.device(args.device)
    # Use float32 for basis calc and effective weight calc
    test_dtype = torch.float32
    print(f"Using device: {device}, Test dtype: {test_dtype}")

    # 1. Load Original GQA State Dict
    print(f"Loading GQA state dict from: {args.ckpt}")
    # (Loading logic - same as in test_reconstruction.py)
    if not os.path.exists(args.ckpt): print(f"ERROR: Ckpt not found: {args.ckpt}"); return
    try:
        if os.path.isfile(args.ckpt):
            full_state_dict = torch.load(args.ckpt, map_location="cpu"); gqa_state_dict = full_state_dict.get('model_state_dict', full_state_dict)
        else:
            bin_files = [f for f in os.listdir(args.ckpt) if f.endswith('.bin') or f.endswith('.pt')]; assert len(bin_files)==1, "Dir must contain exactly one .bin/.pt"
            ckpt_path = os.path.join(args.ckpt, bin_files[0]); full_state_dict = torch.load(ckpt_path, map_location="cpu"); gqa_state_dict = full_state_dict.get('model_state_dict', full_state_dict)
    except Exception as e: print(f"ERROR loading state dict: {e}"); return
    print("GQA state dict loaded.")

    # 2. Get Config
    config = get_base_config(args.variant)
    target_r_v = config.head_dim # Use full rank for this test
    print(f"Using full rank for V-path test: r_v = {target_r_v}")

    # 3. Prepare New State Dict (Copy Original)
    new_state_dict = {k: v.clone() for k, v in gqa_state_dict.items()}
    num_layers = config.num_hidden_layers
    qkv_key_pattern = "model.layers.{}.self_attn.qkv_proj.weight" # Adjust if needed

    # 4. Iterate, Calculate Effective V Weights, Update State Dict
    print("Calculating and updating effective V weights for all layers...")
    for i in range(num_layers):
        print(f"  Processing layer {i}...")
        qkv_key = qkv_key_pattern.format(i)
        if qkv_key not in gqa_state_dict:
            print(f"    Skipping layer {i}: QKV weight not found.")
            continue

        qkv_weight_orig = gqa_state_dict[qkv_key]
        try:
            # Split to get original V weight
            q_orig, k_orig, v_orig = split_combined_qkv_weights(qkv_weight_orig, config)
            # Calculate Z_v basis for this layer's v_orig
            Z_v_basis, max_r_v = calculate_layer_z_v_basis(v_orig, config, target_r_v, device)
            # Calculate effective V weight
            W_v_eff = calculate_effective_v_weight(v_orig, Z_v_basis, config, device)

            # Recombine Q(orig), K(orig), V(eff)
            # Ensure correct layout matching original qkv_weight_orig
            q_part = q_orig
            k_part = k_orig
            v_part = W_v_eff
            if qkv_weight_orig.shape[0] > qkv_weight_orig.shape[1]: # Shape [ProjTotal, Hidden]
                qkv_eff = torch.cat([q_part.t(), k_part.t(), v_part.t()], dim=0)
            else: # Shape [Hidden, ProjTotal]
                qkv_eff = torch.cat([q_part, k_part, v_part], dim=1)

            # Update the new state dictionary
            new_state_dict[qkv_key] = qkv_eff.to(dtype=qkv_weight_orig.dtype, device='cpu') # Store back on CPU in original dtype

        except Exception as e:
            print(f"    ERROR processing layer {i}: {e}")
            import traceback; traceback.print_exc()
            print(f"    Skipping update for layer {i}.")

    print("Effective V weights calculated and state dict updated.")
    gc.collect() # Clean up intermediate tensors

    # 5. Load and Run Standard GQA Model with Modified Weights
    print("\n--- Loading STANDARD GQA model with effective V weights ---")
    # Instantiate standard model
    model = gemma_model.GemmaForCausalLM(config)
    print("Loading modified state dict...")
    try:
        load_result = model.load_state_dict(new_state_dict, strict=True) # Use strict=True
        print("State dict loaded successfully.")
    except Exception as e:
        print(f"ERROR loading modified state dict: {e}")
        # Try strict=False for debugging, but strict=True should pass if keys/shapes correct
        try:
            load_result = model.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded with strict=False. Missing: {load_result.missing_keys}, Unexpected: {load_result.unexpected_keys}")
        except Exception e2:
            print(f"ERROR loading modified state dict even with strict=False: {e2}")
            return

    # Move model to device
    # Determine inference dtype (use compute dtype from config if possible)
    inference_dtype = config.get_dtype() if hasattr(config, 'get_dtype') else torch.float32
    model = model.to(device=device, dtype=inference_dtype).eval()
    print(f"Model moved to {device} with dtype {inference_dtype}")

    # 6. Run Inference
    print(f"\n--- Running Inference with Effective V Weights ---")
    prompt = args.prompt
    print(f"Prompt: \"{prompt}\"")
    gemma_tok = gemma_tokenizer.Tokenizer(config.tokenizer) # Load tokenizer using path in config

    try:
        with torch.no_grad(), _set_default_tensor_type(model.dtype):
            results = model.generate(
                prompts=[prompt],
                device=device,
                output_len=args.output_len,
                temperature=args.temperature if args.temperature > 0 else None,
            )
            output_text = results[0]
    except Exception as e:
        print(f"ERROR during generation: {e}")
        import traceback; traceback.print_exc()
        output_text = "[Generation Failed]"

    # 7. Print Output
    print("\n" + "="*60)
    print("Generation Output (Standard Model with Effective V):")
    print("="*60)
    print(output_text)
    print("="*60 + "\n")
    print("Compare this output to the garbled output from the ISP-KV implementation.")
    print("If this output is coherent, the bug is likely in the ISP-KVAttention online logic (K-path or attention calculation).")
    print("If this output is also garbled, the numerical error from the V-path reconstruction itself is causing instability.")
    print("--- Test Complete ---")

# --- Argparse ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GQA Model with Effective ISP-KV V-Weights")
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the original GQA checkpoint file or directory.')
    parser.add_argument('--variant', type=str, default='1b', help='Model variant (e.g., 1b, 2b) to load config.')
    # parser.add_argument('--layer_idx', type=int, default=0, help='Layer index (currently modifies all layers).')
    # r_v is implicitly set to full rank by config.head_dim inside the script
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda or cpu).')
    parser.add_argument('--prompt', type=str, default="1+1=", help='Input prompt.')
    parser.add_argument('--output_len', type=int, default=20, help='Max new tokens.')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature (0 for greedy).')


    args = parser.parse_args()
    test_effective_weights(args)