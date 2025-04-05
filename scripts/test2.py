#!/usr/bin/env python3
# Copyright 2024 Google LLC & Rohan Tangri
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
convert_tpa_to_gqa_and_run.py

Script for:
  1) Loading a converted SVD-TPA model checkpoint (`svdtpa_model.pt`).
  2) Reconstructing approximate GQA weight matrices (qkv_proj) from the
     TPA factors (W_A, B_const).
  3) Loading these reconstructed weights, along with other original weights,
     into a standard GemmaForCausalLM (GQA) model structure.
  4) Running inference on this reconstructed GQA model.

Purpose: To isolate whether the error lies in the weight factorization process
         or the TPA forward pass implementation.
"""

import contextlib
import random
import os
import sys
from time import time
import gc

from absl import app
from absl import flags
import numpy as np
import torch
import tqdm

# Ensure the project root is in the path to find gemma modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Gemma modules
from gemma import config as gemma_config
from gemma import model as gemma_model # Standard Gemma GQA model
from gemma import tokenizer as gemma_tokenizer

# SVD-TPA modules (need model definition to load checkpoint)
try:
    from gemma.tpa.gemma3_tpa_model import GemmaForCausalLMwithSVDTPA, SVDTPAAttention
    tpa_modules_available = True
except ImportError as e:
    print(f"ERROR: Could not import SVD-TPA model definition: {e}")
    print("Cannot load the TPA checkpoint to perform reconstruction.")
    sys.exit(1)

# ------------------------------------------------------------
# ABSL Flags
# ------------------------------------------------------------

FLAGS = flags.FLAGS

# Model Loading & Conversion
flags.DEFINE_string('tpa_ckpt', None, 'Path to the converted SVD-TPA checkpoint file (.pt).', required=True)
flags.DEFINE_string('variant', '1b', 'Model variant (e.g., 1b, 4b, 12b, etc.). Used to get base GQA config.')
flags.DEFINE_string('device', 'cuda' if torch.cuda.is_available() else 'cpu',
                    'Device to run the model on (cpu or cuda).')
# Note: Precision used during reconstruction and inference
flags.DEFINE_enum('compute_dtype', 'float32', ['float32', 'bfloat16', 'float16'],
                  'Compute dtype for reconstruction and GQA inference.')

# Inference Parameters (copied from standard run.py)
flags.DEFINE_string('prompt', 'Write a short story about a spaceship landing on Mars.',
                    'Input prompt for the model.')
flags.DEFINE_integer('output_len', 128, 'Max number of new tokens to generate.')
flags.DEFINE_float('temperature', 0.9, 'Temperature for sampling. Use 0 for greedy decoding.')
flags.DEFINE_float('top_p', 0.95, 'Top-p sampling parameter.')
flags.DEFINE_integer('top_k', 64, 'Top-k sampling parameter.')

# Other Settings
flags.DEFINE_string('tokenizer_path', os.path.join(project_root, 'tokenizer/tokenizer.model'), # Default relative path
                    'Path to tokenizer model.')
flags.DEFINE_integer('seed', 12345, 'Random seed.')
flags.DEFINE_boolean('cuda_launch_blocking', False, 'Set CUDA_LAUNCH_BLOCKING=1 for sync debugging?')

# Define valid text only model variants
_VALID_MODEL_VARIANTS = ['2b', '2b-v2', '7b', '9b', '27b', '1b'] # From standard run.py

# Define valid devices
_VALID_DEVICES = ['cpu', 'cuda']

# Validator function for the 'variant' flag
def validate_variant(variant):
    if variant not in _VALID_MODEL_VARIANTS:
        raise ValueError(f'Invalid variant: {variant}. Valid variants are: {_VALID_MODEL_VARIANTS}')
    return True

# Validator function for the 'device' flag
def validate_device(device):
    if device not in _VALID_DEVICES:
        raise ValueError(f'Invalid device: {device}. Valid devices are: {_VALID_DEVICES}')
    return True

# Register validators
flags.register_validator('variant', validate_variant, message='Invalid model variant.')
flags.register_validator('device', validate_device, message='Invalid device.')


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    orig = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(orig) # Reset to PyTorch default (usually float32)


def get_base_config_gqa(variant: str, tokenizer_path: str, compute_dtype_str: str) -> gemma_config.GemmaConfig:
    """Gets the standard Gemma GQA configuration."""
    variant = variant.lower()

    # Select base config function based on variant
    if variant == '1b':
        config_func = gemma_config.get_config_for_1b
    elif variant == '4b':
        config_func = gemma_config.get_config_for_4b
    elif variant == '12b':
        config_func = gemma_config.get_config_for_12b
    elif variant == '27b':
        config_func = gemma_config.get_config_for_27b
    else:
        print(f"Warning: Unknown variant '{variant}' for base GQA config. Using 1b.")
        config_func = gemma_config.get_config_for_1b

    # Get base config with specified dtype (don't set quant here)
    model_config = config_func(dtype=compute_dtype_str)
    model_config.tokenizer = tokenizer_path
    model_config.quant = False # We are loading float weights reconstructed from TPA
    # Ensure QK norm is set based on original config defaults if not already present
    model_config.use_qk_norm = getattr(model_config, 'use_qk_norm', True if variant == '1b' else False) # Example default
    print(f"Base GQA Config uses QK Norm: {model_config.use_qk_norm}")
    return model_config


def reconstruct_gqa_weights_from_tpa(tpa_attn_module: SVDTPAAttention, device, dtype):
    """
    Reconstructs approximate GQA qkv_proj weight from TPA factors.

    Args:
        tpa_attn_module: An instance of SVDTPAAttention with factors loaded.
        device: Target device.
        dtype: Target data type.

    Returns:
        torch.Tensor: The reconstructed qkv_proj.weight tensor.
                      Shape [(N_h + 2*N_kv)*H_out, H_in]
    """
    # Retrieve factors and config from the loaded module
    W_A_q = tpa_attn_module.W_A_q.weight.data.to(device=device, dtype=dtype)
    B_const_q = tpa_attn_module.B_const_q.data.to(device=device, dtype=dtype)
    W_A_k = tpa_attn_module.W_A_k.weight.data.to(device=device, dtype=dtype)
    B_const_k = tpa_attn_module.B_const_k.data.to(device=device, dtype=dtype)
    W_A_v = tpa_attn_module.W_A_v.weight.data.to(device=device, dtype=dtype)
    B_const_v = tpa_attn_module.B_const_v.data.to(device=device, dtype=dtype)

    # Rank/Offset info stored during conversion on the module itself
    q_per_head_ranks = tpa_attn_module.q_per_head_ranks
    q_head_offsets = tpa_attn_module.q_head_offsets
    k_rank = tpa_attn_module.k_rank # Max k rank used across groups
    v_rank = tpa_attn_module.v_rank # Max v rank used across groups

    N_h = tpa_attn_module.num_heads
    N_kv = tpa_attn_module.num_kv_heads
    H_in = tpa_attn_module.hidden_size
    H_q_out = tpa_attn_module.head_dim
    H_k_out = tpa_attn_module.k_head_dim
    H_v_out = tpa_attn_module.v_head_dim

    # --- Reconstruct Q ---
    all_W_approx_q_head = []
    for h in range(N_h):
        head_rank = q_per_head_ranks[h]
        if head_rank == 0:
            print(f"Warning: Skipping Q head {h} due to zero rank.")
            all_W_approx_q_head.append(torch.zeros(H_in, H_q_out, device=device, dtype=dtype))
            continue

        start_A_idx = q_head_offsets[h]
        end_A_idx = q_head_offsets[h+1]

        # Extract factors for this head
        # W_A_q shape: [total_q_rank, H_in]
        head_W_A = W_A_q[start_A_idx:end_A_idx, :] # Shape: [head_rank, H_in]
        # B_const_q shape: [N_h, q_max_head_rank, H_q_out]
        head_B_const = B_const_q[h, :head_rank, :] # Shape: [head_rank, H_q_out]

        # Reconstruct W_approx = (W_A^T @ B_const) / R
        # (head_W_A.T @ head_B_const) needs shapes [H_in, head_rank] @ [head_rank, H_q_out]
        head_W_approx = (head_W_A.T @ head_B_const) / head_rank # Shape: [H_in, H_q_out]
        all_W_approx_q_head.append(head_W_approx)

    # Concatenate along the output dimension
    W_approx_q = torch.cat(all_W_approx_q_head, dim=1) # Shape: [H_in, N_h * H_q_out]

    # --- Reconstruct K ---
    all_W_approx_k_group = []
    for g in range(N_kv):
        if k_rank == 0:
            print(f"Warning: Skipping K group {g} due to zero rank.")
            all_W_approx_k_group.append(torch.zeros(H_in, H_k_out, device=device, dtype=dtype))
            continue
        # Extract factors for this group
        # W_A_k shape: [N_kv * k_rank, H_in]
        group_W_A = W_A_k[g*k_rank : (g+1)*k_rank, :] # Shape: [k_rank, H_in]
        # B_const_k shape: [N_kv, k_rank, H_k_out]
        group_B_const = B_const_k[g, :, :] # Shape: [k_rank, H_k_out]

        # Reconstruct W_approx = (W_A^T @ B_const) / R
        group_W_approx = (group_W_A.T @ group_B_const) / k_rank # Shape: [H_in, H_k_out]
        all_W_approx_k_group.append(group_W_approx)

    W_approx_k = torch.cat(all_W_approx_k_group, dim=1) # Shape: [H_in, N_kv * H_k_out]

    # --- Reconstruct V ---
    all_W_approx_v_group = []
    for g in range(N_kv):
        if v_rank == 0:
            print(f"Warning: Skipping V group {g} due to zero rank.")
            all_W_approx_v_group.append(torch.zeros(H_in, H_v_out, device=device, dtype=dtype))
            continue
        # Extract factors for this group
        # W_A_v shape: [N_kv * v_rank, H_in]
        group_W_A = W_A_v[g*v_rank : (g+1)*v_rank, :] # Shape: [v_rank, H_in]
        # B_const_v shape: [N_kv, v_rank, H_v_out]
        group_B_const = B_const_v[g, :, :] # Shape: [v_rank, H_v_out]

        # Reconstruct W_approx = (W_A^T @ B_const) / R
        group_W_approx = (group_W_A.T @ group_B_const) / v_rank # Shape: [H_in, H_v_out]
        all_W_approx_v_group.append(group_W_approx)

    W_approx_v = torch.cat(all_W_approx_v_group, dim=1) # Shape: [H_in, N_kv * H_v_out]

    # --- Assemble final qkv_proj weight ---
    # Target shape: [(N_h*H_q_out + N_kv*H_k_out + N_kv*H_v_out), H_in]
    qkv_proj_weight = torch.cat([
        W_approx_q.T, # Shape: [N_h * H_q_out, H_in]
        W_approx_k.T, # Shape: [N_kv * H_k_out, H_in]
        W_approx_v.T, # Shape: [N_kv * H_v_out, H_in]
    ], dim=0)

    return qkv_proj_weight


def main(_):
    if FLAGS.cuda_launch_blocking:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # --- Setup ---
    start_main = time()
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    torch_device = torch.device(FLAGS.device)

    # Determine compute dtype from flag
    if FLAGS.compute_dtype == 'bfloat16' and torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
    elif FLAGS.compute_dtype == 'float16':
        compute_dtype = torch.float16
    else: # Default to float32
        compute_dtype = torch.float32

    print(f"Using device: {torch_device}, Compute dtype: {compute_dtype}")
    print(f"Target GQA variant: {FLAGS.variant}")

    # Load Tokenizer
    if not os.path.exists(FLAGS.tokenizer_path):
        print(f"ERROR: Tokenizer not found at {FLAGS.tokenizer_path}")
        sys.exit(1)
    print(f"Loading tokenizer from {FLAGS.tokenizer_path}...")
    gemma_tok = gemma_tokenizer.Tokenizer(FLAGS.tokenizer_path)

    # --- 1. Load TPA Model Checkpoint ---
    load_start_time = time()
    if not os.path.isfile(FLAGS.tpa_ckpt):
        print(f"ERROR: TPA checkpoint file not found at {FLAGS.tpa_ckpt}")
        sys.exit(1)

    print(f"Loading SVD-TPA model checkpoint from {FLAGS.tpa_ckpt}...")
    try:
        # Load to CPU first to avoid device mismatches during inspection
        checkpoint = torch.load(FLAGS.tpa_ckpt, map_location='cpu', weights_only=False)
        if 'config' not in checkpoint or 'model_state_dict' not in checkpoint:
            raise ValueError("TPA Checkpoint must contain 'config' and 'model_state_dict'.")

        tpa_config = checkpoint['config']
        tpa_state_dict = checkpoint['model_state_dict']

        # Ensure tokenizer path is set
        tpa_config.tokenizer = getattr(tpa_config, 'tokenizer', FLAGS.tokenizer_path)
        # Ensure necessary rank/offset info is present
        if not hasattr(tpa_config, 'q_per_head_ranks'):
            raise ValueError("TPA config in checkpoint is missing factorization info (e.g., q_per_head_ranks). Cannot reconstruct.")

        # Instantiate TPA model to easily access loaded weights/buffers
        tpa_model = GemmaForCausalLMwithSVDTPA(tpa_config)
        # Load state dict carefully - allow missing/unexpected as we only need factors
        load_res = tpa_model.load_state_dict(tpa_state_dict, strict=False)
        if load_res.missing_keys: print(f"  Info: TPA load missing keys: {load_res.missing_keys[:5]}...")
        if load_res.unexpected_keys: print(f"  Info: TPA load unexpected keys: {load_res.unexpected_keys[:5]}...")

        tpa_model = tpa_model.to(device=torch_device, dtype=compute_dtype).eval()
        print("SVD-TPA model structure loaded successfully for factor extraction.")
        load_time_end = time()
        print(f"TPA Checkpoint loading time: {load_time_end - load_start_time:.2f} seconds")

    except Exception as e:
        print(f"ERROR loading SVD-TPA checkpoint: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --- 2. Instantiate Standard GQA Model ---
    print("\nInstantiating standard GQA model structure...")
    gqa_config = get_base_config_gqa(FLAGS.variant, FLAGS.tokenizer_path, FLAGS.compute_dtype)
    # Use compute_dtype for the GQA model that will run inference
    with _set_default_tensor_type(compute_dtype):
        gqa_model_reconstructed = gemma_model.GemmaForCausalLM(gqa_config)
        gqa_model_reconstructed = gqa_model_reconstructed.to(device=torch_device).eval()
    print("Standard GQA model structure created.")

    # --- 3. Reconstruct Weights and Build New State Dict ---
    print("\nReconstructing GQA weights from TPA factors...")
    reconstruct_start_time = time()
    new_gqa_state_dict = {}
    num_layers = gqa_config.num_hidden_layers

    # Copy non-attention weights from TPA state dict (assuming names mostly match)
    copied_keys = 0
    missing_in_gqa = []
    shape_mismatch = []
    gqa_target_sd = gqa_model_reconstructed.state_dict() # Get target keys/shapes

    for name, param in tpa_state_dict.items():
        is_attn_factor = False
        # Skip TPA-specific factor weights/buffers
        if 'self_attn.' in name and ('W_A_' in name or 'B_const_' in name):
            is_attn_factor = True
        # Skip standard attention projections that will be reconstructed
        if 'self_attn.' in name and ('qkv_proj.' in name or 'o_proj.' in name):
            is_attn_factor = True # Treat o_proj as handled per layer
        # Skip RoPE buffers
        if name in ['local_freqs_cis', 'global_freqs_cis', 'freqs_cis']:
            continue

        if not is_attn_factor:
            # Handle embedder name difference
            target_name = name
            if name == 'text_token_embedder.weight':
                target_name = 'embedder.weight' # Map to standard GQA name

            if target_name in gqa_target_sd:
                if gqa_target_sd[target_name].shape == param.shape:
                    new_gqa_state_dict[target_name] = param.data.clone().to(device=torch_device, dtype=compute_dtype)
                    copied_keys += 1
                else:
                    shape_mismatch.append((target_name, param.shape, gqa_target_sd[target_name].shape))
            else:
                missing_in_gqa.append(name)

    print(f"Copied {copied_keys} non-attention parameter tensors.")
    if missing_in_gqa: print(f"  Warning: TPA keys not found in GQA model: {missing_in_gqa[:5]}...")
    if shape_mismatch: print(f"  Warning: Shape mismatches found: {shape_mismatch[:5]}...")


    # Reconstruct attention weights per layer
    for i in range(num_layers):
        layer_name = f"model.layers.{i}"
        attn_module_name = f"{layer_name}.self_attn"
        print(f"  Reconstructing layer {i} attention weights...")

        try:
            tpa_attn_module = tpa_model.get_submodule(attn_module_name)
            if not isinstance(tpa_attn_module, SVDTPAAttention):
                print(f"    Warning: Module {attn_module_name} is not SVDTPAAttention. Skipping reconstruction.")
                continue

            # Reconstruct qkv_proj.weight
            qkv_rec_weight = reconstruct_gqa_weights_from_tpa(tpa_attn_module, torch_device, compute_dtype)
            qkv_target_key = f"{attn_module_name}.qkv_proj.weight"
            if qkv_target_key in gqa_target_sd and gqa_target_sd[qkv_target_key].shape == qkv_rec_weight.shape:
                new_gqa_state_dict[qkv_target_key] = qkv_rec_weight
            else:
                target_shape = gqa_target_sd.get(qkv_target_key, None)
                print(f"    ERROR: Shape mismatch for reconstructed {qkv_target_key}. Expected: {target_shape}, Got: {qkv_rec_weight.shape}")
                continue # Skip this layer's attn weights if shape is wrong

            # Copy o_proj.weight directly from TPA model (it wasn't factorized)
            o_proj_key = f"{attn_module_name}.o_proj.weight"
            if o_proj_key in tpa_state_dict and o_proj_key in gqa_target_sd:
                if gqa_target_sd[o_proj_key].shape == tpa_state_dict[o_proj_key].shape:
                    new_gqa_state_dict[o_proj_key] = tpa_state_dict[o_proj_key].data.clone().to(device=torch_device, dtype=compute_dtype)
                else:
                    shape_mismatch.append((o_proj_key, tpa_state_dict[o_proj_key].shape, gqa_target_sd[o_proj_key].shape))

        except AttributeError:
            print(f"    Warning: Could not find TPA attention module {attn_module_name}. Skipping reconstruction.")
        except Exception as e:
            print(f"    ERROR reconstructing weights for layer {i}: {e}")
            import traceback
            traceback.print_exc()

    reconstruct_end_time = time()
    print(f"Weight reconstruction finished in {reconstruct_end_time - reconstruct_start_time:.2f} seconds.")
    if shape_mismatch: print(f"  Warning: Additional shape mismatches during attn reconstruction: {shape_mismatch}")

    # --- 4. Load Reconstructed Weights into GQA Model ---
    print("\nLoading reconstructed weights into standard GQA model...")
    load_rec_start = time()
    load_result = gqa_model_reconstructed.load_state_dict(new_gqa_state_dict, strict=False)
    load_rec_end = time()
    print(f"Weight loading time: {load_rec_end - load_rec_start:.2f} seconds")
    if load_result.missing_keys:
        print(f"  Warning: Missing keys loading reconstructed state_dict: {load_result.missing_keys[:10]}...")
    if load_result.unexpected_keys:
        print(f"  Warning: Unexpected keys loading reconstructed state_dict: {load_result.unexpected_keys[:10]}...")
    print("Reconstructed GQA model ready.")

    # Clean up original TPA model
    del tpa_model
    del tpa_state_dict
    del checkpoint
    gc.collect()
    if torch_device.type == 'cuda': torch.cuda.empty_cache()

    # --- 5. Run Inference ---
    print(f"\n--- Starting Inference on Reconstructed GQA Model ---")
    print(f"Prompt: \"{FLAGS.prompt}\"")
    print(f"Settings: temp={FLAGS.temperature}, top_p={FLAGS.top_p}, top_k={FLAGS.top_k}, max_tokens={FLAGS.output_len}")

    # Use the standard GQA generate method
    with _set_default_tensor_type(gqa_model_reconstructed.dtype): # Use model's compute dtype
        generate_start_time = time()
        try:
            # Use the generate method from gemma_model.py
            results = gqa_model_reconstructed.generate(
                prompts=[FLAGS.prompt], # Pass as a list
                device=torch_device,
                output_len=FLAGS.output_len, # Use output_len argument
                temperature=FLAGS.temperature if FLAGS.temperature > 0 else None, # Pass None for greedy
                top_p=FLAGS.top_p,
                top_k=FLAGS.top_k,
            )
            output_text = results[0] # Get the first result

        except Exception as e:
            print(f"\nERROR during generation: {e}")
            import traceback
            traceback.print_exc()
            output_text = "[Generation Failed]"

        generate_end_time = time()

    # --- Print Results ---
    print("\n" + "="*60)
    print("Generation Output (Reconstructed GQA):")
    print("="*60)
    print(output_text)
    print("="*60 + "\n")

    generation_time = generate_end_time - generate_start_time
    try:
        num_output_tokens = len(gemma_tok.encode(output_text))
        tokens_per_sec = num_output_tokens / generation_time if generation_time > 0 else 0
        print(f"Generation completed in {generation_time:.2f} seconds.")
        print(f"Generated approximately {num_output_tokens} tokens ({tokens_per_sec:.2f} tokens/sec).")
    except Exception as e:
        print(f"Could not encode output for token counting: {e}")
        print(f"Generation completed in {generation_time:.2f} seconds.")


    # --- GPU Memory Usage ---
    if torch_device.type == 'cuda':
        mem_alloc = torch.cuda.memory_allocated(torch_device) / (1024**3)
        mem_resv = torch.cuda.memory_reserved(torch_device) / (1024**3)
        print(f"GPU Memory: Allocated={mem_alloc:.2f} GB, Reserved={mem_resv:.2f} GB")

    print(f"Total script time: {time() - start_main:.2f} seconds")
    print("Run complete.")


if __name__ == '__main__':
    # Ensure required flag 'tpa_ckpt' is provided
    flags.mark_flag_as_required('tpa_ckpt')
    app.run(main)