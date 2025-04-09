#!/usr/bin/env python3
# Copyright 2024 Google LLC
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
run_isp_kv.py (Modified from run_tpa.py)

Script for:
  1) Loading a standard Gemma GQA model (GemmaForCausalLM) from a checkpoint.
  2) Converting it to Interaction Subspace Projection KV (ISP-KV) Attention
     using the `prepare_isp_kv_components` function if requested.
  3) Saving the converted ISP-KV model (config + state dict with original
     weights and basis buffers) if requested.
  4) Running inference (text-only) with the standard Gemma model or the
     converted ISP-KV model.

Usage:
  - Convert GQA to ISP-KV and save:
      python scripts/run_isp_kv.py \
        --ckpt /path/to/gemma_model_dir_or_file \
        --variant 2b \
        --prompt "Write a poem about tensors" \
        --convert \
        --save_isp_kv /path/to/save/isp_kv_model.pt \
        --r_k 16 \
        --r_v 16 \
        --device cuda

  - Load an already converted ISP-KV model and run inference:
      python scripts/run_isp_kv.py \
        --ckpt /path/to/save/isp_kv_model.pt \
        --variant 2b \
        --prompt "Explain quantum field theory" \
        --convert=False \
        --device cuda

  - Run inference with the original GQA model (without conversion):
      python scripts/run_isp_kv.py \
        --ckpt /path/to/gemma_model_dir_or_file \
        --variant 2b \
        --prompt "What is attention?" \
        --convert=False \
        --device cuda \
        --force_gqa # Flag to ensure standard model is used

  - Additional settings like temperature, top-p, top-k, etc. can be used.
"""

import contextlib
import random
import os
import sys
from time import time as time
import gc
import json

from absl import app
from absl import flags
import numpy as np
import torch

from gemma.tpa import GemmaForCausalLMwithISP_KV
from gemma.tpa.modules import prepare_isp_kv_components, split_combined_qkv_weights

# Ensure the project root is in the path to find gemma modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Standard Gemma modules
from gemma import config as gemma_config
from gemma import model as gemma_model # Standard Gemma GQA model
from gemma import tokenizer as gemma_tokenizer


# ------------------------------------------------------------
# ABSL Flags
# ------------------------------------------------------------

FLAGS = flags.FLAGS

# Model Loading & Conversion
flags.DEFINE_string('ckpt', None, 'Path to the checkpoint file or directory (standard Gemma or converted ISP-KV).', required=True)
flags.DEFINE_string('variant', '2b', 'Model variant (e.g., 2b, 7b, etc.). Used to get base config.')
flags.DEFINE_string('device', 'cuda' if torch.cuda.is_available() else 'cpu',
                    'Device to run the model on (cpu or cuda).')
flags.DEFINE_boolean('quant', False, 'Use quantization? (Affects standard model loading). ISP-KV conversion currently outputs float/bf16.')
flags.DEFINE_boolean('convert', False, 'Convert standard GQA weights to ISP-KV? Requires standard ckpt.')
flags.DEFINE_string('save_isp_kv', None, 'Path to save converted ISP-KV model (config + state dict).')
flags.DEFINE_boolean('force_gqa', False, 'Force loading as standard GQA model, even if ckpt seems like ISP-KV.')

# ISP-KV Conversion Parameters
flags.DEFINE_integer('r_k', 16, 'Target rank for the Key interaction subspace in ISP-KV.')
flags.DEFINE_integer('r_v', 16, 'Target rank for the Value output subspace in ISP-KV.')

# Inference Parameters
flags.DEFINE_string('prompt', 'Write a short story about a lonely robot learning to paint.',
                    'Input prompt for the model.')
flags.DEFINE_integer('output_len', 128, 'Max number of new tokens to generate.')
flags.DEFINE_float('temperature', 0.9, 'Temperature for sampling. Use 0 for greedy decoding.')
flags.DEFINE_float('top_p', 0.95, 'Top-p sampling parameter.')
flags.DEFINE_integer('top_k', 64, 'Top-k sampling parameter.')

# Other Settings
flags.DEFINE_string('tokenizer_path', os.path.join(project_root, 'tokenizer/tokenizer.model'),
                    'Path to tokenizer model.')
flags.DEFINE_integer('seed', 12345, 'Random seed.')
flags.DEFINE_boolean('cuda_launch_blocking', False, 'Set CUDA_LAUNCH_BLOCKING=1 for sync debugging?')

@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    orig = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(orig)

def get_base_config(variant: str, device: str) -> gemma_config.GemmaConfig:
    """Gets the base Gemma configuration for a given variant."""
    variant = variant.lower()
    # Default compute dtype based on device
    dtype = 'bfloat16' if device == 'cuda' and torch.cuda.is_bf16_supported() else 'float16' if device == 'cuda' else 'float32'

    try:
        config_getter = getattr(gemma_config, f'get_config_for_{variant}')
        return config_getter(dtype=dtype)
    except AttributeError:
        print(f"Warning: Config function 'get_config_for_{variant}' not found in gemma.config. Using defaults.")
        # Provide some basic defaults if variant function missing
        return gemma_config.GemmaConfig(
            num_hidden_layers=18, # Example for ~2B
            num_attention_heads=8,
            num_key_value_heads=1,
            hidden_size=2048,
            intermediate_size=16384,
            head_dim=256,
            dtype=dtype
            # Add other essential fields manually if needed
        )
    except Exception as e:
        print(f"Error getting config for variant {variant}: {e}")
        raise

def main(_):
    if FLAGS.cuda_launch_blocking:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # --- Setup ---
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    torch_device = torch.device(FLAGS.device)
    # Determine compute dtype (prioritize bfloat16 on CUDA if available)
    if torch_device.type == 'cuda':
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        compute_dtype = torch.float32 # CPU generally uses float32

    print(f"Using device: {torch_device}, Compute dtype: {compute_dtype}")
    print(f"Running Gemma variant: {FLAGS.variant}")

    # Load Tokenizer
    if not os.path.exists(FLAGS.tokenizer_path):
        print(f"ERROR: Tokenizer not found at {FLAGS.tokenizer_path}")
        sys.exit(1)
    print(f"Loading tokenizer from {FLAGS.tokenizer_path}...")
    gemma_tok = gemma_tokenizer.Tokenizer(FLAGS.tokenizer_path)

    # --- Model Loading / Conversion ---
    model = None
    model_type = "Unknown"
    model_config = None
    load_start_time = time()

    # Check if the checkpoint path exists
    if not os.path.exists(FLAGS.ckpt):
        print(f"ERROR: Checkpoint file/directory not found at {FLAGS.ckpt}")
        sys.exit(1)

    # Determine if loading Standard GQA or attempting ISP-KV
    load_as_isp_kv = False
    if not FLAGS.convert and not FLAGS.force_gqa:
        # Try to detect ISP-KV format by loading config if checkpoint is a single file
        if os.path.isfile(FLAGS.ckpt):
            try:
                print(f"Peeking into checkpoint file {FLAGS.ckpt} to detect model type...")
                peek_ckpt = torch.load(FLAGS.ckpt, map_location='cpu', weights_only=False) # Need config
                if isinstance(peek_ckpt, dict) and 'config' in peek_ckpt:
                    # Check for ISP-KV specific attributes in the saved config
                    if hasattr(peek_ckpt['config'], 'r_k') and hasattr(peek_ckpt['config'], 'r_v'):
                        print("Checkpoint contains config with ISP-KV ranks (r_k, r_v). Attempting to load as ISP-KV.")
                        load_as_isp_kv = True
                        model_config = peek_ckpt['config'] # Use the loaded config
                del peek_ckpt
                gc.collect()
            except Exception as e:
                print(f"Warning: Could not peek into checkpoint file {FLAGS.ckpt}: {e}. Assuming standard GQA.")
        # If it's a directory, assume standard unless conversion happened before and was saved

    # --- Conversion Path ---
    if FLAGS.convert:
        print(f"\n--- Starting GQA to ISP-KV Conversion ---")
        print(f"Loading standard Gemma model from {FLAGS.ckpt}...")

        # Get base config and update with quantization flag
        base_config = get_base_config(FLAGS.variant, FLAGS.device)
        base_config.quant = FLAGS.quant
        base_config.tokenizer = FLAGS.tokenizer_path

        # Load standard model first (load weights to CPU initially)
        standard_model = gemma_model.GemmaForCausalLM(base_config)
        print("  Loading standard weights...")
        standard_model.load_weights(FLAGS.ckpt) # Assumes this loads to CPU or handled internally
        standard_model = standard_model.eval() # Set to eval mode
        print("Standard GQA model loaded.")

        # Perform conversion using prepare_isp_kv_components
        print("Preparing ISP-KV components (bases, original weights)...")
        convert_start = time()
        try:
            # Pass the standard model's state dict and config
            isp_kv_prep_data = prepare_isp_kv_components(
                gqa_state_dict=standard_model.state_dict(), # Pass the loaded state dict
                config=standard_model.config, # Use config from loaded model
                r_k=FLAGS.r_k,
                r_v=FLAGS.r_v,
                device=FLAGS.device, # Device for SVD computation
                dtype=compute_dtype, # Target dtype for final components
            )

            # Create the ISP-KV specific config
            # Start with the base config and add/update ISP-KV ranks
            model_config = standard_model.config # Use the config from the loaded standard model
            # Add the effective global max ranks found during conversion
            model_config.r_k = isp_kv_prep_data['global_max_ranks']['r_k']
            model_config.r_v = isp_kv_prep_data['global_max_ranks']['r_v']
            # Store per-layer info if needed by the model later (unlikely for forward pass)
            # model_config.layer_specific_ranks = isp_kv_prep_data['layer_specific_data'] # Optional

            print(f"ISP-KV components prepared in {time() - convert_start:.2f} seconds.")
            print(f"  Effective global max ranks: r_k={model_config.r_k}, r_v={model_config.r_v}")

            # Instantiate the ISP-KV model structure
            print("Instantiating ISP-KV model structure...")
            model = GemmaForCausalLMwithISP_KV(model_config)

            # Construct the state dictionary for the ISP-KV model
            print("Constructing state dictionary for ISP-KV model...")
            isp_kv_state_dict = {}
            layer_data = isp_kv_prep_data['layer_specific_data']

            # 1. Copy non-attention weights from standard model state dict
            standard_sd = standard_model.state_dict()
            for name, param in standard_sd.items():
                is_attention_weight = 'self_attn.' in name and ('qkv_proj.' in name or 'o_proj.' in name)
                is_basis_buffer = 'V_r_basis' in name or 'Z_v_basis' in name # New buffers
                if not is_attention_weight and not is_basis_buffer:
                    target_name = name.replace('embedder.', 'text_token_embedder.') # Handle embedder name difference
                    if target_name in model.state_dict(): # Check if key exists in ISP-KV model
                        isp_kv_state_dict[target_name] = param.clone()

            # 2. Add original projection weights and new basis buffers (per layer)
            for i in range(model.config.num_hidden_layers):
                if i not in layer_data: continue # Skip if layer was skipped during conversion

                # Original weights (assuming separate Wq, Wk, Wv, Wo layers in ISP_KVAttention)
                # If ISP_KVAttention uses combined qkv_proj, load that instead. Adjust keys accordingly.
                qkv_orig = layer_data[i]['original_weights']['qkv_proj.weight']
                o_orig = layer_data[i]['original_weights']['o_proj.weight']

                # Option A: If ISP_KVAttention keeps qkv_proj combined layer
                # qkv_target_key = f"model.layers.{i}.self_attn.qkv_proj.weight"
                # if qkv_target_key in model.state_dict():
                #      isp_kv_state_dict[qkv_target_key] = qkv_orig

                # Option B: If ISP_KVAttention uses separate Wq, Wk, Wv layers
                q_orig_split, k_orig_split, v_orig_split = split_combined_qkv_weights(qkv_orig, model.config)
                q_target_key = f"model.layers.{i}.self_attn.W_q.weight"
                k_target_key = f"model.layers.{i}.self_attn.W_k.weight"
                v_target_key = f"model.layers.{i}.self_attn.W_v.weight"
                if q_target_key in model.state_dict(): isp_kv_state_dict[q_target_key] = q_orig_split
                if k_target_key in model.state_dict(): isp_kv_state_dict[k_target_key] = k_orig_split
                if v_target_key in model.state_dict(): isp_kv_state_dict[v_target_key] = v_orig_split

                # Output projection weight
                o_target_key = f"model.layers.{i}.self_attn.o_proj.weight"
                if o_target_key in model.state_dict(): isp_kv_state_dict[o_target_key] = o_orig

                # Basis buffers
                V_r_basis = layer_data[i]['basis_buffers']['V_r_basis']
                Z_v_basis = layer_data[i]['basis_buffers']['Z_v_basis']
                vr_target_key = f"model.layers.{i}.self_attn.V_r_basis"
                zv_target_key = f"model.layers.{i}.self_attn.Z_v_basis"
                if vr_target_key in model.state_dict(): isp_kv_state_dict[vr_target_key] = V_r_basis
                if zv_target_key in model.state_dict(): isp_kv_state_dict[zv_target_key] = Z_v_basis

            # Load the constructed state dictionary
            print("Loading constructed state dictionary into ISP-KV model...")
            load_result = model.load_state_dict(isp_kv_state_dict, strict=False)
            print(f"  Load Result - Missing: {load_result.missing_keys}, Unexpected: {load_result.unexpected_keys}")

            model = model.to(device=torch_device, dtype=compute_dtype).eval()
            model_type = "ISP-KV (Converted)"

            # Save the converted model if requested
            if FLAGS.save_isp_kv:
                print(f"Saving converted ISP-KV model to {FLAGS.save_isp_kv}...")
                save_dir = os.path.dirname(FLAGS.save_isp_kv)
                if save_dir: os.makedirs(save_dir, exist_ok=True)
                # Save both the ISP-KV config and the state dict
                torch.save({'config': model.config, 'model_state_dict': model.state_dict()}, FLAGS.save_isp_kv)
                print("ISP-KV model saved.")

        except Exception as e:
            print(f"ERROR during conversion or ISP-KV model setup: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        finally:
            # Clean up standard model
            del standard_model
            del standard_sd
            del isp_kv_prep_data
            del isp_kv_state_dict
            gc.collect()
            if torch_device.type == 'cuda': torch.cuda.empty_cache()

    # --- Loading Path (Standard GQA or Pre-converted ISP-KV) ---
    else:
        if load_as_isp_kv:
            print(f"Loading pre-converted ISP-KV model from {FLAGS.ckpt}...")
            try:
                # model_config should have been loaded during peeking if ckpt is file
                if model_config is None: # If ckpt was a directory or peeking failed
                    # Need to load config separately if saved alongside weights in dir
                    config_path = os.path.join(FLAGS.ckpt, 'config.json') # Example path
                    if os.path.exists(config_path):
                        model_config = gemma_config.GemmaConfig.from_json_file(config_path) # Assumes method exists
                    else: # Fallback to base config if no saved config found
                        print("Warning: Could not find saved config for ISP-KV model. Using base config.")
                        model_config = get_base_config(FLAGS.variant, FLAGS.device)
                        # Manually add expected ranks if needed (might be inaccurate)
                        model_config.r_k = FLAGS.r_k
                        model_config.r_v = FLAGS.r_v

                # Ensure tokenizer path is set
                model_config.tokenizer = getattr(model_config, 'tokenizer', FLAGS.tokenizer_path)

                # Instantiate the ISP-KV model
                model = GemmaForCausalLMwithISP_KV(model_config)

                # Load weights (expects original W_q/k/v/o + V_r/Z_v buffers)
                model.load_weights(FLAGS.ckpt) # Use the model's loading method

                model = model.to(device=torch_device, dtype=compute_dtype).eval()
                model_type = "ISP-KV (Loaded)"
                print("ISP-KV model loaded successfully.")
            except Exception as e:
                print(f"ERROR loading ISP-KV model from {FLAGS.ckpt}: {e}")
                print("Attempting to load as standard GQA model instead.")
                load_as_isp_kv = False # Force fallback
                model = None # Reset model
                gc.collect()
                if torch_device.type == 'cuda': torch.cuda.empty_cache()

        # Load as standard GQA if not converting, forced, or ISP-KV load failed/skipped
        if not load_as_isp_kv:
            if FLAGS.force_gqa: print("Forcing load as standard GQA model.")
            print(f"Loading standard Gemma GQA model from {FLAGS.ckpt}...")
            try:
                model_config = get_base_config(FLAGS.variant, FLAGS.device)
                model_config.quant = FLAGS.quant
                model_config.tokenizer = FLAGS.tokenizer_path
                model = gemma_model.GemmaForCausalLM(model_config)
                model.load_weights(FLAGS.ckpt)
                model = model.to(torch_device, dtype=compute_dtype).eval()
                model_type = "Standard GQA"
                print("Standard GQA model loaded successfully.")
            except Exception as e:
                print(f"ERROR loading standard GQA model: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)

    if model is None:
        print("ERROR: Model could not be loaded or converted.")
        sys.exit(1)

    load_time_end = time()
    print(f"Model ({model_type}) is on device={torch_device} (dtype={model.dtype}), ready.")
    print(f"Load/Conversion time: {load_time_end - load_start_time:.2f} seconds")

    # --- Inference ---
    print(f"\n--- Starting Inference ---")
    print(f"Prompt: \"{FLAGS.prompt}\"")
    print(f"Settings: temp={FLAGS.temperature}, top_p={FLAGS.top_p}, top_k={FLAGS.top_k}, max_tokens={FLAGS.output_len}")

    # Set default dtype for generation consistency
    with _set_default_tensor_type(model.dtype):
        generate_start_time = time()
        try:
            results = model.generate(
                prompts=[FLAGS.prompt], # Pass as a list
                device=torch_device,
                max_tokens=FLAGS.output_len, # Use max_tokens argument
                temperature=FLAGS.temperature if FLAGS.temperature > 0 else None, # Pass None for greedy
                top_p=FLAGS.top_p,
                top_k=FLAGS.top_k,
            )
            output_text = results[0] # Get the first result from the list

        except Exception as e:
            print(f"\nERROR during generation: {e}")
            import traceback
            traceback.print_exc()
            output_text = "[Generation Failed]"

        generate_end_time = time()

    # --- Print Results ---
    print("\n" + "="*60)
    print("Generation Output:")
    print("="*60)
    print(output_text)
    print("="*60 + "\n")

    generation_time = generate_end_time - generate_start_time
    # Simple token counting (use tokenizer if available)
    num_output_tokens = 0
    if model.tokenizer:
        num_output_tokens = len(model.tokenizer.encode(output_text))
    else:
        num_output_tokens = len(output_text.split()) # Fallback estimate

    tokens_per_sec = num_output_tokens / generation_time if generation_time > 0 else 0

    print(f"Generation completed in {generation_time:.2f} seconds.")
    print(f"Generated approximately {num_output_tokens} tokens ({tokens_per_sec:.2f} tokens/sec).")

    # --- GPU Memory Usage ---
    if torch_device.type == 'cuda':
        mem_alloc = torch.cuda.memory_allocated(torch_device) / (1024**3)
        mem_resv = torch.cuda.memory_reserved(torch_device) / (1024**3)
        max_mem_alloc = torch.cuda.max_memory_allocated(torch_device) / (1024**3)
        max_mem_resv = torch.cuda.max_memory_reserved(torch_device) / (1024**3)
        print(f"GPU Memory Usage (GB):")
        print(f"  Current Allocated: {mem_alloc:.2f}")
        print(f"  Current Reserved:  {mem_resv:.2f}")
        print(f"  Peak Allocated:    {max_mem_alloc:.2f}")
        print(f"  Peak Reserved:     {max_mem_resv:.2f}")


    print("Run complete.")

if __name__ == '__main__':
    # Ensure required flag 'ckpt' is provided
    flags.mark_flag_as_required('ckpt')
    app.run(main)