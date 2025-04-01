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
run_tpa.py

Script for:
  1) Loading a standard Gemma GQA model (GemmaForCausalLM) from a checkpoint.
  2) Converting it to SVD-based Tensor Product Attention (SVD-TPA / Constant B-Factor)
     using the `create_tpa_model_from_standard` function if requested.
  3) Saving the converted SVD-TPA model if requested.
  4) Running inference (text-only) with the standard Gemma model or the converted SVD-TPA model.

Usage:
  - Convert GQA to SVD-TPA and save:
      python scripts/run_tpa.py \
        --ckpt /path/to/gemma_model.ckpt \
        --variant 1b \
        --prompt "Write a poem about tensors" \
        --convert \
        --save_tpa /path/to/save/svdtpa_model.pt \
        --q_rank 6 \
        --k_rank 2 \
        --v_rank 2 \
        --device cuda

  - Load an already converted SVD-TPA model and run inference:
      python scripts/run_tpa.py \
        --ckpt /path/to/svdtpa_model.pt \
        --variant 1b \
        --prompt "Explain quantum field theory" \
        --convert=False \
        --device cuda

  - Run inference with the original GQA model (without conversion):
      python scripts/run_tpa.py \
        --ckpt /path/to/gemma_model.ckpt \
        --variant 1b \
        --prompt "What is attention?" \
        --convert=False \
        --device cuda \
        --force_gqa # Flag to ensure standard model is used even if ckpt name suggests TPA

  - Additional settings like temperature, top-p, top-k, etc. can be used.
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

# SVD-TPA modules (Conversion and Model)
# Assuming svdtpa_model.py contains GemmaForCausalLMwithSVDTPA
# and gqa_to_tpa.py contains create_tpa_model_from_standard
try:
    from gemma.tpa.modules.gqa_to_tpa import create_tpa_model_from_standard
    from gemma.tpa.svdtpa_model import GemmaForCausalLMwithSVDTPA
    tpa_modules_available = True
except ImportError as e:
    print(f"Warning: Could not import SVD-TPA modules: {e}")
    print("Conversion to TPA will not be possible.")
    tpa_modules_available = False
    # Define dummy classes if needed to prevent NameErrors later
    class GemmaForCausalLMwithSVDTPA: pass
    def create_tpa_model_from_standard(*args, **kwargs): pass


# ------------------------------------------------------------
# ABSL Flags
# ------------------------------------------------------------

FLAGS = flags.FLAGS

# Model Loading & Conversion
flags.DEFINE_string('ckpt', None, 'Path to the checkpoint file (standard Gemma or converted SVD-TPA).', required=True)
flags.DEFINE_string('variant', '1b', 'Model variant (e.g., 1b, 4b, 12b, etc.). Used to get base config.')
flags.DEFINE_string('device', 'cuda' if torch.cuda.is_available() else 'cpu',
                    'Device to run the model on (cpu or cuda).')
flags.DEFINE_boolean('quant', False, 'Use quantization? (Currently affects standard model loading).') # Note: Conversion currently outputs float
flags.DEFINE_boolean('convert', False, 'Convert standard weights to SVD-TPA? Requires standard ckpt.')
flags.DEFINE_string('save_tpa', None, 'Path to save converted SVD-TPA model weights and config.')
flags.DEFINE_boolean('force_gqa', False, 'Force loading as standard GQA model, even if ckpt seems like TPA.')

# TPA Conversion Parameters
flags.DEFINE_integer('q_rank', 6, 'Target rank for query factorization in SVD-TPA.')
flags.DEFINE_integer('k_rank', 2, 'Target rank for key factorization in SVD-TPA.')
flags.DEFINE_integer('v_rank', 2, 'Target rank for value factorization in SVD-TPA.')
flags.DEFINE_boolean('use_dynamic_ranks', True, 'Allow conversion to dynamically determine ranks based on SVD?')
flags.DEFINE_boolean('fat_ranks', False, 'Use larger fixed ranks (e.g., 240) during conversion?')

# Inference Parameters
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
    dtype = 'float32' if device == 'cpu' else 'bfloat16' # Default compute dtype

    if variant == '1b':
        return gemma_config.get_config_for_1b(dtype=dtype)
    elif variant == '4b':
        return gemma_config.get_config_for_4b(dtype=dtype)
    elif variant == '12b':
        # Assuming configs exist for these
        return gemma_config.get_config_for_12b(dtype=dtype)
    elif variant == '27b':
        return gemma_config.get_config_for_27b(dtype=dtype)
    else:
        print(f"Warning: Unknown variant '{variant}'. Using 1b config as base.")
        return gemma_config.get_config_for_1b(dtype=dtype)


def main(_):
    if FLAGS.cuda_launch_blocking:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # --- Setup ---
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    torch_device = torch.device(FLAGS.device)
    compute_dtype = torch.bfloat16 if torch_device.type == 'cuda' else torch.float32

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
    load_start_time = time()

    # Check if the checkpoint path exists
    if not os.path.exists(FLAGS.ckpt):
        print(f"ERROR: Checkpoint file/directory not found at {FLAGS.ckpt}")
        sys.exit(1)

    # Determine if loading Standard GQA or attempting SVD-TPA
    load_as_svdtpa = False
    if not FLAGS.convert and not FLAGS.force_gqa:
        # Try to load as SVD-TPA if not converting and not forcing GQA
        # Check if the checkpoint might contain a TPA config
        try:
            # Peek into the checkpoint file if it's a single file
            if os.path.isfile(FLAGS.ckpt):
                peek_ckpt = torch.load(FLAGS.ckpt, map_location='cpu', weights_only=False) # Need config
                if isinstance(peek_ckpt, dict) and 'config' in peek_ckpt:
                    if hasattr(peek_ckpt['config'], 'q_rank'): # Check for a TPA-specific attribute
                        print("Checkpoint seems to contain a config with TPA ranks. Attempting to load as SVD-TPA.")
                        load_as_svdtpa = True
                del peek_ckpt
                gc.collect()
            # If it's a directory, assume it's standard for now, unless conversion was explicitly done before.
        except Exception as e:
            print(f"Warning: Could not peek into checkpoint file {FLAGS.ckpt} to determine type: {e}")

    # --- Conversion Path ---
    if FLAGS.convert:
        if not tpa_modules_available:
            print("ERROR: Cannot convert model because SVD-TPA modules are not available.")
            sys.exit(1)

        print(f"\n--- Starting GQA to SVD-TPA Conversion ---")
        print(f"Loading standard Gemma model from {FLAGS.ckpt}...")

        # Get base config and update with quantization flag
        model_config = get_base_config(FLAGS.variant, FLAGS.device)
        model_config.quant = FLAGS.quant # Set quant based on flag (affects standard loading)
        model_config.tokenizer = FLAGS.tokenizer_path # Store tokenizer path in config

        # Load standard model first
        standard_model = gemma_model.GemmaForCausalLM(model_config)
        standard_model.load_weights(FLAGS.ckpt) # Use the model's loading method
        standard_model = standard_model.to(torch_device).eval() # Move to device AFTER loading
        print("Standard GQA model loaded.")

        # Perform conversion using create_tpa_model_from_standard
        print("Converting GQA model to SVD-TPA format...")
        convert_start = time()
        try:
            model = create_tpa_model_from_standard(
                standard_model,
                q_rank=FLAGS.q_rank,
                k_rank=FLAGS.k_rank,
                v_rank=FLAGS.v_rank,
                dtype=compute_dtype, # Use compute dtype for the converted model
                device=FLAGS.device,
                use_dynamic_ranks=FLAGS.use_dynamic_ranks,
                fat_ranks=FLAGS.fat_ranks,
            )
            model_type = "SVD-TPA (Converted)"
            print(f"Conversion successful in {time() - convert_start:.2f} seconds.")

            # Save the converted model if requested
            if FLAGS.save_tpa:
                print(f"Saving converted SVD-TPA model to {FLAGS.save_tpa}...")
                save_dir = os.path.dirname(FLAGS.save_tpa)
                if save_dir: os.makedirs(save_dir, exist_ok=True)
                # Save both state_dict and the updated config
                torch.save({'config': model.config, 'model_state_dict': model.state_dict()}, FLAGS.save_tpa)
                print("SVD-TPA model saved.")

        except Exception as e:
            print(f"ERROR during conversion: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        finally:
            # Clean up standard model
            del standard_model
            gc.collect()
            if torch_device.type == 'cuda': torch.cuda.empty_cache()

    # --- Loading Path (Standard GQA or Pre-converted SVD-TPA) ---
    else:
        if load_as_svdtpa and tpa_modules_available:
            print(f"Loading pre-converted SVD-TPA model from {FLAGS.ckpt}...")
            try:
                # Load checkpoint which should contain config and state_dict
                checkpoint = torch.load(FLAGS.ckpt, map_location='cpu') # Load to CPU first
                if 'config' not in checkpoint or 'model_state_dict' not in checkpoint:
                    raise ValueError("Checkpoint does not contain 'config' and 'model_state_dict'. Cannot load as SVD-TPA.")

                model_config = checkpoint['config']
                # Ensure tokenizer path is set if loading only weights
                model_config.tokenizer = getattr(model_config, 'tokenizer', FLAGS.tokenizer_path)

                # Instantiate the SVD-TPA model
                model = GemmaForCausalLMwithSVDTPA(model_config)
                model.load_state_dict(checkpoint['model_state_dict'], strict=False) # Allow partial loads
                model = model.to(device=torch_device, dtype=compute_dtype).eval()
                model_type = "SVD-TPA (Loaded)"
                print("SVD-TPA model loaded successfully.")
            except Exception as e:
                print(f"ERROR loading SVD-TPA model: {e}")
                print("Falling back to loading as standard GQA model.")
                load_as_svdtpa = False # Force loading as GQA on error
        else:
            # Load as standard GQA model if not converting, forced, or TPA load failed/skipped
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
    # Simple token counting (adjust if using a more precise method)
    num_output_tokens = len(gemma_tok.encode(output_text))
    tokens_per_sec = num_output_tokens / generation_time if generation_time > 0 else 0

    print(f"Generation completed in {generation_time:.2f} seconds.")
    print(f"Generated approximately {num_output_tokens} tokens ({tokens_per_sec:.2f} tokens/sec).")

    # --- GPU Memory Usage ---
    if torch_device.type == 'cuda':
        mem_alloc = torch.cuda.memory_allocated(torch_device) / (1024**3)
        mem_resv = torch.cuda.memory_reserved(torch_device) / (1024**3)
        print(f"GPU Memory: Allocated={mem_alloc:.2f} GB, Reserved={mem_resv:.2f} GB")

    print("Run complete.")

if __name__ == '__main__':
    # Ensure required flag 'ckpt' is provided
    flags.mark_flag_as_required('ckpt')
    app.run(main)