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
  1) Loading a standard Gemma GQA model (GemmaForCausalLM) from a checkpoint
  2) Converting it to Tensor Product Attention (TPA) if requested
  3) Running inference (text-only) with the resulting TPA model

Usage:
  - Convert from GQA to TPA:
      python scripts/run_tpa.py \
        --ckpt /path/to/gemma_model.ckpt \
        --variant 1b \
        --prompt "Write a poem about mathematics" \
        --convert \
        --save_tpa /path/to/save/tpa_model.pt \
        --q_rank 6 \
        --k_rank 2 \
        --v_rank 2 \
        --device cuda

  - Load an already converted TPA model and run inference:
      python scripts/run_tpa.py \
        --ckpt /path/to/tpa_model.pt \
        --variant 1b \
        --prompt "Explain quantum mechanics" \
        --convert=False \
        --device cuda

  - Additional settings like temperature, top-p, top-k, etc. can be used.
"""

import contextlib
import random
import os
from time import time

from absl import app
from absl import flags
import numpy as np
import torch
import tqdm

# Gemma modules
from gemma import config as gemma_config
from gemma import model as gemma_model
from gemma import tokenizer as gemma_tokenizer

# gqa_to_tpa converts GQA to TPA
from gemma.tpa.modules.gqa_to_tpa import (
    convert_gqa_model_to_tpa,
    create_tpa_model_from_standard,
)

# The new TPA-based class (text-only):
from gemma.tpa.gemma3_tpa_model import GemmaForCausalLMwithTPA

# ------------------------------------------------------------
# ABSL Flags
# ------------------------------------------------------------

FLAGS = flags.FLAGS

flags.DEFINE_string('ckpt', None, 'Path to the checkpoint file.', required=True)
flags.DEFINE_string('variant', '1b', 'Model variant (e.g., 1b, 4b, 12b, etc.).')
flags.DEFINE_string('device', 'cuda' if torch.cuda.is_available() else 'cpu',
                    'Device to run the model on (cpu or cuda).')
flags.DEFINE_integer('output_len', 100, 'Max number of tokens to generate.')
flags.DEFINE_integer('seed', 12345, 'Random seed.')
flags.DEFINE_boolean('quant', False, 'Use quantization?')
flags.DEFINE_boolean('convert', True, 'Convert standard weights to TPA?')
flags.DEFINE_string('save_tpa', None, 'Path to save converted TPA weights.')
flags.DEFINE_boolean('cuda_launch_blocking', False,
                     'Set CUDA_LAUNCH_BLOCKING=1 for sync debugging?')
flags.DEFINE_integer('q_rank', 6, 'Rank for query factorization in TPA.')
flags.DEFINE_integer('k_rank', 2, 'Rank for key factorization in TPA.')
flags.DEFINE_integer('v_rank', 2, 'Rank for value factorization in TPA.')
flags.DEFINE_string('prompt', 'What are large language models?',
                    'Input prompt for the model.')
flags.DEFINE_float('temperature', 0.9, 'Temperature for sampling.')
flags.DEFINE_float('top_p', 0.95, 'Top-p sampling parameter.')
flags.DEFINE_integer('top_k', 64, 'Top-k sampling parameter.')
flags.DEFINE_string('tokenizer_path', 'tokenizer/tokenizer.model',
                    'Path to tokenizer model.')
flags.DEFINE_string('extra_config', None,
                    'JSON string for extra config. E.g. \'{"use_tensorly": true}\'.')
flags.DEFINE_boolean('fat_ranks', False,
                     'Whether to use larger ranks (240) for higher accuracy.')


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    orig = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(orig)


def main(_):
    # Possibly enable CUDA launch blocking if debug
    if FLAGS.cuda_launch_blocking:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    print(f"Running Gemma variant={FLAGS.variant}, TPA config: "
          f"q_rank={FLAGS.q_rank}, k_rank={FLAGS.k_rank}, v_rank={FLAGS.v_rank}")

    # Construct model config from gemma_config
    # (Here we do a minimal approach; you may have custom config getters.)
    # We handle some known model sizes:
    variant = FLAGS.variant.lower()
    if variant == '1b':
        model_config = gemma_config.get_config_for_1b(
            dtype='float32' if FLAGS.device == 'cpu' else 'bfloat16')
    elif variant == '4b':
        model_config = gemma_config.get_config_for_4b(
            dtype='float32' if FLAGS.device == 'cpu' else 'bfloat16')
    elif variant == '12b':
        model_config = gemma_config.get_config_for_12b(
            dtype='float32' if FLAGS.device == 'cpu' else 'bfloat16')
    elif variant == '27b':
        model_config = gemma_config.get_config_for_27b(
            dtype='float32' if FLAGS.device == 'cpu' else 'bfloat16')
    else:
        # fallback or custom
        model_config = gemma_config.get_config_for_1b(
            dtype='float32' if FLAGS.device == 'cpu' else 'bfloat16'
        )

    # Set TPA ranks
    model_config.q_rank = FLAGS.q_rank
    model_config.k_rank = FLAGS.k_rank
    model_config.v_rank = FLAGS.v_rank
    model_config.quant = FLAGS.quant

    # If no sliding window size was set, let's default it
    if not hasattr(model_config, 'sliding_window_size') or model_config.sliding_window_size is None:
        print("No sliding_window_size in config, setting to 4096 by default.")
        model_config.sliding_window_size = 4096

    # seed
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    device = torch.device(FLAGS.device)

    # Load the tokenizer
    print(f"Loading tokenizer from {FLAGS.tokenizer_path}...")
    gemma_tok = gemma_tokenizer.Tokenizer(FLAGS.tokenizer_path)
    # store in config
    model_config.tokenizer = FLAGS.tokenizer_path

    start_time = time()

    # Decide convert vs not
    if FLAGS.convert:
        # We load the standard GemmaForCausalLM, then convert to TPA
        print(f"Loading standard Gemma model from {FLAGS.ckpt} for conversion to TPA...")

        # Load standard checkpoint
        checkpoint = torch.load(FLAGS.ckpt, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            checkpoint_sd = checkpoint['model_state_dict']
        else:
            checkpoint_sd = checkpoint

        # Create standard model
        standard_model = gemma_model.GemmaForCausalLM(model_config)
        standard_model.load_state_dict(checkpoint_sd, strict=False)
        standard_model = standard_model.to(device).eval()
        print("Standard model loaded & on device.")

        # Convert to TPA
        convert_start = time()

        # We'll do it in two steps:
        #  (a) Create TPA model instance
        #  (b) Factorize weights with either gqa_to_tpa or "create_tpa_model_from_standard"

        tpa_model = GemmaForCausalLMwithTPA(model_config).to(device).eval()

        # For GQA -> TPA specifically, we can do more specialized factorization
        try:
            # If we want a specialized GQA to TPA approach, we do so:
            # We'll leverage create_tpa_model_from_standard(...) from gqa_to_tpa
            # Possibly with or without dynamic ranks
            import json
            extra_conf = {}
            if FLAGS.extra_config:
                try:
                    extra_conf = json.loads(FLAGS.extra_config)
                except Exception as e:
                    print(f"Warning: could not parse extra_config: {e}")

            use_dynamic_ranks = extra_conf.get('use_dynamic_ranks', True)
            fat_ranks = FLAGS.fat_ranks

            # Actually create new TPA model that copies over non-attention weights & does factorization
            new_tpa_model = create_tpa_model_from_standard(
                standard_model,
                q_rank=FLAGS.q_rank,
                k_rank=FLAGS.k_rank,
                v_rank=FLAGS.v_rank,
                dtype=tpa_model.dtype,
                device=device,
                use_dynamic_ranks=use_dynamic_ranks,
                fat_ranks=fat_ranks,
            )
            del tpa_model
            torch.cuda.empty_cache()
            tpa_model = new_tpa_model
            print("Converted GQA -> TPA successfully.")
        except Exception as e:
            print(f"Error in specialized GQA->TPA conversion: {e}")
            print("Falling back to direct gqa_to_tpa.convert_gqa_model_to_tpa(...)")
            tpa_model = tpa_model.to(device)
            tpa_model.eval()
            # We can attempt a simpler approach, but if we have partial conflicts, the user might handle them.

            convert_gqa_model_to_tpa(
                model=standard_model,
                q_rank=FLAGS.q_rank,
                k_rank=FLAGS.k_rank,
                v_rank=FLAGS.v_rank,
                dtype=tpa_model.dtype,
                device=str(device),
                use_dynamic_ranks=True,
                fat_ranks=FLAGS.fat_ranks,
            )
            # copy over final weights
            # This step depends on how the function modifies the standard_model in-place
            # so we can also do a manual copy if needed.

        convert_end = time()
        print(f"Conversion to TPA took {convert_end - convert_start:.2f} seconds")

        # Save TPA if needed
        if FLAGS.save_tpa:
            print(f"Saving TPA model to {FLAGS.save_tpa} ...")
            save_dir = os.path.dirname(FLAGS.save_tpa)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            torch.save({'model_state_dict': tpa_model.state_dict(), 'config': model_config}, FLAGS.save_tpa)

            print("TPA model saved successfully.")

        # Free the standard model from memory
        del standard_model
        torch.cuda.empty_cache()

        model = tpa_model

    else:
        # Load an already TPA converted model from checkpoint
        print(f"Loading existing TPA model from {FLAGS.ckpt}...")
        checkpoint = torch.load(FLAGS.ckpt, map_location='cpu')

        # If it has a config, we might override
        if 'config' in checkpoint:
            model_config = checkpoint['config']
            print("Using config from checkpoint for TPA model.")
        # Create TPA model
        model = GemmaForCausalLMwithTPA(model_config)
        if 'model_state_dict' in checkpoint:
            sd = checkpoint['model_state_dict']
        else:
            sd = checkpoint
        model.load_state_dict(sd, strict=False)

        model = model.to(device).eval()

    load_time = time()
    print(f"Model is on device={device} and ready (load time {load_time - start_time:.2f}s).")

    # Now run inference
    print(f"Inference with temperature={FLAGS.temperature}, top_p={FLAGS.top_p}, top_k={FLAGS.top_k} ...")

    generate_start = time()

    # We'll do a single prompt or a list
    user_prompt = FLAGS.prompt
    if user_prompt.strip() == "":
        user_prompt = "Hello world."

    # simple usage of generate
    output_text = model.generate(
        prompts=user_prompt,
        device=device,
        max_tokens=FLAGS.output_len,
        temperature=FLAGS.temperature,
        top_p=FLAGS.top_p,
        top_k=FLAGS.top_k,
    )

    generate_end = time()

    # Print results
    print("\n" + "="*50)
    print(f"PROMPT: {user_prompt}")
    print(f"RESULT: {output_text}")
    print("="*50 + "\n")

    generation_time = generate_end - generate_start
    total_tokens = len(output_text.split())
    tokens_per_s = total_tokens / generation_time if generation_time > 0 else 0
    print(f"Generation took {generation_time:.2f}s for {total_tokens} tokens => {tokens_per_s:.2f} tokens/s")

    # If on CUDA, show memory usage
    if device.type == 'cuda':
        mem_alloc = torch.cuda.memory_allocated(device) / (1024**3)
        mem_resv = torch.cuda.memory_reserved(device) / (1024**3)
        print(f"GPU Memory: allocated={mem_alloc:.2f}GB, reserved={mem_resv:.2f}GB")

    # Possibly show TPA KV cache stats
    # We can estimate:
    #   standard KV size: 2 * batch_size * seq_len * num_heads * head_dim * bytes
    #   TPA KV size: (k_rank + v_rank)*(num_heads + head_dim) * ...
    # but that is optional to print.


if __name__ == '__main__':
    app.run(main)
