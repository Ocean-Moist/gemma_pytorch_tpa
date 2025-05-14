# Copyright 2025
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
"""Command‑line runner for Gemma models compressed with Energy DK‑SVD.

Example:
---------
python scripts/run_edksvd.py \
    --ckpt /path/to/edksvd_checkpoint.pt \
    --variant 1b \
    --qk_rank 32 \
    --device cuda \
    --prompt "Explain grouped‑query attention in 3 sentences." \
    --output_len 64
"""
from __future__ import annotations

import contextlib
import json
import os
import random
from typing import Any

import numpy as np
import torch
from absl import app, flags

from gemma import config as gemma_config
from gemma import model_dksvd as gemma_model_dksvd

# -----------------------------------------------------------------------------
# Flag definitions
# -----------------------------------------------------------------------------
FLAGS = flags.FLAGS

flags.DEFINE_string("ckpt", None, "Path to the Energy DK‑SVD checkpoint.", required=True)
flags.DEFINE_string("variant", "1b", "Model variant (only 1b is currently supported).")
flags.DEFINE_integer("qk_rank", None, "Compressed rank r_g used for Q/K (must match the checkpoint)", required=True)
flags.DEFINE_string("device", "cpu", "Device to run the model on: 'cpu' or 'cuda'.")
flags.DEFINE_integer("output_len", 128, "Number of new tokens to generate.")
flags.DEFINE_integer("seed", 12345, "Random seed for reproducibility.")
flags.DEFINE_boolean("quant", False, "Whether to load 8‑bit quantised weights (if available).")
flags.DEFINE_string("dtype", "float32", "Computation dtype for the model (float32, float16, bfloat16, etc.)")
flags.DEFINE_string("prompt", "Hello, Gemma!", "Input prompt for generation.")
flags.DEFINE_float("temperature", 1.0, "Sampling temperature; use 0 or None for greedy decoding.")
flags.DEFINE_float("top_p", 0.95, "Nucleus (top‑p) sampling threshold.")
flags.DEFINE_integer("top_k", 40, "Top‑k sampling cutoff.")

_VALID_VARIANTS = {"1b"}
_VALID_DEVICES = {"cpu", "cuda"}


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _validate_variant(variant: str) -> bool:
    if variant not in _VALID_VARIANTS:
        raise ValueError(f"Unsupported model variant '{variant}'. Supported variants: {_VALID_VARIANTS}")
    return True


def _validate_device(device: str) -> bool:
    if device not in _VALID_DEVICES:
        raise ValueError(f"Invalid device '{device}'. Choose from {_VALID_DEVICES}.")
    return True


flags.register_validator("variant", _validate_variant)
flags.register_validator("device", _validate_device)


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Temporarily sets the default torch dtype inside the context."""
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(torch.float32)


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def main(_: Any) -> None:
    # ------------------------------------------------------------------
    # 1.  Construct the Gemma configuration
    # ------------------------------------------------------------------
    if FLAGS.variant == "1b":
        model_config: gemma_config.GemmaConfig = gemma_config.get_config_for_1b(dtype=FLAGS.dtype)
    else:
        # Future variants can be added to gemma.config; fall back to generic helper.
        model_config = gemma_config.get_model_config(FLAGS.variant)
        model_config.dtype = FLAGS.dtype

    # Inject Energy DK‑SVD‑specific parameters
    model_config.qk_rank = FLAGS.qk_rank  # r_g
    model_config.quant = FLAGS.quant

    # ------------------------------------------------------------------
    # 2.  Seed RNGs for reproducibility
    # ------------------------------------------------------------------
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    # ------------------------------------------------------------------
    # 3.  Instantiate the model and load weights
    # ------------------------------------------------------------------
    device = torch.device(FLAGS.device)
    dtype = model_config.get_dtype()  # resolves gemma_config → torch dtype

    with _set_default_tensor_type(dtype):
        model = gemma_model_dksvd.GemmaForCausalLMDKSVD(model_config)
        _load_edksvd_weights(model, FLAGS.ckpt)
        model = model.to(device).eval()

    print("Model initialised ✔  |  parameters: {0:.2f} M".format(sum(p.numel() for p in model.parameters()) / 1e6))

    # ------------------------------------------------------------------
    # 4.  Perform generation
    # ------------------------------------------------------------------
    temperature = None if FLAGS.temperature in (0, 0.0, None) else float(FLAGS.temperature)
    from debug_wrap import attach_probes

    with attach_probes(model, every_layer=False):   # set True to dump all 26 layers
        generated = model.generate(
            prompts=FLAGS.prompt,
            device=device,
            output_len=FLAGS.output_len,
            temperature=temperature,
            top_p=float(FLAGS.top_p),
            top_k=int(FLAGS.top_k),
        )

    print("\n=== PROMPT ===\n" + FLAGS.prompt)
    print("\n=== COMPLETION ===\n" + generated)


# -----------------------------------------------------------------------------
# We split the loader into a helper so we can support both single‑file and
# sharded checkpoints produced by convert_weights.py.
# -----------------------------------------------------------------------------

def _load_edksvd_weights(model: torch.nn.Module, ckpt_path: str) -> None:
    """Loads a checkpoint produced by convert_weights.py into the model.

    The function supports both a single `.pt/.bin` file and the Transformers
    style sharded directory with an `index.json` file.
    """
    if os.path.isfile(ckpt_path):
        print(f"Loading single‑file checkpoint from {ckpt_path} …")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]  # support training checkpoints
        missing, unexpected = model.load_state_dict(state, strict=False)
    else:
        index_file = os.path.join(ckpt_path, "pytorch_model.bin.index.json")
        if not os.path.isfile(index_file):
            raise FileNotFoundError(
                f"Could not find a checkpoint at '{ckpt_path}'. "
                "Provide either a single .pt/.bin file or a directory with Transformers sharded weights."
            )
        print(f"Loading sharded checkpoint from directory {ckpt_path} …")
        with open(index_file, "r", encoding="utf‑8") as f:
            index = json.load(f)
        shard_files = list(set(index["weight_map"].values()))
        missing, unexpected = [], []
        for shard in shard_files:
            shard_path = os.path.join(ckpt_path, shard)
            shard_state = torch.load(shard_path, map_location="cpu", weights_only=True)
            m, u = model.load_state_dict(shard_state, strict=False)
            missing.extend(m)
            unexpected.extend(u)
            del shard_state  # free RAM
            torch.cuda.empty_cache()
    missing.remove("local_freqs_cis")
    missing.remove("global_freqs_cis")
    if missing:
        print("⚠  Missing keys when loading checkpoint (keys exist in model but not in checkpoint):")
        for k in missing:
            print("   •", k)
    if unexpected:
        print("⚠  Unexpected keys in checkpoint (ignored):")
        for k in unexpected:
            print("   •", k)


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(main)
