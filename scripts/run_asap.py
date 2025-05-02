#!/usr/bin/env python3
# scripts/run_asap.py

import contextlib
import random
import numpy as np
import torch
from absl import app, flags

from gemma import config as gemma_config
from gemma.model_asap import GemmaASAPForCausalLM

# ---------------------
# CLI Flags
# ---------------------
FLAGS = flags.FLAGS

flags.DEFINE_string('ckpt_asap', None, 'Path to the ASAP-augmented checkpoint.', required=True)
flags.DEFINE_string('variant', '1b', 'Gemma model variant (e.g., 1b, 2b, 7b).')
flags.DEFINE_string('device', 'cpu', 'Device to run on: "cpu" or "cuda"')
flags.DEFINE_integer('rank', 8, 'ASAP rank (e.g., 8, 16, 32)')
flags.DEFINE_string('prompt', 'What is gauge invariance in transformers?', 'Input prompt')
flags.DEFINE_integer('output_len', 128, 'Number of tokens to generate')
flags.DEFINE_integer('seed', 42, 'Random seed')
flags.DEFINE_float('temperature', 0.7, 'Sampling temperature')
flags.DEFINE_float('top_p', 0.95, 'Top-p sampling')
flags.DEFINE_integer('top_k', 64, 'Top-k sampling')

# ---------------------
# Setup helpers
# ---------------------
@contextlib.contextmanager
def _set_default_dtype(dtype: torch.dtype):
    """Temporarily sets torch default dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)

# ---------------------
# Main
# ---------------------
def main(_):
    # Set random seed
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    # Config
    cfg = gemma_config.get_model_config(FLAGS.variant)
    cfg.dtype = 'float32'
    cfg.quant = False  # ASAP can work with quant but load fp32 first
    rank = FLAGS.rank

    # Load model
    device = torch.device(FLAGS.device)
    with _set_default_dtype(cfg.get_dtype()):
        model = GemmaASAPForCausalLM(cfg, rank).to(device).eval()
        model.load_weights_asap(FLAGS.ckpt_asap)

    print(f"✓ Loaded Gemma-ASAP model ({FLAGS.variant}, rank={rank})")

    # Run generation
    result = model.generate(
        prompts=FLAGS.prompt,
        device=device,
        output_len=FLAGS.output_len,
        temperature=FLAGS.temperature,
        top_p=FLAGS.top_p,
        top_k=FLAGS.top_k,
    )

    print("=" * 40)
    print("PROMPT:", FLAGS.prompt)
    print("OUTPUT:", result)
    print("=" * 40)

# ---------------------
if __name__ == '__main__':
    app.run(main)
