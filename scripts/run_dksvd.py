# Copyright 2025
"""Runner for DK‑SVD–compressed Gemma 1b checkpoints.

This script mirrors *scripts/run.py* but targets DK‑SVD‑compressed
checkpoints produced by *scripts/convert_weights.py*.  It:

1.  Loads the compressed checkpoint (which already contains an updated
    `GemmaConfig` with `use_dksvd=True` and `dksvd_rank` set).
2.  Instantiates `gemma.model_dksvd.GemmaForCausalLMDKSVD` and loads the
    weights.
3.  Generates text for a prompt using autoregressive decoding (top‑p/
    top‑k, temperature) — a minimal, single‑batch inference path.

This version purposefully supports **Gemma‑1b** (single GQA group) and
GLOBAL attention only — exactly the subset handled by
*gemma/model_dksvd.py*.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
from typing import Any, Sequence, Union

import numpy as np
import torch
from absl import app, flags

from gemma import config as gemma_config
from gemma import model_dksvd as gemma_model_dksvd

FLAGS = flags.FLAGS

# -----------------------------------------------------------------------------
#  CLI flags
# -----------------------------------------------------------------------------
flags.DEFINE_string("ckpt", None, "Path to the DK‑SVD checkpoint (\n"
                                  "produced by scripts/convert_weights.py).", required=True)
flags.DEFINE_string("variant", "1b", "Model variant. Currently only '1b' is"
                                     " supported.")
flags.DEFINE_string("device", "cpu", "Device to run on: 'cpu' or 'cuda'.")
flags.DEFINE_integer("output_len", 50, "Number of tokens to generate.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_float("temperature", 1.0, "Sampling temperature. Use 0 or None"
                                       " for greedy decoding.")
flags.DEFINE_float("top_p", 0.95, "Nucleus sampling (top‑p) parameter.")
flags.DEFINE_integer("top_k", 64, "Top‑k sampling parameter.")
flags.DEFINE_string("prompt", "Tell me about DK‑SVD compression.",
                    "Prompt to feed into the model.")

_VALID_DEVICES = {"cpu", "cuda"}


# -----------------------------------------------------------------------------
#  Utility helpers
# -----------------------------------------------------------------------------
@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Temporarily sets the default tensor dtype."""
    orig = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(orig)


def _load_config_from_ckpt(ckpt_path: str, fallback_variant: str) -> gemma_config.GemmaConfig:
    """Loads a GemmaConfig from the checkpoint or derives it from the variant."""
    chk = torch.load(ckpt_path, map_location="cpu")
    if "config" in chk:
        cfg = chk["config"]
        # If it was saved as a dict, rebuild GemmaConfig.
        if isinstance(cfg, dict):
            cfg = gemma_config.GemmaConfig(**cfg)
        # Ensure DK‑SVD flags are present.
        cfg.use_dksvd = getattr(cfg, "use_dksvd", True)
        if not cfg.use_dksvd:
            raise ValueError("Checkpoint does not indicate DK‑SVD usage.")
        if getattr(cfg, "dksvd_rank", None) is None:
            raise ValueError("Checkpoint config missing dksvd_rank.")
        return cfg
    # Fallback: use builtin variant config then update flags (rank has to be
    # guessed from metadata file).
    cfg = gemma_config.get_model_config(fallback_variant)
    metadata = chk.get("dksvd_metadata", {})
    rank = metadata.get("dksvd_rank")
    if rank is None:
        raise ValueError("Could not infer dksvd_rank from checkpoint; please"
                         " regenerate checkpoint with proper metadata.")
    cfg.use_dksvd = True
    cfg.dksvd_rank = rank
    # Ensure dtype matches the checkpoint if provided.
    if isinstance(metadata.get("dtype"), str):
        cfg.dtype = metadata["dtype"]
    return cfg


# -----------------------------------------------------------------------------
#  Main generation routine (single‑batch)
# -----------------------------------------------------------------------------

def _generate(
        model: gemma_model_dksvd.GemmaForCausalLMDKSVD,
        prompt: Union[str, Sequence[str]],
        device: torch.device,
        output_len: int,
        temperature: Union[float, None],
        top_p: float,
        top_k: int,
) -> Union[str, Sequence[str]]:
    """Autoregressively generates *output_len* new tokens."""
    is_single = isinstance(prompt, str)
    prompts = [prompt] if is_single else list(prompt)

    # -------------------- tokenization --------------------
    prompt_tokens = [model.tokenizer.encode(p) for p in prompts]
    min_prompt_len = min(len(p) for p in prompt_tokens)
    max_prompt_len = max(len(p) for p in prompt_tokens)
    max_seq_len = max_prompt_len + output_len

    cfg = model.config  # GemmaConfig
    if max_seq_len > cfg.max_position_embeddings:
        raise ValueError(
            f"Requested sequence length {max_seq_len} exceeds model limit "
            f"{cfg.max_position_embeddings}.")

    batch_size = len(prompts)

    # -------------------- build KV caches --------------------
    rank_r = cfg.dksvd_rank
    head_dim = cfg.head_dim  # value projection dim remains 256 for 1b.

    kv_caches = []
    dtype = cfg.get_dtype()
    for _ in range(cfg.num_hidden_layers):
        k_cache = torch.zeros(
            (batch_size, max_seq_len, 1, rank_r), dtype=dtype, device=device
        )
        v_cache = torch.zeros(
            (batch_size, max_seq_len, 1, head_dim), dtype=dtype, device=device
        )
        kv_caches.append((k_cache, v_cache))

    # -------------------- prepare tensors --------------------
    token_ids_tensor = torch.full(
        (batch_size, max_seq_len), model.tokenizer.pad_id, dtype=torch.long
    )
    input_token_ids_tensor = torch.full(
        (batch_size, min_prompt_len), model.tokenizer.pad_id, dtype=torch.long
    )
    for i, tok in enumerate(prompt_tokens):
        token_ids_tensor[i, : len(tok)] = torch.tensor(tok)
        input_token_ids_tensor[i, : min_prompt_len] = torch.tensor(tok[:min_prompt_len])

    token_ids_tensor = token_ids_tensor.to(device)
    input_token_ids_tensor = input_token_ids_tensor.to(device)

    # Causal mask (float32 large negative for masked positions)
    mask_tensor = torch.full(
        (1, 1, max_seq_len, max_seq_len), -2.3819763e38, dtype=torch.float32, device=device
    )
    mask_tensor = torch.triu(mask_tensor, diagonal=1)

    # DK‑SVD implementation currently ignores local sliding masks, so we set to None.
    local_mask_tensor = None

    input_positions_tensor = torch.arange(min_prompt_len, dtype=torch.long, device=device)
    curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)

    output_positions_tensor = torch.tensor([min_prompt_len - 1], dtype=torch.long, device=device)

    temperatures_tensor = None if temperature in (None, 0) else torch.full(
        (batch_size,), float(temperature), dtype=torch.float32, device=device
    )
    top_ps_tensor = torch.full((batch_size,), float(top_p), dtype=torch.float32, device=device)
    top_ks_tensor = torch.full((batch_size,), int(top_k), dtype=torch.long, device=device)

    output_index = torch.tensor(min_prompt_len, dtype=torch.long, device=device)

    # -------------------- prefill (min_prompt_len tokens) --------------------
    model(
        input_token_ids=input_token_ids_tensor,
        input_positions=input_positions_tensor,
        kv_write_indices=input_positions_tensor,  # Ensure caches are written.
        kv_caches=kv_caches,
        mask=curr_mask_tensor,
        output_positions=output_positions_tensor,  # Not used in prefill
        temperatures=temperatures_tensor,
        top_ps=top_ps_tensor,
        top_ks=top_ks_tensor,
        local_mask=local_mask_tensor,
    )

    # After prefill, continue generating one token at a time.
    for _ in range(output_len):
        # Use last token as input.
        input_token_ids_step = token_ids_tensor.index_select(1, output_index - 1)
        input_positions_step = output_index - 1  # scalar tensor

        curr_mask_tensor = mask_tensor.index_select(2, input_positions_step)

        next_token_ids, _ = model(
            input_token_ids=input_token_ids_step,
            input_positions=input_positions_step,
            kv_write_indices=input_positions_step,
            kv_caches=kv_caches,
            mask=curr_mask_tensor,
            output_positions=torch.zeros(1, dtype=torch.long, device=device),
            temperatures=temperatures_tensor,
            top_ps=top_ps_tensor,
            top_ks=top_ks_tensor,
            local_mask=local_mask_tensor,
        )

        # Write generated token into tensor.
        token_ids_tensor.index_copy_(1, output_index, next_token_ids.unsqueeze(1))

        # Advance pointer.
        input_positions_step = output_index
        output_index = output_index + 1

    # -------------------- detokenize --------------------
    results = []
    for i, toks in enumerate(token_ids_tensor.tolist()):
        # Skip the original prompt tokens.
        generated = toks[len(prompt_tokens[i]) : len(prompt_tokens[i]) + output_len]
        if model.tokenizer.eos_id in generated:
            eos_idx = generated.index(model.tokenizer.eos_id)
            generated = generated[:eos_idx]
        results.append(model.tokenizer.decode(generated))

    return results[0] if is_single else results


# -----------------------------------------------------------------------------
#  Main entrypoint
# -----------------------------------------------------------------------------

def main(_):
    if FLAGS.device not in _VALID_DEVICES:
        raise ValueError(f"--device must be one of {_VALID_DEVICES}")

    torch.manual_seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # -------------------- load config & model --------------------
    cfg = _load_config_from_ckpt(FLAGS.ckpt, FLAGS.variant)

    device = torch.device(FLAGS.device)

    with _set_default_tensor_type(cfg.get_dtype()):
        model = gemma_model_dksvd.GemmaForCausalLMDKSVD(cfg)
        model.load_weights(FLAGS.ckpt)
        model.to(device).eval()

    # -------------------- generate --------------------
    output = _generate(
        model=model,
        prompt=FLAGS.prompt,
        device=device,
        output_len=FLAGS.output_len,
        temperature=None if FLAGS.temperature in (None, 0) else FLAGS.temperature,
        top_p=FLAGS.top_p,
        top_k=FLAGS.top_k,
    )

    # -------------------- print --------------------
    print("======================================")
    print(f"PROMPT: {FLAGS.prompt}")
    print(f"OUTPUT: {output}")
    print("======================================")


if __name__ == "__main__":
    app.run(main)
