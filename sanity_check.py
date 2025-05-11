#!/usr/bin/env python3
"""sanity_check.py  ──  Behaviour-level parity test for an Energy-DK-SVD
compressed Gemma checkpoint.

The script feeds **exactly the same prompt** into

  • the *reference* Gemma-1B model (full-width Q/K/V)
  • the *Energy-DK-SVD* variant that has been converted from that
    checkpoint at rank *r* (often r = 256 for a no-op conversion).

It then reports L-∞ and L-2 statistics of the difference between the two
logit tensors.  For a *successful* conversion with r = 256 you should see
max |Δ| ≲ 1e-4 (pure floating-point round-off).

If you supply a *smaller* rank (e.g. 64) the numbers tell you the true
Frobenius reconstruction loss expressed at the output layer – a good
reference point for regression tests.

Usage
-----
python sanity_check.py \
    --orig_ckpt   model.ckpt \
    --edksvd_ckpt gemma_edksvd.pt \
    --rank        256          \
    --prompt      "1+1="       \
    --device      cuda
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import sys

import torch

from gemma import config as gemma_config
from gemma import model as gemma_ref
from gemma import model_dksvd as gemma_edk

# -----------------------------------------------------------------------------
# helper utilities
# -----------------------------------------------------------------------------

def _build_kv_caches(cfg: gemma_config.gemmaconfig, seq_len: int, batch: int, *, device: torch.device) -> list[tuple[torch.tensor, torch.tensor]]:
    """allocates empty kv caches that match *cfg* (key width = head-dim or r)."""
    dtype = cfg.get_dtype()
    k_width = cfg.qk_rank if getattr(cfg, "qk_rank", None) else cfg.head_dim
    caches: list[tuple[torch.tensor, torch.tensor]] = []
    for _ in range(cfg.num_hidden_layers):
        k = torch.zeros(batch, seq_len, cfg.num_key_value_heads, k_width, dtype=dtype, device=device)
        v = torch.zeros(batch, seq_len, cfg.num_key_value_heads, cfg.head_dim, dtype=dtype, device=device)
        caches.append((k, v))
    return caches


def _parse_args() -> argparse.namespace:
    p = argparse.argumentparser("compare vanilla gemma checkpoint against an energy-dk-svd converted one.")
    p.add_argument("--orig_ckpt", required=True, help="path to the *original* gemma checkpoint (model.ckpt or directory).")
    p.add_argument("--edksvd_ckpt", required=True, help="path to the converted checkpoint produced by convert_weights.py")
    p.add_argument("--rank", type=int, default=256, help="qk_rank that was used during conversion (256 ⇒ exact match)")
    p.add_argument("--prompt", default="hello, gemma!", help="prompt to feed into both models.")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"], help="computation dtype")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="device to run the check on.")
    p.add_argument("--atol", type=float, default=1e-4, help="fail if max |δ| exceeds this absolute tolerance (only when rank == 256)")
    return p.parse_args()

def _load_edksvd_weights(model: torch.nn.Module, ckpt: str) -> None:
    # identical to scripts/run_edksvd.py::_load_edksvd_weights  (we inline it)
    if os.path.isfile(ckpt):
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
    else:
        index_path = os.path.join(ckpt, "pytorch_model.bin.index.json")
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        for shard in set(index["weight_map"].values()):
            shard_state = torch.load(os.path.join(ckpt, shard),
                                     map_location="cpu", weights_only=True)
            model.load_state_dict(shard_state, strict=False)
            del shard_state; gc.collect()


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    torch.set_grad_enabled(False)

    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # 1  instantiate configs & models
    # ------------------------------------------------------------------
    cfg_orig = gemma_config.get_config_for_1b(dtype=args.dtype)
    cfg_edk = copy.deepcopy(cfg_orig)
    cfg_edk.qk_rank = args.rank

    # original full-width model ----------------------------------------
    model_orig = gemma_ref.gemmaforcausallm(cfg_orig).to(device).eval()
    model_orig.load_weights(args.orig_ckpt)

    # energy-dk-svd model ----------------------------------------------
    model_edk = gemma_edk.gemmaforcausallmdksvd(cfg_edk).to(device).eval()
    model_edk.load_weights(args.edksvd_ckpt)

    tok = model_orig.tokenizer

    # ------------------------------------------------------------------
    # 2  prepare inputs common to both models
    # ------------------------------------------------------------------
    input_ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)  # (1, t)
    seq_len = input_ids.size(1)

    input_positions = torch.arange(seq_len, dtype=torch.long, device=device)
    output_positions = torch.tensor([seq_len - 1], dtype=torch.long, device=device)

    mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)

    top_ps = torch.tensor([1.0], device=device)
    top_ks = torch.tensor([1], dtype=torch.long, device=device)

    # ------------------------------------------------------------------
    # 3  run forwards
    # ------------------------------------------------------------------
    kv_caches_orig = _build_kv_caches(cfg_orig, seq_len, 1, device=device)
    kv_caches_edk  = _build_kv_caches(cfg_edk,  seq_len, 1, device=device)

    logits_orig = model_orig(
        input_token_ids=input_ids,
        input_positions=input_positions,
        kv_write_indices=input_positions,
        kv_caches=kv_caches_orig,
        mask=mask,
        output_positions=output_positions,
        temperatures=None,
        top_ps=top_ps,
        top_ks=top_ks,
        local_mask=None,
    )[1]  # (1, v)

    logits_edk = model_edk(
        input_token_ids=input_ids,
        input_positions=input_positions,
        kv_write_indices=input_positions,
        kv_caches=kv_caches_edk,
        mask=mask,
        output_positions=output_positions,
        temperatures=None,
        top_ps=top_ps,
        top_ks=top_ks,
        local_mask=None,
    )[1]

    # ------------------------------------------------------------------
    # 4  report statistics
    # ------------------------------------------------------------------
    delta = (logits_orig - logits_edk).abs()
    max_err = delta.max().item()
    l2_err  = math.sqrt(delta.pow(2).mean().item())

    print("\n=== sanity_check.py results ===")
    print(f"prompt             : '{args.prompt}'")
    print(f"rank (qk_rank)     : {args.rank}")
    print(f"device / dtype     : {device} / {args.dtype}")
    print("----------------------------------------")
    print(f"max |Δ logits|     : {max_err:.6e}")
    print(f"rms |Δ logits|     : {l2_err: .6e}")

    if args.rank == cfg_orig.head_dim:
        # exact-match case
        if max_err > args.atol:
            print(
                f"❌  FAILED  –  max error {max_err:.2e} exceeds atol {args.atol}\n"
                "   The conversion is NOT an exact no-op.  Investigate the RMS-Norm or RoPE steps."
            )
            sys.exit(1)
        else:
            print("✅  PASSED  –  converted model reproduces the original within tolerance.")
    else:
        print("ℹ️   Non-trivial rank selected – the errors reflect legitimate compression loss.")


if __name__ == "__main__":
    main()
