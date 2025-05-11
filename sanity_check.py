#!/usr/bin/env python3
"""
sanity_check.py  – compare logits of the *original* Gemma checkpoint and an
Energy-DK-SVD conversion at **r = 256** (no-op rank).

The script prints the max / RMS absolute difference of the two logit
vectors for the last token of the prompt and exits ≠ 0 if the max delta is
≥ 1e-4  (tweak the threshold with --tol).
"""
from __future__ import annotations
import argparse, copy, json, os, sys, torch, gc
from typing import Any, Tuple

# --------------------------------------------------------------------- helpers
def _load_original_weights(model: torch.nn.Module, ckpt: str) -> None:
    if os.path.isfile(ckpt):
        state = torch.load(ckpt, mmap=True, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
    else:                                  # sharded HF directory
        index_path = os.path.join(ckpt, "pytorch_model.bin.index.json")
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        for shard in set(index["weight_map"].values()):
            shard_state = torch.load(os.path.join(ckpt, shard),
                                     map_location="cpu", weights_only=True)
            model.load_state_dict(shard_state, strict=False)
            del shard_state; gc.collect()

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

# --------------------------------------------------------------------- main
def main(argv: list[str]) -> None:
    # lazy heavy imports only after arg-parse
    from gemma import config as gcfg
    from gemma import model   as gm_ref
    from gemma import model_dksvd as gm_edk

    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_ckpt", required=True, help="Original Gemma checkpoint")
    ap.add_argument("--edk_ckpt",  required=True, help="Converted E-DK-SVD checkpoint")
    ap.add_argument("--prompt",    default="Hello", help="Prompt to test")
    ap.add_argument("--dtype",     default="float32",
                    choices=["float32","bfloat16","float16","fp16","fp32","bf16"])
    ap.add_argument("--tol", type=float, default=1e-4, help="failure threshold on max-abs logit diff")
    args = ap.parse_args(argv)

    # ---------------------------------------------------------------- cfg
    cfg = gcfg.get_config_for_1b(dtype=args.dtype)
    tok = gm_ref.tokenizer.Tokenizer(cfg.tokenizer)
    prompt_ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.int64)

    # ---------------------------------------------------------------- reference
    model_ref = gm_ref.GemmaForCausalLM(copy.deepcopy(cfg)).eval()
    _load_original_weights(model_ref, args.orig_ckpt)

    # ---------------------------------------------------------------- edk (r=256)
    cfg_edk = copy.deepcopy(cfg)
    cfg_edk.qk_rank = cfg.head_dim            # 256 → no-op
    model_edk = gm_edk.GemmaForCausalLMDKSVD(cfg_edk).eval()
    _load_edksvd_weights(model_edk, args.edk_ckpt)

    # ---------------------------------------------------------------- build tiny caches (in-RAM, CPU)
    L  = prompt_ids.size(1)
    kv_ref = [(torch.zeros(1,L,cfg.num_key_value_heads,cfg.head_dim),
               torch.zeros(1,L,cfg.num_key_value_heads,cfg.head_dim))
              for _ in range(cfg.num_hidden_layers)]
    kv_edk = [(torch.zeros(1,L,cfg.num_key_value_heads,cfg_edk.qk_rank),
               torch.zeros(1,L,cfg.num_key_value_heads,cfg.head_dim))
              for _ in range(cfg.num_hidden_layers)]
    mask = torch.zeros(1,1,L,L)

    # ---------------------------------------------------------------- forward once (greedy)
    with torch.no_grad():
        logits_ref = model_ref(
            input_token_ids=prompt_ids,
            input_positions = torch.arange(L),
            kv_write_indices=torch.arange(L),
            kv_caches=kv_ref,
            mask=mask,
            output_positions=torch.tensor([L-1]),
            temperatures=None,
            top_ps=torch.tensor([1.]),
            top_ks=torch.tensor([1]),
        )[1].squeeze(0)           # (V,)

        logits_edk = model_edk(
            input_token_ids=prompt_ids,
            input_positions = torch.arange(L),
            kv_write_indices=torch.arange(L),
            kv_caches=kv_edk,
            mask=mask,
            output_positions=torch.tensor([L-1]),
            temperatures=None,
            top_ps=torch.tensor([1.]),
            top_ks=torch.tensor([1]),
        )[1].squeeze(0)

    diff = (logits_ref - logits_edk).abs()
    print(f"max |Δ| = {diff.max():.5e}   •   RMS |Δ| = {diff.pow(2).mean().sqrt():.5e}")
    if diff.max() >= args.tol:
        print("❌  Sanity check FAILED – conversion is not a no-op at r = 256.")
        sys.exit(1)
    print("✅  Sanity check passed.")
    sys.exit(0)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main(sys.argv[1:])
