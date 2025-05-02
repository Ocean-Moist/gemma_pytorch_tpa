#!/usr/bin/env python3
"""
convert_weights.py
~~~~~~~~~~~~~~~~~~

Offline utility that augments an *original* Gemma checkpoint with the
rank-`r` ASAP factors described in the “Analytical Spectrum-Aware Projection”
plan.

For **each layer** and **each attention head** it:

    1. extracts the (hidden_size → head_dim) query and key sub-matrices
       hidden inside the big `qkv_proj.weight`;
    2. forms      C = Wqᵀ · Wk                (  d × d  );
    3. runs       U, Σ, Vᵀ = svd(C);
    4. keeps the top `r` singular triplet;
    5. stores per-layer tensors
          • U_r : (n_heads, d, r)
          • V_r : (n_heads, d, r)
          • S_r : (n_heads, r)
       into the new state_dict under keys

          model.layers.{L}.self_attn.U_r
          model.layers.{L}.self_attn.V_r
          model.layers.{L}.self_attn.S_r

The original weights stay unchanged.

---------------------------------------------------------------------------
CLI
---------------------------------------------------------------------------

$ python convert_weights.py \
      --ckpt_in   model.ckpt \
      --ckpt_out  gemma_1b_asap.pt \
      --variant   1b \
      --rank      256

  * `--ckpt_in` may be
        • a single *.pt / *.bin* file   **or**
        • a directory that contains the usual HF shard files +
          `pytorch_model.bin.index.json`.
  * `--rank` may also be set implicitly with `--energy 0.999`
      (retain just enough singular values so that
       Σ₁…r² / Σ₁…d² ≥ energy).

Requires **PyTorch ≥ 1.12**.

---------------------------------------------------------------------------
"""

from __future__ import annotations
import argparse, json, math, os, sys, gc, tempfile
from pathlib import Path
from typing import Dict, Tuple

import torch
from gemma import config as gemma_cfg


# -------------------------------------------------------------------------
# Helpers -----------------------------------------------------------------
# -------------------------------------------------------------------------
def _load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    """Loads either a single pt/bin file or a HF sharded directory."""
    if path.is_file():
        print(f"- reading single checkpoint file  {path}")
        return torch.load(path, mmap=True, weights_only=True)
    if not path.is_dir():
        raise FileNotFoundError(path)

    index_file = path / "pytorch_model.bin.index.json"
    if not index_file.exists():
        raise FileNotFoundError(
            f"could not find weight index json in {path}")

    print(f"- reading HF sharded checkpoint from   {path}")
    with index_file.open() as f:
        weight_map = json.load(f)["weight_map"]

    out: Dict[str, torch.Tensor] = {}
    for shard in sorted(set(weight_map.values())):
        shard_path = path / shard
        print(f"    loading shard {shard_path.name}")
        out.update(torch.load(shard_path, mmap=True, weights_only=True))
    return out


def _save_state_dict(state: Dict[str, torch.Tensor], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": state}, out_path)
    print(f"\n✓ wrote ASAP checkpoint → {out_path}")


# -------------------------------------------------------------------------
# Core --------------------------------------------------------------------
# -------------------------------------------------------------------------
def compute_asap_factors(
        state: Dict[str, torch.Tensor],
        cfg: gemma_cfg.GemmaConfig,
        rank: int,
        target_energy: float | None,
        dtype_out: torch.dtype,
):
    """
    Mutates *state* by inserting U_r / V_r / S_r tensors.
    """

    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    n_kv     = cfg.num_key_value_heads
    d_model  = cfg.hidden_size
    d        = cfg.head_dim
    n_q_per_kv = n_heads // n_kv

    print(
        f"\n=== Computing ASAP factors  "
        f"(layers={n_layers}, heads={n_heads}, d={d}, rank={rank}) ==="
    )

    for layer_idx in range(n_layers):
        print(state.keys())
        state = state["state_dict"]
        key_prefix = f"model.layers.{layer_idx}.self_attn.qkv_proj.weight"
        if key_prefix not in state:
            raise KeyError(f"expected key '{key_prefix}' in state_dict")
        W_qkv = state[key_prefix].float()                     # (out, in)

        q_rows = n_heads * d
        k_rows = n_kv    * d

        Wq_all = W_qkv[0:q_rows, :]                # (q_rows, in)
        Wk_all = W_qkv[q_rows:q_rows + k_rows, :]

        # allocate per-layer tensors for all heads
        U_r_layer = torch.zeros(n_heads, d, rank, dtype=dtype_out)
        V_r_layer = torch.zeros(n_heads, d, rank, dtype=dtype_out)
        S_r_layer = torch.zeros(n_heads, rank,   dtype=dtype_out)

        for h in range(n_heads):
            q_start = h * d
            q_end   = q_start + d
            kv_h    = h // n_q_per_kv
            k_start = kv_h * d
            k_end   = k_start + d

            Wq  = Wq_all[q_start:q_end, :].T.contiguous()    # (in , d)
            Wk  = Wk_all[k_start:k_end, :].T.contiguous()    # (in , d)
            C   = Wq.T @ Wk                                  # (d , d)

            # full SVD
            U, S, Vh = torch.linalg.svd(C, full_matrices=False)

            if target_energy is not None:
                # pick smallest r s.t. cumulative energy ≥ target
                total = (S * S).sum()
                cum   = torch.cumsum(S * S, dim=0)
                rank  = int(torch.searchsorted(cum, total * target_energy).item() + 1)

            U_r_layer[h] = U[:, :rank].to(dtype_out)
            V_r_layer[h] = Vh.T[:, :rank].to(dtype_out)
            S_r_layer[h] = S[:rank].to(dtype_out)

        # add to state_dict
        base_key = f"model.layers.{layer_idx}.self_attn"
        state[f"{base_key}.U_r"] = U_r_layer
        state[f"{base_key}.V_r"] = V_r_layer
        state[f"{base_key}.S_r"] = S_r_layer

        print(f"  layer {layer_idx:>2}:  stored  "
              f"U_r/V_r/S_r  shapes = "
              f"{tuple(U_r_layer.shape)}, {tuple(V_r_layer.shape)}, {tuple(S_r_layer.shape)}")

        # free memory
        del W_qkv, Wq_all, Wk_all, U_r_layer, V_r_layer, S_r_layer
        gc.collect()


# -------------------------------------------------------------------------
# Main --------------------------------------------------------------------
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ckpt_in",  required=True, type=Path,
                        help="Path to original Gemma checkpoint (file or dir).")
    parser.add_argument("--ckpt_out", required=True, type=Path,
                        help="Where to write the ASAP-augmented checkpoint (*.pt).")
    parser.add_argument("--variant",  required=True, choices=["1b","2b","2b-v2","7b","9b","27b"],
                        help="Gemma variant (for hidden-size / head-count).")
    parser.add_argument("--rank",     type=int, default=8,
                        help="Fixed rank r to keep (ignored if --energy is given).")
    parser.add_argument("--energy",   type=float,
                        help="If set, ignore --rank and keep as many singular values "
                             "as needed so cumulative energy ≥ this fraction (e.g. 0.999).")
    parser.add_argument("--dtype",    default="float16", choices=["float32","float16"],
                        help="Precision used to store U_r / V_r / S_r in the new file.")
    args = parser.parse_args()

    if args.energy is not None and not (0.0 < args.energy < 1.0):
        parser.error("--energy must be in (0,1)")

    cfg = gemma_cfg.get_model_config(args.variant)
    dtype_out = torch.float32 if args.dtype == "float32" else torch.float16

    state = _load_state_dict(args.ckpt_in)
    compute_asap_factors(
        state,
        cfg,
        rank=args.rank,
        target_energy=args.energy,
        dtype_out=dtype_out,
    )
    _save_state_dict(state, args.ckpt_out)


if __name__ == "__main__":
    main()
