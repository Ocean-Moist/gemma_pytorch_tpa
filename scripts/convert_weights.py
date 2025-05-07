#!/usr/bin/env python3
"""convert_weights.py

Utility script that converts an *original* Gemma checkpoint (with full‑size
Q/K/V projections) into a checkpoint compatible with the *Energy‑DK‑SVD*
variant of the model (``gemma.model_dksvd``).

The script assumes that the **architecture of the original checkpoint is the
same** as the target model (only the attention projections change).  For
Gemma‑1B this means:

*   ``num_attention_heads   = 4``
*   ``num_key_value_heads   = 1`` (→ one GQA group)
*   ``head_dim             = 256`` (original Q/K/V width)
*   ``hidden_size          = 1152``

After the conversion each attention layer stores separate *low‑rank* modules
(``q_linears``, ``k_linears``, ``v_linears``) whose weights have the usual
*pytorch* layout ``(out_features, in_features)``.

Only the **Q/K** matrices are re‑parameterised; **V** and **O** projections are
copied verbatim.  In addition, fresh zero‑initialised ``query_norm`` and
``key_norm`` parameters (size = ``rank``) are created for every layer if the
original model used them.

The resulting file has the *same* container structure as the original
(checkpoint == ``torch.save({'model_state_dict': ...})``) so that
``GemmaForCausalLMDKSVD.load_weights`` can read it without changes.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple, List

import torch
from gemma import config as gemma_config

DTYPE_MAP = {
    "float32": torch.float32,
    "fp32":    torch.float32,
    "float16": torch.float16,
    "fp16":    torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16":     torch.bfloat16,
}

# -----------------------------------------------------------------------------
# ----------  E‑DK‑SVD core ----------------------------------------------------
# -----------------------------------------------------------------------------

def _edksvd_factorise(
        W_q_list: List[torch.Tensor],
        W_k_shared: torch.Tensor,
        rank: int,
        eps: float = 1e-12,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Energy DK‑SVD compression for a *single* GQA group.

    Args:
        W_q_list:  list of *s* **original** full‑width query matrices, each
                    shape ``(d, d_k)``.
        W_k_shared: original shared key matrix, shape ``(d, d_k)``.
        rank:       target low rank *r*.
        eps:        minimal eigen‑value clamp for numerical stability.

    Returns:
        W_k_tilde:          optimal shared low‑rank key,  shape ``(d, r)``.
        W_q_tilde_list:     list of *s* optimal low‑rank queries, each
                             ``(d, r)``.
    """

    device = W_k_shared.device
    dtype = W_k_shared.dtype


    # ------------------------------------------------------------------
    # 1.  Interaction kernels A_i = W_q_i  W_k^T  (d × d)
    # ------------------------------------------------------------------
    A_list = [W_q @ W_k_shared.T for W_q in W_q_list]

    # ------------------------------------------------------------------
    # 2.  Energy matrix  C = Σ_i A_i^T A_i  (d × d)
    # ------------------------------------------------------------------
    C = torch.zeros((W_k_shared.shape[0], W_k_shared.shape[0]), dtype=dtype, device=device)
    for A in A_list:
        C = C + A.T @ A

    # ------------------------------------------------------------------
    # 3.  Eigen‑decomposition  C = U Λ U^T   (ascending eigen‑values)
    # ------------------------------------------------------------------
    eigvals, eigvecs = torch.linalg.eigh(C)                 # eigvals: (d,)
    # Select *rank* largest eigen‑pairs (descending order)
    topk = torch.topk(eigvals, k=rank, largest=True, sorted=True)
    lambda_r = topk.values.clamp(min=eps)                   # (r,)
    U_r = eigvecs[:, topk.indices]                          # (d, r)

    # ------------------------------------------------------------------
    # 4.  Optimal shared key   W_k^* = U_r  Λ_r^{1/2}
    # ------------------------------------------------------------------
    sqrt_lambda = torch.sqrt(lambda_r)
    W_k_tilde = U_r * sqrt_lambda.unsqueeze(0)              # broadcast columns

    # ------------------------------------------------------------------
    # 5.  Optimal queries  W_q_i^* = A_i W_k^* Λ_r^{-1}
    # ------------------------------------------------------------------
    inv_lambda = 1.0 / lambda_r                             # (r,)
    W_q_tilde_list: List[torch.Tensor] = []
    for A in A_list:
        W_q_i = (A @ W_k_tilde) * inv_lambda.unsqueeze(0)
        W_q_tilde_list.append(W_q_i)

    return W_k_tilde, W_q_tilde_list


# -----------------------------------------------------------------------------
# ----------  Conversion routine ---------------------------------------------
# -----------------------------------------------------------------------------

def convert_checkpoint(
        input_ckpt: str,
        output_ckpt: str,
        variant: str,
        rank: int,
        dtype: str = "float32",
        eigen_clamp_min: float = 1e-12,
):
    """Main entry: loads ``input_ckpt``, performs E‑DK‑SVD, writes ``output_ckpt``."""

    assert os.path.isfile(input_ckpt), f"Input checkpoint not found: {input_ckpt}"
    os.makedirs(os.path.dirname(os.path.abspath(output_ckpt)), exist_ok=True)

    # ------------------------------------------------------------------
    # 1.  Load model config (for shapes) & constants --------------------
    # ------------------------------------------------------------------
    model_cfg = gemma_config.get_model_config(variant)
    model_cfg.dtype = dtype

    # -----------------------------------------------------------------
    # Inside convert_checkpoint(...)
    # -----------------------------------------------------------------
    if isinstance(dtype, str):
        try:
            dtype = DTYPE_MAP[dtype.lower()]
        except KeyError:
            raise ValueError(
                f"Unknown dtype '{dtype}'. Allowed values: {list(DTYPE_MAP)}"
            )

    d_model      = model_cfg.hidden_size
    d_k_full     = model_cfg.head_dim                      # original Q/K/V dim
    n_heads      = model_cfg.num_attention_heads
    n_kv_heads   = model_cfg.num_key_value_heads
    s_per_group  = n_heads // n_kv_heads                   # queries per group

    # ------------------------------------------------------------------
    # 2.  Load *original* state‑dict -----------------------------------
    # ------------------------------------------------------------------
    raw = torch.load(input_ckpt, map_location="cpu")
    if "model_state_dict" in raw:           # <== common layout from run.py
        orig_state = raw["model_state_dict"]
    else:
        orig_state = raw

    # New state dict we will populate.
    new_state: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # 3.  Copy *unchanged* parameters ----------------------------------
    # ------------------------------------------------------------------
    for key, tensor in orig_state.items():
        # Skip parameters that are re‑factorised / change size.
        if ".self_attn.qkv_proj." in key:
            continue  # replaced by q_linears / k_linears / v_linears
        if ".self_attn.query_norm." in key or ".self_attn.key_norm." in key:
            continue  # dimensions change → freshly initialised later
        new_state[key] = tensor.clone().to(dtype=dtype)

    # ------------------------------------------------------------------
    # 4.  Per‑layer re‑parameterisation --------------------------------
    # ------------------------------------------------------------------
    for layer_idx in range(model_cfg.num_hidden_layers):
        # ---------- Fetch original concatenated QKV weight -------------
        w_qkv_key = f"model.layers.{layer_idx}.self_attn.qkv_proj.weight"
        W_qkv_full = orig_state[w_qkv_key].to(dtype)        # (out, in)  = ((n_h+2*n_kv)*d_k, d_model)
        W_qkv_full_t = W_qkv_full.T.contiguous()            # (d_model, out)

        # Slicing indices ------------------------------------------------
        q_end   = n_heads * d_k_full
        k_end   = q_end + n_kv_heads * d_k_full
        v_end   = k_end + n_kv_heads * d_k_full

        W_Q_all = W_qkv_full_t[:, 0:q_end]                  # (d, n_h*d_k)
        W_K_all = W_qkv_full_t[:, q_end:k_end]              # (d, n_kv*d_k)
        W_V_all = W_qkv_full_t[:, k_end:v_end]              # (d, n_kv*d_k)  (d_k == d_v)

        # ---------- Process each GQA group -----------------------------
        for g in range(n_kv_heads):
            # Shared K & V slices.
            K_slice = slice(g * d_k_full, (g + 1) * d_k_full)
            W_K_shared = W_K_all[:, K_slice]
            W_V_shared = W_V_all[:, K_slice]

            # All queries belonging to this group.
            W_q_list: List[torch.Tensor] = []
            for i in range(s_per_group):
                q_col_start = (g * s_per_group + i) * d_k_full
                q_slice = slice(q_col_start, q_col_start + d_k_full)
                W_q_list.append(W_Q_all[:, q_slice])

            # -------- E‑DK‑SVD ----------------------------------------
            W_K_tilde, W_Q_tilde_list = _edksvd_factorise(
                W_q_list,W_K_shared,rank,eps=eigen_clamp_min)

            # -------- Store new parameters (transpose!) ---------------
            k_weight_key = f"model.layers.{layer_idx}.self_attn.k_linears.{g}.weight"
            new_state[k_weight_key] = W_K_tilde.T.contiguous()

            v_weight_key = f"model.layers.{layer_idx}.self_attn.v_linears.{g}.weight"
            new_state[v_weight_key] = W_V_shared.T.contiguous()

            # Per‑head query linears
            for i, W_q_tilde in enumerate(W_Q_tilde_list):
                q_weight_key = (
                    f"model.layers.{layer_idx}.self_attn.q_linears.{g}.{i}.weight"
                )
                new_state[q_weight_key] = W_q_tilde.T.contiguous()

        # -------- Fresh query/key norm parameters ----------------------
        query_norm_key = f"model.layers.{layer_idx}.self_attn.query_norm.weight"
        key_norm_key   = f"model.layers.{layer_idx}.self_attn.key_norm.weight"
        new_state[query_norm_key] = torch.zeros(rank, dtype=torch.float32)
        new_state[key_norm_key]   = torch.zeros(rank, dtype=torch.float32)

    # ------------------------------------------------------------------
    # 5.  Save ----------------------------------------------------------
    # ------------------------------------------------------------------
    torch.save({"model_state_dict": new_state}, output_ckpt)
    print(f"[✓]  Saved E‑DK‑SVD checkpoint → {output_ckpt}")


# -----------------------------------------------------------------------------
# ----------  CLI -------------------------------------------------------------
# -----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert Gemma checkpoint to E‑DK‑SVD form.")
    p.add_argument("--input_ckpt", required=True, help="Path to *original* Gemma checkpoint (single .pt / .ckpt file).")
    p.add_argument("--output_ckpt", required=True, help="Destination path for converted checkpoint.")
    p.add_argument("--variant", default="1b", choices=["1b"], help="Gemma model variant (only '1b' tested so far).")
    p.add_argument("--rank", type=int, required=True, help="Target low rank r for Q/K projections.")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"], help="Data type for computations & saved tensors.")
    p.add_argument("--eigen_clamp_min", type=float, default=1e-12, help="Smallest eigen‑value allowed before sqrt/inversion.")
    return p.parse_args()


def main():
    args = _parse_args()
    torch.set_grad_enabled(False)

    convert_checkpoint(
        input_ckpt=args.input_ckpt,
        output_ckpt=args.output_ckpt,
        variant=args.variant,
        rank=args.rank,
        dtype=args.dtype,
        eigen_clamp_min=args.eigen_clamp_min,
    )


if __name__ == "__main__":
    main()
