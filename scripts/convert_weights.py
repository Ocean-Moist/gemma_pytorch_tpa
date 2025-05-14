#!/usr/bin/env python3
"""convert_weights.py (Energy DK‑SVD, *fixed* RMS‑Norm handling)

This utility converts an **original** Gemma checkpoint that uses full‑width
Q/K/V projections **and** (scaled) RMS‑Norm into a checkpoint that is
consumable by ``gemma.model_dksvd``.  Compared to the initial draft, this
version *correctly* projects the per‑channel RMS‑Norm scale vectors onto the
compressed sub‑space, following §5 of the E‑DK‑SVD derivation document.

Key improvements
----------------
*   Projects the original scale vectors (``gamma_Q``, ``gamma_K``) to the new
    width *r* analytically – no longer initialises them to **zero**.
*   Stores the projected vectors as the new ``query_norm`` / ``key_norm``
    parameters so that the runtime model applies them exactly once.
*   Keeps the projection mathematics self‑contained; we do *not* require the
    full eigen‑decomposition outside the helper.

The resulting checkpoint can be loaded with:

>>> cfg = gemma_config.get_config_for_1b(dtype="float32")
>>> cfg.qk_rank = RANK
>>> model = GemmaForCausalLMDKSVD(cfg)
>>> model.load_weights("/path/to/edksvd_checkpoint.pt")
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import torch
from gemma import config as gemma_config

DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}

# -----------------------------------------------------------------------------
# ----------  E‑DK‑SVD core ----------------------------------------------------
# -----------------------------------------------------------------------------

def _edksvd_factorise(
        W_q_list: List[torch.Tensor],
        W_k_shared: torch.Tensor,
        rank: int,
        eps: float = 1.0e-12,
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Energy DK‑SVD for one GQA group *including* eigen‑info.

    Returns
    -------
    W_k_tilde : ``(d, r)``
        Optimal shared *skinny* key.
    W_q_tilde_list : list[Tensor]
        Optimal skinny queries for the *s* heads (each ``(d, r)``).
    lambda_r : ``(r,)``
        Eigen‑values λ_1 … λ_r (already **clamped** ≥ eps).
    U_r : ``(d, r)``
        Corresponding orthonormal eigen‑vectors.
    """
    device, dtype = W_k_shared.device, W_k_shared.dtype

    # 1) A_i = W_q_i  W_k^T
    A_list = [W_q @ W_k_shared.T for W_q in W_q_list]

    # 2) C = Σ_i  A_i^T A_i
    d = W_k_shared.shape[0]
    C = torch.zeros((d, d), dtype=dtype, device=device)
    for A in A_list:
        C.add_(A.T @ A)

    # 3) top‑*r* eigenspace of C – torch.linalg.eigh returns ascending order
    eigvals, eigvecs = torch.linalg.eigh(C)
    lambda_r, idx = torch.topk(eigvals, k=rank, largest=True, sorted=True)
    lambda_r = lambda_r.clamp_min(eps)
    U_r = eigvecs[:, idx]

    # 4) W_K^* = U_r Λ_r^{1/2}
    W_k_tilde = U_r * lambda_r.sqrt().unsqueeze(0)

    # 5) W_Q,i^* = A_i W_K^* Λ_r^{-1}
    inv_lambda = lambda_r.reciprocal().unsqueeze(0)  # broadcast over rows
    W_q_tilde_list = [(A @ W_k_tilde) * inv_lambda for A in A_list]

    return W_k_tilde, W_q_tilde_list, lambda_r, U_r


# -----------------------------------------------------------------------------
# ----------  γ′ projection helper --------------------------------------------
# -----------------------------------------------------------------------------

def _project_rms_scale(
        gamma: torch.Tensor,  # original γ  (d_k,)
        W_k_shared_f: torch.Tensor,  # fused shared K (d, d_k)
        U_r: torch.Tensor,          # (d, r)
        lambda_r: torch.Tensor,     # (r,)
        eps: float = 1e-6,
) -> torch.Tensor:               # returns γ′  (r,)
    """Implements eq. (γ′) from §5.4 of the E‑DK‑SVD doc.

    γ′ = ((Π ⊙ Π)^T (1+γ)) ⊙ Λ^{-1} − 1.
    """
    d_k = gamma.shape[0]
    ones_plus_gamma = 1.0 + gamma  # (d_k,)

    # Π = W_K_f^T U_r    (d_k × r)
    Pi = W_k_shared_f.T @ U_r     # (d_k, r)
    Pi_sq = Pi.pow(2)             # element‑wise square

    # (Π²)^T @ (1+γ)    → (r,)
    gamma_plus_1_prime = (Pi_sq.T @ ones_plus_gamma) / lambda_r

    # numerical safety   (1+γ′) ≥ eps → γ′ ≥ eps−1
    gamma_plus_1_prime = torch.clamp(gamma_plus_1_prime, min=eps)
    gamma_prime = gamma_plus_1_prime - 1.0
    return gamma_prime


# -----------------------------------------------------------------------------
# ----------  Main conversion routine ----------------------------------------
# -----------------------------------------------------------------------------

def convert_checkpoint(
        input_ckpt: str,
        output_ckpt: str,
        variant: str,
        rank: int,
        *,
        dtype: str | torch.dtype = "float32",
        eigen_clamp_min: float = 1.0e-12,
):
    """Converts *input_ckpt* (original Gemma) → *output_ckpt* (E‑DK‑SVD)."""

    if isinstance(dtype, str):
        try:
            dtype_t = DTYPE_MAP[dtype.lower()]
        except KeyError as err:
            raise ValueError(
                f"Unknown dtype '{dtype}'. Choose from {list(DTYPE_MAP)}"
            ) from err
    else:
        dtype_t = dtype

    if not os.path.isfile(input_ckpt):
        raise FileNotFoundError(input_ckpt)
    os.makedirs(os.path.dirname(os.path.abspath(output_ckpt)), exist_ok=True)

    # 1) model config ----------------------------------------------------------------
    if variant == "1b":
        model_cfg: gemma_config.GemmaConfig = gemma_config.get_config_for_1b(dtype="float32")
    else:
        model_cfg = gemma_config.get_model_config(variant)
    model_cfg.dtype = str(dtype) if isinstance(dtype, str) else str(dtype_t)

    d_model = model_cfg.hidden_size
    d_k_full = model_cfg.head_dim          # 256 for Gemma‑1B
    n_heads = model_cfg.num_attention_heads
    n_kv_heads = model_cfg.num_key_value_heads
    s_per_group = n_heads // n_kv_heads

    # 2) load checkpoint --------------------------------------------------------------
    raw = torch.load(input_ckpt, map_location="cpu")
    orig_state: Dict[str, torch.Tensor]
    if isinstance(raw, dict) and "model_state_dict" in raw:
        orig_state = raw["model_state_dict"]
    else:
        orig_state = raw  # sharded‑style single file

    new_state: Dict[str, torch.Tensor] = {}

    # 3) copy *unchanged* tensors -----------------------------------------------------
    SKIP_SUBSTRINGS = {
        ".self_attn.qkv_proj.",
        ".self_attn.query_norm.",
        ".self_attn.key_norm.",
        "freqs_cis",
    }
    for name, tensor in orig_state.items():
        if any(s in name for s in SKIP_SUBSTRINGS):
            continue
        new_state[name] = tensor.to(dtype_t).clone()

    # 4) per‑layer refactorisation ----------------------------------------------------
    for layer_idx in range(model_cfg.num_hidden_layers):
        # original concatenated QKV weight (out_features, in_features)
        qkv_key = f"model.layers.{layer_idx}.self_attn.qkv_proj.weight"
        W_qkv_full = orig_state[qkv_key].to(dtype_t)              # ((n_h+2*n_kv)*d_k, d)
        W_qkv_full_t = W_qkv_full.T.contiguous()                  # (d, out)

        # slices ------------------------------------------------------------------
        q_end = n_heads * d_k_full
        k_end = q_end + n_kv_heads * d_k_full

        W_Q_all = W_qkv_full_t[:, :q_end]                         # (d, n_h*d_k)
        W_K_all = W_qkv_full_t[:, q_end:k_end]                    # (d, n_kv*d_k)
        W_V_all = W_qkv_full_t[:, k_end:k_end + n_kv_heads * d_k_full]  # (d, n_kv*d_k)

        # 4‑a) fuse original RMS scales -------------------------------------
        q_norm_key = f"model.layers.{layer_idx}.self_attn.query_norm.weight"
        k_norm_key = f"model.layers.{layer_idx}.self_attn.key_norm.weight"
        gamma_q = orig_state[q_norm_key].to(dtype_t)             # (d_k,)
        gamma_k = orig_state[k_norm_key].to(dtype_t)             # (d_k,)

        # repeat per head / kv‑head so dimensions align with concatenation
        scale_q_vec = (1.0 + gamma_q).repeat(n_heads)            # (n_h*d_k,)
        scale_k_vec = (1.0 + gamma_k).repeat(n_kv_heads)         # (n_kv*d_k,)

        W_Q_all.mul_(scale_q_vec.unsqueeze(0))
        W_K_all.mul_(scale_k_vec.unsqueeze(0))

        # 4‑b) process each GQA group --------------------------------------
        # For Gemma‑1B there is *exactly* one group; but code supports >1.
        projected_gamma_q_group: List[torch.Tensor] = []
        projected_gamma_k_group: List[torch.Tensor] = []

        for g in range(n_kv_heads):
            K_slice = slice(g * d_k_full, (g + 1) * d_k_full)
            W_K_shared_f = W_K_all[:, K_slice]                   # (d, d_k)
            W_V_shared = W_V_all[:, K_slice]

            # gather queries belonging to this group
            W_q_list = []
            for i in range(s_per_group):
                q_start = (g * s_per_group + i) * d_k_full
                W_q_list.append(W_Q_all[:, q_start:q_start + d_k_full])

            # Energy DK‑SVD --------------------------------------------
            W_K_tilde, W_Q_tilde_list, lambda_r, U_r = _edksvd_factorise(
                W_q_list,
                W_K_shared_f,
                rank,
                eps=eigen_clamp_min,
            )

            # store K / V ------------------------------------------------
            k_weight_key = f"model.layers.{layer_idx}.self_attn.k_linears.{g}.weight"
            new_state[k_weight_key] = W_K_tilde.T.contiguous()

            v_weight_key = f"model.layers.{layer_idx}.self_attn.v_linears.{g}.weight"
            new_state[v_weight_key] = W_V_shared.T.contiguous()

            # store Q_i --------------------------------------------------
            for i, W_q_tilde in enumerate(W_Q_tilde_list):
                q_weight_key = f"model.layers.{layer_idx}.self_attn.q_linears.{g}.{i}.weight"
                new_state[q_weight_key] = W_q_tilde.T.contiguous()

            # γ′ projection for this group ------------------------------
            gamma_q_prime = _project_rms_scale(
                gamma_q,
                W_K_shared_f,
                U_r,
                lambda_r,
            )
            gamma_k_prime = _project_rms_scale(
                gamma_k,
                W_K_shared_f,
                U_r,
                lambda_r,
            )
            # ---- A½ (colour)  — eq. (5-1) & (5-2) ----------------------
            Pi   = W_K_shared_f.T @ U_r                                   # (d_k, r)
            Pi   = Pi * lambda_r.rsqrt().unsqueeze(0)                     # Λ^{-½}

            T_q  = Pi * (1.0 + gamma_q).unsqueeze(1)                      # Γ_Q·…
            T_k  = Pi * (1.0 + gamma_k).unsqueeze(1)                      # Γ_K·…

            A_q  = (T_q.T @ T_q) / d_k_full                               # (r,r)  PSD
            A_k  = (T_k.T @ T_k) / d_k_full

            def _safe_cholesky(M: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
                """Cholesky with auto-jitter for PSD inputs."""
                try:
                    return torch.linalg.cholesky(M)
                except RuntimeError:              # 1st try failed → add jitter
                    jitter = eps * torch.trace(M).div(M.shape[0])
                    M_jit = M + torch.eye(M.shape[0], dtype=M.dtype, device=M.device) * jitter
                    return torch.linalg.cholesky(M_jit)

            A_q_half = _safe_cholesky(A_q).T.contiguous()                 # upper-tri
            A_k_half = _safe_cholesky(A_k).T.contiguous()

            projected_gamma_q_group.append(gamma_q_prime)
            projected_gamma_k_group.append(gamma_k_prime)

            # store A½ ------------------------------------------------------------
            new_state[f"model.layers.{layer_idx}.self_attn.query_colour.{g}"] = A_q_half
            new_state[f"model.layers.{layer_idx}.self_attn.key_colour.{g}"]   = A_k_half

        # 4‑c) validate/aggregate γ′ across groups -------------------------
        # If multiple groups exist we expect the projected vectors to be
        # (near‑)identical.  Otherwise we raise to avoid silent misuse.
        if n_kv_heads == 1:
            gamma_q_prime_layer = projected_gamma_q_group[0]
            gamma_k_prime_layer = projected_gamma_k_group[0]
        else:
            for idx in range(1, n_kv_heads):
                dq = (projected_gamma_q_group[idx] - projected_gamma_q_group[0]).abs().max()
                dk = (projected_gamma_k_group[idx] - projected_gamma_k_group[0]).abs().max()
                if dq > 1e-4 or dk > 1e-4:
                    raise RuntimeError(
                        "Projected RMS‑Norm scales differ between GQA groups (layer {}, Δ={:.2e}/{:.2e}).".format(
                            layer_idx, dq, dk
                        )
                    )
            gamma_q_prime_layer = projected_gamma_q_group[0]
            gamma_k_prime_layer = projected_gamma_k_group[0]

        # store γ′ as new norm weights (size r)
        query_norm_key_new = f"model.layers.{layer_idx}.self_attn.query_norm.weight"
        key_norm_key_new = f"model.layers.{layer_idx}.self_attn.key_norm.weight"
        new_state[query_norm_key_new] = gamma_q_prime_layer.to(dtype_t).clone()
        new_state[key_norm_key_new] = gamma_k_prime_layer.to(dtype_t).clone()

    # 5) save -------------------------------------------------------------------------
    torch.save({"model_state_dict": new_state}, output_ckpt)
    print(f"[✓]  Saved E‑DK‑SVD checkpoint → {output_ckpt}")


# -----------------------------------------------------------------------------
# ----------  CLI -------------------------------------------------------------
# -----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Convert Gemma checkpoint → Energy DK‑SVD format")
    p.add_argument("--input_ckpt", required=True, help="Path to *original* Gemma checkpoint (.pt/.bin)")
    p.add_argument("--output_ckpt", required=True, help="Destination path for the converted checkpoint")
    p.add_argument("--variant", default="1b", choices=["1b"], help="Gemma variant (currently only 1b tested)")
    p.add_argument("--rank", type=int, required=True, help="Target rank r for Q/K projections")
    p.add_argument("--dtype", default="float32", choices=list(DTYPE_MAP), help="Computation / save dtype")
    p.add_argument("--eigen_clamp_min", type=float, default=1.0e-12, help="Minimal eigen‑value before inversion")
    return p.parse_args()


def main() -> None:
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
