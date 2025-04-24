#!/usr/bin/env python3
"""
convert_weights.py  –  build the analytical GCB metadata + (optionally)
                        gauge-fixed checkpoint for Gemma-3 1 B.

Example:
  python convert_weights.py --ckpt gemma1b_orig.pt --out gemma1b_gcb
"""
import math, argparse, torch
from pathlib import Path
from gemma_gcb.gcb_meta import GCBMeta, LayerAux

# ---- hyper-params for 1 B ------------------------------------------------
NUM_LAYERS   = 26
NUM_HEADS    = 4
NUM_KV_HEADS = 1
HEAD_DIM     = 256
R_K, R_A, R_B, R_V = 8, 4, 4, 8
EPS = 0.05
# --------------------------------------------------------------------------

def cp_factor(P_r: torch.Tensor):
    """
    Quickly obtain rank-(r_a,r_b) CP factors for the core projector.
    We split the right-singular space so that  P_r ≈ (A ⊙ B).
    Returns matrices with shape (r_k , r_a / r_b) as expected by the runtime.
    """
    _, _, Vh = torch.linalg.svd(P_r, full_matrices=False)   # Vh: (r_k , r_k)
    V = Vh.T                                                # (r_k , r_k)
    A = V[:, :R_A].contiguous()
    B = V[:, R_A:R_A + R_B].contiguous()
    return A, B

# --------------------------------------------------------------------------
def fit_powerlaw(sig_tail: torch.Tensor):
    """σ_i ≃ λ * i^{-α}   (i is 1-based index into the tail)."""
    i = torch.arange(1, sig_tail.numel() + 1, device=sig_tail.device, dtype=sig_tail.dtype)
    y = sig_tail.log().unsqueeze(1)
    X = torch.stack((i.log(), torch.ones_like(i)), 1)      # [n , 2]
    sol, *_ = torch.linalg.lstsq(X, y)
    slope, bias = sol.squeeze()
    alpha = float(-slope)
    lam   = float(torch.exp(bias))
    return alpha, lam

# --------------------------------------------------------------------------
def build_gauge(C: torch.Tensor, eps: float = 0.05):
    # ---- single-step Lanczos gauge -----------------------------------
    S  = 0.5 * (C + C.T)
    v0 = torch.randn(HEAD_DIM, device=C.device)
    v0 = v0 / v0.norm()
    w  = S @ v0
    alpha = torch.dot(v0, w)
    w  = w - alpha * v0
    beta = w.norm()
    v1  = w / (beta + 1e-8)
    coeff = beta / (alpha + 1e-8)
    r = (v0 + coeff * v1).div_((v0 + coeff * v1).norm())

    Delta = torch.outer(r, r) - torch.diag(r ** 2)
    A     = torch.eye(HEAD_DIM, device=C.device) + eps * Delta
    return A, torch.linalg.inv(A).T

# --------------------------------------------------------------------------
def convert(orig_ckpt: Path, out_stem: Path):

    sd = torch.load(orig_ckpt, mmap=True, weights_only=True)
    sd = sd['model_state_dict']

    meta = GCBMeta()

    for l in range(NUM_LAYERS):
        # ----- Value projector -------------------------------------------
        Wv = sd[f'model.layers.{l}.self_attn.qkv_proj.weight']
        Wv = Wv.view(3, NUM_HEADS, HEAD_DIM, -1)[2]        # (H , d_k , d_in)
        Wv = Wv.permute(0, 2, 1).reshape(-1, HEAD_DIM)      # (d_in*H , d_k)
        # Get right singular vectors (for value space projector)
        _, _, Vh = torch.linalg.svd(Wv, full_matrices=False)
        Z_r = Vh.T[:, :R_V].contiguous()  # (d_k , r_v) -> (256, R_V)
        meta.layers[l] = LayerAux(Z_r.half())

        for h in range(NUM_HEADS):
            # ---- original per-head weights -----------------------------
            Wqkv = sd[f'model.layers.{l}.self_attn.qkv_proj.weight']
            Wqkv = Wqkv.view(3, NUM_HEADS, HEAD_DIM, -1)
            Wq = Wqkv[0, h].T     # (d_in , d_k)
            Wk = Wqkv[1, h].T

            # ------------- ∆-gauge heuristic ----------------------------
            C = Wq.T @ Wk
            A, A_invT = build_gauge(C, EPS)

            Wq_g, Wk_g = Wq @ A, Wk @ A_invT
            Cg = Wq_g.T @ Wk_g
            _, _, Vh = torch.linalg.svd(Cg, full_matrices=False)
            P_r = Vh[:R_K].T         # (d_k , r_k)

            sig = torch.linalg.svdvals(Cg)
            alpha, lam = fit_powerlaw(sig[R_K:])

            P_a, P_b = cp_factor(P_r)
            meta.add_head(l, h,
                          A.half(), A_invT.half(),
                          P_r.half(), P_a.half(), P_b.half(),
                          alpha, lam)

    meta.save(out_stem.with_suffix('.pkl'))
    torch.save(sd, out_stem.with_suffix('.pt'))
    print(f"✓  Saved  {out_stem}.pt   and   {out_stem}.pkl")

# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, type=Path, help="Original Gemma-3 1B .pt")
    ap.add_argument('--out',  required=True, type=Path, help="Output stem (no ext)")
    args = ap.parse_args()
    convert(args.ckpt, args.out)