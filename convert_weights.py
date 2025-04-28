
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
import gc # Import garbage collector

# ---- hyper-params for 1 B ------------------------------------------------
NUM_LAYERS   = 26
NUM_HEADS    = 4
NUM_KV_HEADS = 1
HEAD_DIM     = 256
R_K, R_A, R_B, R_V = 16, 8, 8, 8  # Increased R_K, R_A, R_B for better core coverage
EPS = 0.15
# --------------------------------------------------------------------------

def cp_factor(P_r: torch.Tensor, device):
    """
    Quickly obtain rank-(r_a,r_b) CP factors for the core projector.
    We split the right-singular space so that  P_r ≈ (A ⊙ B).
    Returns matrices with shape (r_k , r_a / r_b) as expected by the runtime.
    Ensures intermediate tensors are on the correct device.
    """
    # Ensure P_r is on the target device
    P_r_dev = P_r.to(device)
    _, _, Vh = torch.linalg.svd(P_r_dev, full_matrices=False)   # Vh: (r_k , r_k) on device
    V = Vh.T                                                # (r_k , r_k) on device
    # take two disjoint slices and QR-orthonormalise each
    A0 = V[:, :R_A].contiguous()
    B0 = V[:, R_A:R_A + R_B].contiguous()
    A, _ = torch.linalg.qr(A0, mode="reduced")           # (r_k , r_a) on device
    B, _ = torch.linalg.qr(B0, mode="reduced")           # (r_k , r_b) on device
    del P_r_dev, Vh, V, A0, B0 # Cleanup intermediate GPU tensors
    return A, B

# --------------------------------------------------------------------------
def fit_powerlaw(sig_tail: torch.Tensor, device):
    """
    σ_i ≃ λ * i^{-α}   (i is 1-based index into the tail).
    Ensures tensors are created on the specified device.
    """
    # Ensure sig_tail is on the target device
    sig_tail_dev = sig_tail.to(device)
    i = torch.arange(1, sig_tail_dev.numel() + 1, device=device, dtype=sig_tail_dev.dtype)
    y = sig_tail_dev.log().unsqueeze(1)
    X = torch.stack((i.log(), torch.ones_like(i)), 1)      # [n , 2] on device
    sol, *_ = torch.linalg.lstsq(X, y) # lstsq happens on device
    slope, bias = sol.squeeze()
    alpha = float(-slope) # Convert scalars to float on CPU

    # Get raw lambda from the fit
    raw_lam = float(torch.exp(bias)) # Convert scalar to float on CPU

    # Use raw lambda with a small floor for numerical stability
    # without the extra normalization that could cause scaling issues
    lam = max(raw_lam, 1e-4)

    print(f"Power-law fit: alpha={alpha:.3f}, raw_lambda={raw_lam:.3f}, final_lambda={lam:.3f}")
    del sig_tail_dev, i, y, X, sol, slope, bias # Cleanup GPU tensors
    return alpha, lam

# --------------------------------------------------------------------------
def build_gauge(C: torch.Tensor, eps: float = 0.05, device=None):
    """
    Builds the gauge matrix A and its inverse transpose.
    Ensures tensors are created on the specified device.
    """
    # Ensure C is on the target device
    C_dev = C.to(device)
    # ---- single-step Lanczos gauge -----------------------------------
    S  = 0.5 * (C_dev + C_dev.T)
    # Create random tensor directly on the device
    v0 = torch.randn(HEAD_DIM, device=device, dtype=C_dev.dtype)
    v0 = v0 / v0.norm()
    w  = S @ v0
    alpha = torch.dot(v0, w)
    w  = w - alpha * v0
    beta = w.norm()
    v1  = w / (beta + 1e-8)
    coeff = beta / (alpha + 1e-8)
    r = (v0 + coeff * v1).div_((v0 + coeff * v1).norm())

    Delta = torch.outer(r, r) - torch.diag(r ** 2)
    # Create eye tensor directly on the device
    A     = torch.eye(HEAD_DIM, device=device, dtype=C_dev.dtype) + eps * Delta
    A_invT = torch.linalg.inv(A).T # Inverse happens on device
    del C_dev, S, v0, w, alpha, beta, v1, coeff, r, Delta # Cleanup GPU tensors
    return A, A_invT

# --- iterative Δ-gauge ------------------------------------------------
def improved_gauge(Wq, Wk, max_iter=50, tol=0.01, device=None):
    """
    Iteratively improve the gauge to maximize energy captured by the first R_K singular values.
    
    Args:
        Wq: Query weight matrix
        Wk: Key weight matrix
        max_iter: Maximum number of iterations (default: 8)
        tol: Tolerance for improvement in beta (default: 0.01)
        device: Device for computation
        
    Returns:
        A: Gauge matrix
        A_invT: Inverse transpose of gauge matrix
        beta: Energy ratio captured by the first R_K singular values
    """
    A = torch.eye(HEAD_DIM, device=device, dtype=Wq.dtype)
    A_invT = torch.eye(HEAD_DIM, device=device, dtype=Wq.dtype)
    beta_old = 0.0
    
    for i in range(max_iter):
        C = (Wq @ A).T @ (Wk @ A_invT)          # gauge-current interaction
        _, s, Vh = torch.linalg.svd(C, full_matrices=False)
        beta = (s[:R_K] ** 2).sum() / (s ** 2).sum()   # energy share

        print(f"      Iteration {i}: β={beta:.3f}")
        
        if i > 0 and (beta - beta_old) < tol:
            print(f"      Stopping early at iteration {i}: Δβ={beta-beta_old:.4f} < {tol}")
            # break                                # no useful progress
        beta_old = beta

        # one Lanczos / Newton step as before
        A_step, A_invT_step = build_gauge(C, EPS, device=device)
        A = A @ A_step
        A_invT = A_invT_step @ A_invT            # keep them coherent
        
    return A, A_invT, beta

# --------------------------------------------------------------------------
def convert(orig_ckpt: Path, out_stem: Path):

    # --- Determine Device ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA ({torch.cuda.get_device_name(device)})")
    # --- Add MPS Check (Apple Silicon) ---
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon GPU)")
    # --- Fallback to CPU ---
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    print(f"Loading checkpoint from {orig_ckpt}...")
    # Load to CPU first to avoid potential large VRAM usage for the whole dict
    sd = torch.load(orig_ckpt, map_location='cpu', weights_only=True)
    sd = sd['model_state_dict']

    print("Starting GCB metadata generation...")
    meta = GCBMeta()

    for l in range(NUM_LAYERS):
        print(f"Processing layer {l}/{NUM_LAYERS-1}")
        # ----- Value projector -------------------------------------------
        Wv_cpu = sd[f'model.layers.{l}.self_attn.qkv_proj.weight']
        Wv_cpu = Wv_cpu.view(3, NUM_HEADS, HEAD_DIM, -1)[2]        # (H , d_k , d_in)
        Wv_cpu = Wv_cpu.permute(0, 2, 1).reshape(-1, HEAD_DIM)     # (d_in*H , d_k)

        # Move to device for SVD, cast to float32
        Wv_float_dev = Wv_cpu.float().to(device)
        print(f"  Value matrix shape: {Wv_float_dev.shape}, dtype: {Wv_float_dev.dtype}, device: {Wv_float_dev.device}")

        # Get right singular vectors (for value space projector) - SVD on device
        _, _, Vh_dev = torch.linalg.svd(Wv_float_dev, full_matrices=False)
        Z_r_dev = Vh_dev.T[:, :R_V].contiguous()  # (d_k , r_v) -> (256, R_V) on device

        # Store Z_r on CPU as float16
        meta.layers[l] = LayerAux(Z_r_dev.cpu().half())

        # Cleanup GPU memory
        del Wv_cpu, Wv_float_dev, Vh_dev, Z_r_dev
        if device.type == 'cuda': torch.cuda.empty_cache()
        if device.type == 'mps': torch.mps.empty_cache()
        gc.collect()


        for h in range(NUM_HEADS):
            print(f"  Processing head {h}/{NUM_HEADS-1}")
            # ---- original per-head weights (keep on CPU initially) ------
            Wqkv = sd[f'model.layers.{l}.self_attn.qkv_proj.weight']
            Wqkv = Wqkv.view(3, NUM_HEADS, HEAD_DIM, -1)
            Wq_cpu = Wqkv[0, h].T.float()     # (d_in , d_k), float32 on CPU
            Wk_cpu = Wqkv[1, h].T.float()     # float32 on CPU

            # --- Move to device for computation ---
            Wq_dev = Wq_cpu.to(device)
            Wk_dev = Wk_cpu.to(device)

            # ------------- Iterative ∆-gauge improvement ----------------
            print(f"    Calculating improved Gauge for head {h} using tensors on device: {Wq_dev.device}, {Wk_dev.device}")
            A_dev, A_invT_dev, beta = improved_gauge(Wq_dev, Wk_dev, device=device)
            print(f"    Final β={beta:.3f} after iterative gauge")

            # --- Core projection SVD (on device) ---
            Wq_g_dev = Wq_dev @ A_dev       # Matmul on device
            Wk_g_dev = Wk_dev @ A_invT_dev  # Matmul on device
            Cg_dev = Wq_g_dev.T @ Wk_g_dev  # Matmul on device
            print(f"    Performing Core SVD on tensor with device: {Cg_dev.device}")
            _, s_dev, Vh_core_dev = torch.linalg.svd(Cg_dev, full_matrices=False) # SVD on device
            P_r_dev = Vh_core_dev[:R_K].T.contiguous()         # (d_k , r_k) on device

            # --- Power Law Fit (operates on device tensors) ---
            # Pass singular values tail, ensuring it's on the device
            alpha, lam = fit_powerlaw(s_dev[R_K:], device=device)

            # --- CP Factor (operates on device tensors) ---
            # Pass P_r, ensuring it's on the device
            P_a_dev, P_b_dev = cp_factor(P_r_dev, device=device) # CP factoring on device

            # --- Store results (move back to CPU, convert to half) ---
            meta.add_head(l, h,
                          A_dev.cpu().half(), A_invT_dev.cpu().half(),
                          P_r_dev.cpu().half(), P_a_dev.cpu().half(), P_b_dev.cpu().half(),
                          alpha, lam)

            # --- Aggressive Cleanup within head loop ---
            del Wq_cpu, Wk_cpu, Wq_dev, Wk_dev, A_dev, A_invT_dev
            del Wq_g_dev, Wk_g_dev, Cg_dev, s_dev, Vh_core_dev, P_r_dev, P_a_dev, P_b_dev
            if device.type == 'cuda': torch.cuda.empty_cache()
            if device.type == 'mps': torch.mps.empty_cache()
            gc.collect()


    print("Saving GCB metadata and model weights...")
    meta.save(out_stem.with_suffix('.pkl'))
    # Save the original state dict (which remained on CPU)
    torch.save({'model_state_dict': sd}, out_stem.with_suffix('.pt'))
    print(f"✓  Saved  {out_stem}.pt   and   {out_stem}.pkl")

# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, type=Path, help="Original Gemma-3 1B .pt")
    ap.add_argument('--out',  required=True, type=Path, help="Output stem (no ext)")
    args = ap.parse_args()
    convert(args.ckpt, args.out)