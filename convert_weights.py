#!/usr/bin/env python3
"""
convert_weights.py  –  build the analytical GCB metadata + (optionally)
                        gauge-fixed checkpoint for Gemma-3 1 B.

Uses orthogonal Procrustes gauge which:
1. Keeps algebraic guarantees for GCB
2. Maintains exact compatibility with RoPE
3. Delivers spectrum-sharpening benefits

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

# Core rank settings - adjust as needed:
# - For full-rank sanity check: R_K = HEAD_DIM (script auto-handles empty tail)
# - For compression: R_K = 8 or 16 (much smaller than HEAD_DIM)
R_K = HEAD_DIM  # Full-rank mode (script handles power-law fit automatically)
# R_K = 16      # Uncomment for compression mode

# Define SKIP_CP based on R_K automatically
R_V = HEAD_DIM  # Keep value rank high for now (adjust if needed)
SKIP_CP = (R_K == HEAD_DIM)  # Skip CP factorization if core is full rank
# --------------------------------------------------------------------------

def cp_factor(P_r: torch.Tensor, device):
    """
    Quickly obtain rank-(r_a,r_b) CP factors for the core projector.
    We split the right-singular space so that  P_r ≈ (A ⊙ B).
    Returns matrices with shape (r_k , r_a / r_b) as expected by the runtime.
    Ensures intermediate tensors are on the correct device.
    
    If SKIP_CP is True, returns (None, None) - runtime checks for None.
    """
    if SKIP_CP:
        # Return None for CP factors when core is full rank or skipping is requested
        return None, None
    
    # Ensure P_r is on the target device
    P_r_dev = P_r.to(device)
    _, _, Vh = torch.linalg.svd(P_r_dev, full_matrices=False)   # Vh: (r_k , r_k) on device
    V = Vh.T                                                # (r_k , r_k) on device
    
    # Define R_A and R_B here since they're no longer global constants
    R_A = min(8, R_K // 2)  # Ensure it doesn't exceed R_K/2
    R_B = min(8, R_K // 2)  # Ensure it doesn't exceed R_K/2
    
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
def orthogonal_procrustes_gauge(Wq: torch.Tensor, Wk: torch.Tensor) -> tuple:
    """
    Return the Procrustes gauge A=UVᵀ for C = Wqᵀ Wk.
    
    Args:
        Wq: Query weight matrix (d_in, d_k) in FP32 on the target device
        Wk: Key weight matrix (d_in, d_k) in FP32 on the target device
        
    Returns:
        A: Orthogonal gauge matrix
        A: A second copy of A (for A_invT slot - should be A.T because A is orthogonal and A_invT = A.T)
        beta: Energy ratio captured by the first R_K singular values
        s: Singular values of P_r
        Vh: Right singular vectors of P_r
    """
    # Compute C = Wq.T @ Wk
    C = Wq.T @ Wk                        # (d_k, d_k)
    
    # Apply SVD to get optimal orthogonal solution
    U, _, Vh = torch.linalg.svd(C, full_matrices=False)
    A = U @ Vh                          # orthogonal, so A⁻¹ = A.T and A⁻ᵀ = A.T.T = A
    
    # Check the energy captured in the new gauge
    P_r = (Wq @ A).T @ (Wk @ A)          # interaction in the new gauge
    _, s, Vh = torch.linalg.svd(P_r, full_matrices=False)
    beta = (s[:R_K] ** 2).sum() / (s ** 2).sum()   # energy share
    
    return A, A, beta, s, Vh  # Return A twice - caller will use A.T for A_invT

# --------------------------------------------------------------------------
def convert(orig_ckpt: Path, out_stem: Path, auto_beta=False, verbose=False, global_beta=None):

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
    
    # Initialize beta table if using auto_beta
    beta_table = [[(1.0, 1.0, 1.0, 1.0) for _ in range(NUM_HEADS)] for _ in range(NUM_LAYERS)] if auto_beta else None

    for l in range(NUM_LAYERS):
        print(f"Processing layer {l}/{NUM_LAYERS-1}")
        # ----- Value projector -------------------------------------------
        Wv_cpu = sd[f'model.layers.{l}.self_attn.qkv_proj.weight']
        Wv_cpu = Wv_cpu.view(3, NUM_HEADS, HEAD_DIM, -1)[2]        # (H , d_k , d_in)
        Wv_cpu = Wv_cpu.permute(0, 2, 1).reshape(-1, HEAD_DIM)     # (d_in*H , d_k)

        # Move to device for SVD, cast to float32
        Wv_float_dev = Wv_cpu.float().to(device)
        print(f"  Value matrix shape: {Wv_float_dev.shape}, dtype: {Wv_float_dev.dtype}, device: {Wv_float_dev.device}")

        # Get RIGHT singular vectors (Vh.T) for the value space projector - SVD on device
        # Runtime reconstructs values as v_recon = (v_head @ Z_r) @ Z_r.T.
        # Using Vh.T optimizes reconstruction error for v_head in the value space.
        _, _, Vh_dev = torch.linalg.svd(Wv_float_dev, full_matrices=False)
        Z_r_dev = Vh_dev.T[:, :R_V].contiguous()  # (d_v, r_v) - correct shape for projection

        # Store Z_r on CPU as float16
        meta.layers[l] = LayerAux(Z_r_dev.cpu().half())

        # Cleanup GPU memory
        del Wv_cpu, Wv_float_dev, Z_r_dev
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
            
            # ➋ Automatic β search or global beta application
            if global_beta is not None:
                # Apply global beta directly to Wk
                β = abs(global_beta)  # Use absolute value for consistency
                cond = 1.0  # No conditioning for global beta
                sign = 1.0 if global_beta >= 0 else -1.0  # Preserve sign
                scale = β * cond * sign
                
                if verbose:
                    print(f"→ Using global β={β:.3f} sign={sign:+.0f} scale={scale:.3f} for L{l:02d}H{h}")
                
                # Store in beta_table if it exists
                if beta_table is not None:
                    beta_table[l][h] = (β, cond, sign, scale)
                    
                Wk_dev.mul_(scale)
                
            elif auto_beta:
                best_E = 1e30
                best_B = 1.0
                best_cnd = 1.0
                
                for cnd in [1.0, 1.2, 1.3, 1.4]:
                    # Apply the conditioning factor
                    Wk_test = Wk_dev * cnd
                    
                    # Proper Frobenius inner product for min_β ‖Wq – β·Wk‖_F²
                    num = torch.dot(Wq_dev.flatten(), Wk_test.flatten())  # <Wq, Wk>_F
                    den = torch.dot(Wk_test.flatten(), Wk_test.flatten())  # ||Wk||²_F
                    beta = (num / den).item()
                    
                    # Keep sign information separate
                    sign = 1.0
                    if beta < 0:
                        sign = -1.0
                        beta = -beta  # Work with |β| for optimization
                    
                    E = torch.norm(Wq_dev - beta * Wk_test * sign, p='fro') / torch.norm(Wq_dev, p='fro')
                    
                    if E < best_E - 1e-4:  # tiny tolerance to avoid flip-flop
                        best_E, best_B, best_cnd, best_sign = E, beta, cnd, sign
                
                β = best_B
                cond = best_cnd
                sign = best_sign if 'best_sign' in locals() else 1.0  # Default to positive if not set
                
                # Calculate total scale factor (always positive from our optimization)
                scale = β * cond * sign  # Apply sign here
                
                if verbose:
                    print(f"→ auto-β L{l:02d}H{h}: β={β:.3f} cond={cond} sign={sign:+.0f} scale={scale:.3f}  E={best_E:6.2e}")
                
                # Store beta, cond, sign, and scale in the metadata
                beta_table[l][h] = (β, cond, sign, scale)
                
                # Apply the total scale factor to Wk
                Wk_dev.mul_(scale)

            # ------------- Orthogonal Procrustes gauge ----------------
            print(f"    Calculating orthogonal Procrustes gauge for head {h} using tensors on device: {Wq_dev.device}, {Wk_dev.device}")
            A_dev, _, beta, s_dev, Vh_dev = orthogonal_procrustes_gauge(Wq_dev, Wk_dev)
            print(f"    Final β={beta:.3f} with orthogonal Procrustes gauge")

            # --- Use SVD results from Procrustes calculation (avoid recomputing) ---
            # P_r and s are already available from the orthogonal_procrustes_gauge function
            P_r_dev = Vh_dev[:R_K].T.contiguous()         # (d_k , r_k) on device

            # --- Power Law Fit (only if the tail is non-empty and has enough points) ---
            tail = s_dev[R_K:]  # length = HEAD_DIM - R_K
            if R_K < HEAD_DIM and tail.numel() >= 4:  # Need a tail & at least 4 points for reliable power-law fit
                alpha, lam = fit_powerlaw(tail, device=device)
            else:  # full-rank core or too few points ⇒ no blanket needed
                alpha, lam = 0.0, 0.0
                print(f"    R_K={R_K}: skipping power-law fit (insufficient tail), alpha=lam=0")

            # --- CP Factor (operates on device tensors) ---
            P_a_dev, P_b_dev = cp_factor(P_r_dev, device=device) # CP factoring on device (returns None if SKIP_CP is True)
            
            # In full-rank mode, P_a_dev and P_b_dev will be None

            # --- Store results (move back to CPU, convert to half) ---
            A_half = A_dev.cpu().half()  # Single allocation
            meta.add_head(l, h,
                          A_half, A_half.T,  # For orthogonal matrix, A_invT = A.T
                          P_r_dev.cpu().half(), 
                          P_a_dev.cpu().half() if P_a_dev is not None else None,
                          P_b_dev.cpu().half() if P_b_dev is not None else None,
                          alpha, lam)

            # --- Aggressive Cleanup within head loop ---
            del Wq_cpu, Wk_cpu, Wq_dev, Wk_dev, A_dev
            del P_r_dev, P_a_dev, P_b_dev, s_dev, Vh_dev
            if device.type == 'cuda': torch.cuda.empty_cache()
            if device.type == 'mps': torch.mps.empty_cache()
            gc.collect()


    print("Saving GCB metadata and model weights...")
    meta.save(out_stem.with_suffix('.pkl'))
    # Save the original state dict (which remained on CPU)
    torch.save({'model_state_dict': sd}, out_stem.with_suffix('.pt'))
    
    # Save beta table if auto_beta was used
    if auto_beta:
        import json
        beta_file = out_stem.with_suffix('.betas.json')
        with open(beta_file, 'w') as f:
            json.dump(beta_table, f, indent=2)
        print(f"✓  Saved beta scaling factors to {beta_file}")
    
    print(f"✓  Saved  {out_stem}.pt   and   {out_stem}.pkl")

# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, type=Path, help="Original Gemma-3 1B .pt")
    ap.add_argument('--out',  required=True, type=Path, help="Output stem (no ext)")
    ap.add_argument('--auto_beta', action='store_true', help='search β per head before gauge/SVD')
    ap.add_argument('-v', '--verbose', action='store_true', help='Show detailed output')
    ap.add_argument('--global_beta', type=float, help='Global beta value to apply to all heads (bypass auto search)')
    args = ap.parse_args()
    convert(args.ckpt, args.out, 
            auto_beta=args.auto_beta, 
            verbose=args.verbose, 
            global_beta=args.global_beta)