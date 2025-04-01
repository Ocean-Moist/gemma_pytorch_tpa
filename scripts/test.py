import torch
import torch.nn.functional as F
import math
import sys
import os

# --- Configuration ---
HIDDEN_DIM = 1152
HEAD_DIM = 256
# Use a LOW rank first to verify the core math
# If this fails, higher ranks definitely will too.
TARGET_RANK = 6
BATCH_SIZE = 1 # Keep batch and seq simple for verification
SEQ_LEN = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Use float32 for verification script to minimize dtype precision issues
DTYPE = torch.float32
# Set a seed for reproducible random tensors
SEED = 42

# --- Ensure necessary module can be imported ---
try:
    # Adjust path if necessary, e.g., if gqa_to_tpa is in a subdirectory
    # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from gemma.tpa.modules.gqa_to_tpa import compute_svd_tpa_factors
    print("Successfully imported compute_svd_tpa_factors.")
except ImportError as e:
    print(f"ERROR: Could not import compute_svd_tpa_factors: {e}")
    print("Please ensure gqa_to_tpa.py is in the Python path.")
    sys.exit(1)

def verify_tpa_reconstruction(
        hidden_dim: int,
        head_dim: int,
        rank: int,
        batch_size: int,
        seq_len: int,
        device: str,
        dtype: torch.dtype,
        seed: int,
):
    """
    Verifies Constant B-Factor TPA reconstruction against SVD approximation.

    Args:
        hidden_dim: Input hidden dimension.
        head_dim: Output head dimension.
        rank: Target rank for SVD/TPA.
        batch_size: Batch size for input tensor.
        seq_len: Sequence length for input tensor.
        device: Computation device ('cuda' or 'cpu').
        dtype: Data type for tensors.
        seed: Random seed.
    """
    print("\n" + "="*50)
    print(f"Verification Test: Rank={rank}, Hidden={hidden_dim}, Head={head_dim}")
    print(f"Device={device}, DType={dtype}")
    print("="*50)

    torch.manual_seed(seed)
    torch_device = torch.device(device)

    # --- 1. Create Sample Data ---
    # Use float32 for original weight and input for better precision baseline
    W_orig = torch.randn(hidden_dim, head_dim, device=torch_device, dtype=torch.float32)
    x = torch.randn(batch_size, seq_len, hidden_dim, device=torch_device, dtype=torch.float32)
    print(f"Sample shapes: W_orig={W_orig.shape}, x={x.shape}")

    # --- 2. Ground Truth 1: Original Projection ---
    # y_orig = x @ W_orig # Shape: [b, s, head_dim]
    # Use einsum for clarity with batch/seq dimensions
    y_orig = torch.einsum('bsh,hd->bsd', x, W_orig)
    print(f"y_orig shape: {y_orig.shape}")

    # --- 3. Ground Truth 2: SVD Approximation ---
    try:
        # SVD requires float32 or float64
        U, S, Vh = torch.linalg.svd(W_orig.float(), full_matrices=False)
        # U: [hidden_dim, full_rank], S: [full_rank], Vh: [full_rank, head_dim]
    except Exception as e:
        print(f"ERROR during SVD: {e}. Cannot proceed.")
        return

    # Determine effective rank
    max_possible_rank = min(W_orig.shape[0], W_orig.shape[1])
    effective_rank = min(rank, max_possible_rank, len(S))
    if effective_rank <= 0:
        print(f"ERROR: Effective SVD rank is {effective_rank}. Cannot proceed.")
        return
    print(f"Using effective SVD rank: {effective_rank}")

    # Truncate
    U_r = U[:, :effective_rank]      # [hidden_dim, R]
    S_r = S[:effective_rank]         # [R]
    Vh_r = Vh[:effective_rank, :]    # [R, head_dim]

    # Reconstruct approximate weight matrix
    W_approx = U_r @ torch.diag(S_r) @ Vh_r # [hidden_dim, head_dim]
    # y_svd_approx = x @ W_approx # Shape: [b, s, head_dim]
    y_svd_approx = torch.einsum('bsh,hd->bsd', x, W_approx)
    print(f"W_approx shape: {W_approx.shape}")
    print(f"y_svd_approx shape: {y_svd_approx.shape}")

    # --- 4. Compute TPA Factors using the function under test ---
    # Use the target rank, the function will cap it internally if needed
    # Pass W_orig (float32), get factors potentially in target dtype
    print(f"\nCalling compute_svd_tpa_factors with target rank {rank}...")
    wa_comp, b_const_comp = compute_svd_tpa_factors(
        weight_matrix=W_orig, # Use original weight
        rank=rank,            # Pass target rank
        name="TestWeight",
        head_dim=head_dim,
        hidden_dim=hidden_dim,
        device=torch_device,
        dtype=dtype           # Request factors in target dtype
    )
    # Ensure factors are on the correct device and use float32 for recon calc
    wa_comp = wa_comp.to(torch_device, dtype=torch.float32)
    b_const_comp = b_const_comp.to(torch_device, dtype=torch.float32)

    # Check if the effective rank used by the function matches SVD rank
    actual_tpa_rank = wa_comp.shape[0]
    print(f"compute_svd_tpa_factors returned rank: {actual_tpa_rank}")
    print(f"Factor shapes: wa_comp={wa_comp.shape}, b_const_comp={b_const_comp.shape}") # wa: [R, hidden], b: [R, head]
    if actual_tpa_rank != effective_rank:
        print(f"WARNING: Rank used by compute_svd_tpa_factors ({actual_tpa_rank}) differs from SVD effective rank ({effective_rank}). Comparison might be less meaningful.")
        # Use the rank returned by the function for the TPA reconstruction step
        tpa_recon_rank = actual_tpa_rank
    else:
        tpa_recon_rank = effective_rank


    # --- 5. Simulate TPA Reconstruction ---
    # A = x @ W_A^T
    # W_A weights are [Rank, hidden_dim], so W_A^T is [hidden_dim, Rank]
    A_factors = torch.einsum('bsh,rh->bsr', x, wa_comp) # x:[b,s,h], wa:[R,h] -> A:[b,s,R]
    print(f"A_factors shape: {A_factors.shape}")

    # y = einsum(A, B) / Rank
    # A:[b,s,R], B:[R,d] -> y:[b,s,d]
    if tpa_recon_rank > 0:
        y_tpa_recon = torch.einsum('bsr,rd->bsd', A_factors, b_const_comp) / tpa_recon_rank
        print(f"y_tpa_recon shape: {y_tpa_recon.shape}")
    else:
        print("Skipping TPA reconstruction due to zero rank.")
        y_tpa_recon = torch.zeros_like(y_svd_approx) # Create zero tensor for comparison

    # --- 6. Compare Outputs ---
    # Convert all outputs to float32 for comparison
    y_orig_f32 = y_orig.float()
    y_svd_approx_f32 = y_svd_approx.float()
    y_tpa_recon_f32 = y_tpa_recon.float()

    # Error 1: SVD Approximation Error (inherent loss from rank reduction)
    diff_svd_vs_orig = torch.abs(y_svd_approx_f32 - y_orig_f32)
    rel_err_svd_vs_orig = diff_svd_vs_orig / (torch.abs(y_orig_f32) + 1e-9) # Add epsilon for stability
    print(f"\n--- SVD Error (Rank {effective_rank}) ---")
    print(f"  Max Abs Diff (SVD vs Orig):  {diff_svd_vs_orig.max().item():.6e}")
    print(f"  Mean Abs Diff (SVD vs Orig): {diff_svd_vs_orig.mean().item():.6e}")
    print(f"  Max Rel Err (SVD vs Orig):   {rel_err_svd_vs_orig.max().item():.6e}")
    print(f"  Mean Rel Err (SVD vs Orig):  {rel_err_svd_vs_orig.mean().item():.6e}")

    # Error 2: TPA Reconstruction Error (difference between TPA and SVD target)
    # THIS IS THE CRITICAL ONE. It should be very close to zero if the mapping is correct.
    diff_tpa_vs_svd = torch.abs(y_tpa_recon_f32 - y_svd_approx_f32)
    rel_err_tpa_vs_svd = diff_tpa_vs_svd / (torch.abs(y_svd_approx_f32) + 1e-9)
    print(f"\n--- TPA Reconstruction Error (vs SVD Approx) ---")
    print(f"  Max Abs Diff (TPA vs SVD):   {diff_tpa_vs_svd.max().item():.6e}")
    print(f"  Mean Abs Diff (TPA vs SVD):  {diff_tpa_vs_svd.mean().item():.6e}")
    print(f"  Max Rel Err (TPA vs SVD):    {rel_err_tpa_vs_svd.max().item():.6e}")
    print(f"  Mean Rel Err (TPA vs SVD):   {rel_err_tpa_vs_svd.mean().item():.6e}")

    # --- 7. Conclusion ---
    # Threshold for "very close to zero" (e.g., machine epsilon for float32)
    tolerance = 1e-5 # Adjust as needed based on expected precision
    mean_tpa_error = diff_tpa_vs_svd.mean().item()

    if mean_tpa_error < tolerance:
        print(f"\nSUCCESS: TPA reconstruction error ({mean_tpa_error:.6e}) is below tolerance ({tolerance:.1e}).")
        print("The Constant B-Factor TPA formula seems to correctly match the SVD approximation.")
    else:
        print(f"\nFAILURE: TPA reconstruction error ({mean_tpa_error:.6e}) is ABOVE tolerance ({tolerance:.1e}).")
        print("There is likely an error in the compute_svd_tpa_factors mapping or the TPA reconstruction formula simulation.")
        print("Focus on scaling factors (sqrt(R), 1/R) and einsum/matmul operations.")

    print("="*50)


# --- Run the verification ---
if __name__ == "__main__":
    verify_tpa_reconstruction(
        hidden_dim=HIDDEN_DIM,
        head_dim=HEAD_DIM,
        rank=TARGET_RANK, # Start with a low rank
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        device=DEVICE,
        dtype=DTYPE, # Use float32 for verification
        seed=SEED,
    )

    # Optional: Test with a higher rank if the low rank passes
    print("\nRunning again with higher rank (e.g., 128)...")
    verify_tpa_reconstruction(
        hidden_dim=HIDDEN_DIM,
        head_dim=HEAD_DIM,
        rank=128, # Test a higher rank
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        device=DEVICE,
        dtype=DTYPE,
        seed=SEED,
    )

    # Optional: Test with full rank
    print("\nRunning again with 'full' rank (head_dim)...")
    verify_tpa_reconstruction(
        hidden_dim=HIDDEN_DIM,
        head_dim=HEAD_DIM,
        rank=HEAD_DIM, # Test 'full' rank
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        device=DEVICE,
        dtype=DTYPE,
        seed=SEED,
    )