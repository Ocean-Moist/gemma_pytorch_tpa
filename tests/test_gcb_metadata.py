import pytest
import torch
from pathlib import Path
from gemma_gcb.gcb_meta import GCBMeta
from gemma_gcb.phi import PowerMap, core_residual # Assuming phi.py is accessible
# Add imports for build_gauge, fit_powerlaw if testing requires re-running parts of convert_weights

# --- Constants (adjust if needed) ---
MODEL_DIR = Path("./gemma_models") # Adjust as needed
ORIG_CKPT = MODEL_DIR / "gemma1b_orig.pt"
GCB_META_FILE = MODEL_DIR / "gemma1b_gcb.pkl"
GCB_CKPT = MODEL_DIR / "gemma1b_gcb.pt" # Might need this if testing gauged weights

NUM_LAYERS = 26
NUM_HEADS = 4
HEAD_DIM = 256
R_K = 8
DEVICE = "cpu" # Run offline tests on CPU

# --- Load Data Fixture ---
@pytest.fixture(scope="module")
def gcb_data():
    if not GCB_META_FILE.exists():
        pytest.skip(f"GCB metadata file not found: {GCB_META_FILE}")
    # sd_orig = torch.load(ORIG_CKPT, map_location=DEVICE, weights_only=True)['model_state_dict']
    meta = GCBMeta.load(GCB_META_FILE)
    return meta #, sd_orig

# --- Tests ---
def test_metadata_loading(gcb_data):
    meta = gcb_data
    assert meta is not None
    assert len(meta.heads) == NUM_LAYERS * NUM_HEADS
    assert len(meta.layers) == NUM_LAYERS

@pytest.mark.parametrize("l_idx", range(NUM_LAYERS))
@pytest.mark.parametrize("h_idx", range(NUM_HEADS))
def test_gauge_matrices(gcb_data, l_idx, h_idx):
    meta = gcb_data
    aux = meta.heads[(l_idx, h_idx)]
    A = aux.A.float().to(DEVICE)
    A_invT = aux.A_invT.float().to(DEVICE)

    # Check A is invertible (approx)
    identity = torch.eye(HEAD_DIM, device=DEVICE)
    assert torch.allclose(A @ torch.linalg.inv(A), identity, atol=1e-3), f"L{l_idx}H{h_idx}: A not invertible"
    # Check A_invT is indeed (A^-1)^T
    A_inv = torch.linalg.inv(A)
    assert torch.allclose(A_invT, A_inv.T, atol=1e-3), f"L{l_idx}H{h_idx}: A_invT is not inv(A)^T"

@pytest.mark.parametrize("l_idx", range(NUM_LAYERS))
@pytest.mark.parametrize("h_idx", range(NUM_HEADS))
def test_projector_orthonormality(gcb_data, l_idx, h_idx):
    meta = gcb_data
    aux = meta.heads[(l_idx, h_idx)]
    P_r = aux.P_r.float().to(DEVICE) # (d_k, r_k)

    # P_r.T @ P_r should be Identity
    identity_rk = torch.eye(R_K, device=DEVICE)
    assert torch.allclose(P_r.T @ P_r, identity_rk, atol=1e-3), f"L{l_idx}H{h_idx}: P_r not orthonormal (P_r^T P_r != I)"

    # P_r @ P_r.T should be rank R_K
    proj_mat = P_r @ P_r.T
    rank = torch.linalg.matrix_rank(proj_mat, tol=1e-3).item()
    assert rank == R_K, f"L{l_idx}H{h_idx}: P_r @ P_r^T has rank {rank}, expected {R_K}"

def test_hadamard_tail_isolation(gcb_data):
    meta = gcb_data
    # Use params from the first head as representative
    aux = meta.heads[(0, 0)]
    alpha = aux.alpha
    pow_map = PowerMap(alpha, R_K, HEAD_DIM).to(DEVICE)

    # Create random input vectors
    B = 4 # Batch size for test
    x_rand = torch.randn(B, HEAD_DIM, device=DEVICE)
    # Project out the core component manually for this test
    P_r = aux.P_r.float().to(DEVICE)
    x_tail_approx = core_residual(x_rand, P_r) # Simulate input to PowerMap

    phi_x = pow_map(x_tail_approx) # Calculate PowerMap output

    # Check first r_k coordinates are zero
    assert torch.all(phi_x[:, :R_K] == 0.0), f"PowerMap output has non-zero core components: {phi_x[:, :R_K]}"

@pytest.mark.parametrize("l_idx", range(NUM_LAYERS))
@pytest.mark.parametrize("h_idx", range(NUM_HEADS))
def test_alpha_lambda_sanity(gcb_data, l_idx, h_idx):
     meta = gcb_data
     aux = meta.heads[(l_idx, h_idx)]
     alpha = aux.alpha
     lam = aux.lam
     # Basic sanity checks
     assert 0.8 < alpha < 2.5, f"L{l_idx}H{h_idx}: Alpha ({alpha}) seems out of expected range (0.8-2.5)"
     assert 1e-5 < lam < 80.0, f"L{l_idx}H{h_idx}: Lambda ({lam}) seems out of expected range (1e-5 - 80.0)"
     # Add test to re-fit if you have the original singular values available


# TODO (Optional): Add test_gauge_invariance if feasible by loading Wq/Wk from *original* ckpt
# This requires more setup to extract original weights.