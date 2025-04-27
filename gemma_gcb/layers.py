# -*- coding: utf-8 -*-
"""Per-head GCB logic + lightweight caches."""
import math, torch
from torch import Tensor
from typing import Optional, Tuple, List

from .gcb_meta import HeadAux
from .phi import PowerMap, core_residual
from .debug_utils import dbg, check

DEBUG = True          # flip to False for normal runs

def _stats(name, t, step, ten, k=3):
    """print mean/abs-mean/max of a tensor every k steps"""
    if DEBUG and step % k == 0:
        m = ten.float().mean().item()
        am = ten.float().abs().mean().item()
        mx = ten.float().abs().max().item()
        print(f"[{step:>4}] {name:<15}  mean={m:+.3e}  |mean|={am:.3e}  max={mx:.3e}")

# --------------------------------------------------------------------
class GCCache(torch.nn.Module):
    """Tiny per-layer cache that holds core vectors, value projections + blanket sum."""
    def __init__(self, max_seq: int, n_h: int,
                 r_k: int, r_v: int, d_k: int, device):
        super().__init__()
        # Store the full core vector instead of factorized components
        self.register_buffer('P', torch.zeros(max_seq, n_h, r_k, dtype=torch.float16, device=device))
        self.register_buffer('V', torch.zeros(max_seq, n_h, r_v, dtype=torch.float16, device=device))
        self.register_buffer('S', torch.zeros(n_h, d_k, dtype=torch.float32, device=device))

# --------------------------------------------------------------------
class GCBHead(torch.nn.Module):
    """
    Implements *one* logical attention head under Gauge-Core-Blanket.
    We assume rank-r_k core, factors r_a + r_b, blanket tail d_k-r_k.
    """

    def __init__(self, aux: HeadAux,
                 d_k: int, d_v: int,
                 r_k=8, r_v=8):
        super().__init__()
        # fixed (offline) params
        self.register_buffer('A',      aux.A)
        self.register_buffer('A_invT', aux.A_invT)
        self.register_buffer('P_r',    aux.P_r)
        # We're no longer using the factorization
        # self.register_buffer('P_a',    aux.P_a)
        # self.register_buffer('P_b',    aux.P_b)
        self.alpha: float = aux.alpha
        self.lam:   float = aux.lam

        # run-time helpers (injected later)
        self.powmap: Optional[PowerMap] = None
        self.Z_r:    Optional[Tensor]   = None   # (d_v , r_v)

        self.r_v = r_v
        self.d_k, self.r_k = d_k, r_k

    # ----------------------------------------------------------------
    def _dot(self, a: Tensor, b: Tensor) -> Tensor:
        """Dot product along last dim, keep (B , S) layout."""
        return a @ b.T     # (B , S)

    # ----------------------------------------------------------------
    def forward(
        self,
        q_head: Tensor,    # (B , d_k)
        k_head: Tensor,    # (B , d_k)
        v_head: Tensor,    # (B , d_v)
        freqs_row: Tensor, # (1 , d_k/2 , 2)
        cache: GCCache,
        step: int,
        h_idx: int,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """
        Returns:
          core+blanket logits for this head  – shape (B , step)
          value matrix slice                – shape (step , d_v)
        or (None, None) during pure pre-fill (step==0).
        """
        device, dtype = q_head.device, q_head.dtype
        l_idx = -1  # We'll need to get this from outside later
        
        if self.powmap is None:
            self.powmap = PowerMap(self.alpha, self.r_k, self.d_k).to(device)
            
        # promote the half-precision buffers to the incoming compute dtype and device
        A        = self.A.to(device=device, dtype=dtype)
        A_invT   = self.A_invT.to(device=device, dtype=dtype)
        A_T      = A.transpose(-1, -2)  # Add transposed A for correct key un-gauging
        P_r      = self.P_r.to(device=device, dtype=dtype)
        Z_r      = self.Z_r.to(device=device, dtype=dtype)

        # ----------------------------------------------------------
        # 1. raw projections are already in physical space
        q_phys = q_head          # (B , d_k)
        k_phys = k_head
        dbg("q_phys", q_phys, l_idx, h_idx, step)
        dbg("k_phys", k_phys, l_idx, h_idx, step)
        _stats("q_phys", step, step, q_phys)
        _stats("k_phys", step, step, k_phys)

        # 2. Go to the gauge
        q_g = q_phys @ A              # Query: phys → A
        k_g = k_phys @ A_invT         # Key:   phys → A⁻ᵀ
        dbg("q_g", q_g, l_idx, h_idx, step)
        dbg("k_g", k_g, l_idx, h_idx, step)
        _stats("q_g", step, step, q_g)
        _stats("k_g", step, step, k_g)
        
        # Check gauge dot-product invariance
        check(torch.allclose(q_phys @ k_phys.T, q_g @ k_g.T, atol=1e-3),
              f"Gauge dot-product mismatch: max diff = {(q_phys @ k_phys.T - q_g @ k_g.T).abs().max()}",
              l_idx, h_idx, step, level="WARN")

        # 3. Back to physical basis & apply RoPE
        from .rope_utils import apply_rope_query, apply_rope_key
        A_inv = A_invT.transpose(-1, -2)  # Calculate the correct inverse needed
        q_rot_phys = apply_rope_query(q_g @ A_inv, freqs_row)  # gauge → phys → RoPE
        k_rot_phys = apply_rope_key(k_g @ A_T, freqs_row)  # Corrected: use A_T instead of A for key
        dbg("q_rot_phys", q_rot_phys, l_idx, h_idx, step)
        dbg("k_rot_phys", k_rot_phys, l_idx, h_idx, step)
        _stats("q_rot_phys", step, step, q_rot_phys)
        _stats("k_rot_phys", step, step, k_rot_phys)

        # 4. Return to the gauge with the correct matrices
        q_rot = q_rot_phys @ A              # map query into gauge basis
        k_rot = k_rot_phys @ A_invT         # map key into dual gauge basis
        dbg("q_rot", q_rot, l_idx, h_idx, step)
        dbg("k_rot", k_rot, l_idx, h_idx, step)
        _stats("q_rot", step, step, q_rot)
        _stats("k_rot", step, step, k_rot)
        
        # Check RoPE+Gauge norm invariance
        check(torch.allclose(q_rot_phys.norm(dim=-1), q_rot.norm(dim=-1), rtol=5e-3), 
              f"RoPE+Gauge norm mismatch: max rel diff = {((q_rot_phys.norm(dim=-1) - q_rot.norm(dim=-1)) / q_rot_phys.norm(dim=-1)).abs().max()}",
              l_idx, h_idx, step, level="WARN")

        # ---------------- Core projection ----------------------------
        # Project directly without normalization to preserve mathematical consistency
        p_q = q_rot @ P_r                # (B , r_k)
        p_k = k_rot @ P_r                # (B , r_k)
        dbg("p_q", p_q, l_idx, h_idx, step)
        dbg("p_k", p_k, l_idx, h_idx, step)
        _stats("p_q", step, step, p_q)
        _stats("p_k", step, step, p_k)

        # --------------- Value projection (stays physical) --------
        p_v = v_head @ Z_r                   # (B , r_v)
        dbg("p_v", p_v, l_idx, h_idx, step)

        # --------------- Cache write ------------------------------
        # Store the full core vector instead of trying to factorize it
        cache.P[step, h_idx] = p_k.to(torch.float16)
        cache.V[step, h_idx] = p_v.to(torch.float16)
        
        # Log S[h] before update
        dbg("S[h] before", cache.S[h_idx], l_idx, h_idx, step)

        # --- Blanket Update ---
        # Calculate raw phi_k based on key's residual
        res_k = core_residual(k_rot, P_r)
        dbg("res_k", res_k, l_idx, h_idx, step) 
        phi_k_raw = self.powmap(res_k, l_idx, h_idx, step)  # Shape: (B, d_k)
        dbg("phi_k_raw", phi_k_raw, l_idx, h_idx, step)

        # CHANGED: Accumulate raw phi vectors instead of normalized ones
        dbg("phi_k_raw", phi_k_raw, l_idx, h_idx, step)
        _stats("phi_k_raw", step, step, phi_k_raw) # Log raw version

        # Accumulate the raw vectors (sum over batch dim B if B>1)
        cache.S[h_idx] += phi_k_raw.sum(0).float()
        dbg("S[h] after", cache.S[h_idx], l_idx, h_idx, step)
        _stats("S[h]", step, step, cache.S[h_idx]) # Log the state S

        # During the first token there is nothing to attend to.
        if step == 0:
            return None, None

        # --------------- Read history ----------------------------
        p_hist = cache.P[:step, h_idx].to(dtype)          # (S, r_k) - direct retrieval
        v_hist = cache.V[:step, h_idx].to(dtype) @ Z_r.T   # (S, d_v)
        
        dbg("p_hist (read)", p_hist, l_idx, h_idx, step)
        dbg("v_hist", v_hist, l_idx, h_idx, step)
        _stats("p_hist", step, step, p_hist)
        
        # Cast to float32 for numerical stability in the dot product
        p_q_f32 = p_q.float()
        p_hist_f32 = p_hist.float()
        
        # Compute core logits with proper scaling and cast back to original dtype
        # <<< FIX: Scale by sqrt(r_k) for the r_k dimensional core subspace >>>
        core_log_f32 = (p_q_f32 @ p_hist_f32.T) / math.sqrt(self.r_k)
        
        # Safety check and clamp if needed (in debug mode)
        if DEBUG:
            max_val = core_log_f32.abs().max().item()
            if max_val > 80:
                print(f"WARNING: core_log max value {max_val} exceeds safe threshold (80)")
                core_log_f32 = torch.clamp(core_log_f32, -80, 80)
                
        core_log = core_log_f32.to(dtype)
        dbg("core_log", core_log, l_idx, h_idx, step)
        
        # Core energy check
        if step > 0:
            # Simplification: Check energy ratio for current token q_rot @ k_rot vs p_q @ p_k
            core_energy_step = (p_q * p_k).sum(-1).abs().float() # (B,)
            full_energy_step = (q_rot * k_rot).sum(-1).abs().float() # (B,)
            valid_mask = full_energy_step > 1e-6
            if valid_mask.any():
                ratio = (core_energy_step[valid_mask] / full_energy_step[valid_mask]).mean()
                check(ratio > 0.6, 
                    f"Rank-{self.r_k} core energy ratio too low: {ratio:.3f} (expected >0.6)",
                    l_idx, h_idx, step, level="WARN")
                dbg("core_energy_ratio", torch.tensor([ratio]), l_idx, h_idx, step)

        # --- Blanket Logits ---
        # Calculate normalized phi_q for the current query
        res_q = core_residual(q_rot, P_r)
        dbg("res_q", res_q, l_idx, h_idx, step)
        phi_q_raw = self.powmap(res_q, l_idx, h_idx, step)
        dbg("phi_q_raw", phi_q_raw, l_idx, h_idx, step)
        
        # CHANGED: Use raw phi_q directly
        dbg("phi_q_raw", phi_q_raw, l_idx, h_idx, step)
        _stats("phi_q_raw", step, step, phi_q_raw)

        # Compute blanket logits using raw phi_q and accumulated S (float32)
        phi_q_f32 = phi_q_raw.float()
        S_f32 = cache.S[h_idx].float() # S is sum of raw phi_k
        dbg("S[h] read", S_f32, l_idx, h_idx, step)
        
        # Log lambda value for debugging
        # Apply optional lambda clamp (safety valve)
        lam = min(self.lam, 8.0)  # empirical soft-cap
        dbg("lambda", torch.tensor(lam, device=q_head.device), l_idx, h_idx, step)
        
        # Calculate blanket interaction 
        blanket_interaction = phi_q_f32 @ S_f32.T
        dbg("blanket_interaction", blanket_interaction, l_idx, h_idx, step)
        
        # Updated scaling to account for sequence length
        ell = max(1, step)  # length of history
        tail_log_f32 = (lam * blanket_interaction) / (math.sqrt(self.d_k) * math.sqrt(ell))
        dbg("tail_log", tail_log_f32, l_idx, h_idx, step)
        
        # Double-check calculation for verification
        tail_log_recheck = (lam * (phi_q_f32 @ S_f32.T)) / (math.sqrt(self.d_k) * math.sqrt(ell))
        dbg("tail_log_recheck", tail_log_recheck, l_idx, h_idx, step)
        
        # Safety check and clamp if needed (in debug mode)
        if DEBUG:
            max_val = tail_log_f32.abs().max().item()
            if max_val > 80:
                print(f"WARNING: tail_log max value {max_val} exceeds safe threshold (80)")
                tail_log_f32 = torch.clamp(tail_log_f32, -80, 80)
                
        tail_log = tail_log_f32.to(dtype)

        # logging just before return (during non-first tokens)
        if step > 0:  # not the first token
            _stats("core_log", step, step, core_log)
            _stats("tail_log", step, step, tail_log)
            
        # broadcast tail_log over sequence length and add to core_log
        # both should now be in the same dtype (original dtype of inputs)
        return core_log + tail_log.unsqueeze(-1), v_hist