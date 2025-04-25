# -*- coding: utf-8 -*-
"""Per-head GCB logic + lightweight caches."""
import math, torch
from torch import Tensor
from typing import Optional, Tuple, List

from .gcb_meta import HeadAux
from .phi import PowerMap, core_residual

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
    """Tiny per-layer cache that holds compressed KV + blanket sum."""
    def __init__(self, max_seq: int, n_h: int,
                 r_a: int, r_b: int, r_v: int, d_k: int, device):
        super().__init__()
        self.register_buffer('A', torch.zeros(max_seq, n_h, r_a, dtype=torch.float16, device=device))
        self.register_buffer('B', torch.zeros(max_seq, n_h, r_b, dtype=torch.float16, device=device))
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
                 r_k=8, r_a=4, r_b=4, r_v=8):
        super().__init__()
        # fixed (offline) params
        self.register_buffer('A',      aux.A)
        self.register_buffer('A_invT', aux.A_invT)
        self.register_buffer('P_r',    aux.P_r)
        self.register_buffer('P_a',    aux.P_a)
        self.register_buffer('P_b',    aux.P_b)
        self.alpha: float = aux.alpha
        self.lam:   float = aux.lam

        # run-time helpers (injected later)
        self.powmap: Optional[PowerMap] = None
        self.Z_r:    Optional[Tensor]   = None   # (d_v , r_v)

        self.r_a, self.r_b, self.r_v = r_a, r_b, r_v
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
        if self.powmap is None:
            self.powmap = PowerMap(self.alpha, self.r_k, self.d_k).to(device)
            
        # promote the half-precision buffers to the incoming compute dtype and device
        A        = self.A.to(device=device, dtype=dtype)
        A_invT   = self.A_invT.to(device=device, dtype=dtype)
        A_T      = A.transpose(-1, -2)  # Add transposed A for correct key un-gauging
        P_r      = self.P_r.to(device=device, dtype=dtype)
        P_a      = self.P_a.to(device=device, dtype=dtype)
        P_b      = self.P_b.to(device=device, dtype=dtype)
        Z_r      = self.Z_r.to(device=device, dtype=dtype)

        # ----------------------------------------------------------
        # 1. raw projections are already in physical space
        q_phys = q_head          # (B , d_k)
        k_phys = k_head
        _stats("q_phys", step, step, q_phys)
        _stats("k_phys", step, step, k_phys)

        # 2. Go to the gauge
        q_g = q_phys @ A              # Query: phys → A
        k_g = k_phys @ A_invT         # Key:   phys → A⁻ᵀ
        _stats("q_g", step, step, q_g)
        _stats("k_g", step, step, k_g)

        # 3. Back to physical basis & apply RoPE
        from .rope_utils import apply_rope_query, apply_rope_key
        q_rot_phys = apply_rope_query(q_g @ A_invT, freqs_row)  # gauge → phys → RoPE
        k_rot_phys = apply_rope_key(k_g @ A_T, freqs_row)  # Corrected: use A_T instead of A for key
        _stats("q_rot_phys", step, step, q_rot_phys)
        _stats("k_rot_phys", step, step, k_rot_phys)

        # 4. Return to the gauge with the correct matrices
        q_rot = q_rot_phys @ A              # map query into gauge basis
        k_rot = k_rot_phys @ A_invT         # map key into dual gauge basis
        _stats("q_rot", step, step, q_rot)
        _stats("k_rot", step, step, k_rot)

        # ---------------- Core factors ----------------------------
        # Project directly without normalization to preserve mathematical consistency
        p_q = q_rot @ P_r                # (B , r_k)
        p_k = k_rot @ P_r                # (B , r_k)
        _stats("p_q", step, step, p_q)
        _stats("p_k", step, step, p_k)

        a_k = p_k @ P_a                       # (B , r_a)
        b_k = p_k @ P_b                       # (B , r_b)

        # --------------- Value projection (stays physical) --------
        p_v = v_head @ Z_r                   # (B , r_v)

        # --------------- Cache write ------------------------------
        cache.A[step, h_idx] = a_k.to(torch.float16)
        cache.B[step, h_idx] = b_k.to(torch.float16)
        cache.V[step, h_idx] = p_v.to(torch.float16)

        # Blanket running-sum (gauge basis, fp32)
        phi_k = self.powmap(core_residual(k_rot, P_r))     # (B , d_k)
        _stats("phi_k", step, step, phi_k)
        cache.S[h_idx] += phi_k.sum(0).float()    # true cumulative sum

        # During the first token there is nothing to attend to.
        if step == 0:
            return None, None

        # --------------- Read history ----------------------------
        a_hist = cache.A[:step, h_idx].to(dtype)          # (S , r_a)
        b_hist = cache.B[:step, h_idx].to(dtype)          # (S , r_b)
        v_hist = cache.V[:step, h_idx].to(dtype) @ Z_r.T   # (S , d_v)

        # Core logits using proper TPA formula - reconstruct the history vectors
        p_hist = a_hist @ P_a.T + b_hist @ P_b.T     # shape: (S, r_k)
        _stats("p_hist", step, step, p_hist)
        
        # Cast to float32 for numerical stability in the dot product
        p_q_f32 = p_q.float()
        p_hist_f32 = p_hist.float()
        
        # Compute core logits with proper scaling and cast back to original dtype
        core_log_f32 = (p_q_f32 @ p_hist_f32.T) / math.sqrt(self.d_k)
        
        # Safety check and clamp if needed (in debug mode)
        if DEBUG:
            max_val = core_log_f32.abs().max().item()
            if max_val > 80:
                print(f"WARNING: core_log max value {max_val} exceeds safe threshold (80)")
                core_log_f32 = torch.clamp(core_log_f32, -80, 80)
                
        core_log = core_log_f32.to(dtype)

        # Blanket logits
        phi_q = self.powmap(core_residual(q_rot, P_r))   # (B , d_k)
        
        # Cast to float32 for numerical stability in the blanket logit computation
        phi_q_f32 = phi_q.float()
        S_f32 = cache.S[h_idx].float()  # Already float32, but being explicit
        
        # Compute blanket logits with consistent scaling (same as core_log)
        tail_log_f32 = (self.lam * (phi_q_f32 @ S_f32.T)) / math.sqrt(self.d_k)
        
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