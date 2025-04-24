# -*- coding: utf-8 -*-
"""Per-head GCB logic + lightweight caches."""
import math, torch
from torch import Tensor
from typing import Optional, Tuple, List

from .gcb_meta import HeadAux
from .phi import PowerMap, core_residual
from .rope_utils import apply_rope_phys

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
        P_r      = self.P_r.to(device=device, dtype=dtype)
        P_a      = self.P_a.to(device=device, dtype=dtype)
        P_b      = self.P_b.to(device=device, dtype=dtype)
        Z_r      = self.Z_r.to(device=device, dtype=dtype)

        # ----------------------------------------------------------
        # 1)  phys → gauge
        # 2)  apply RoPE in physical basis
        # 3)  bring the *rotated* vectors back into the gauge basis
        # ----------------------------------------------------------
        qg = q_head @ A                           # gauge
        kg = k_head @ A

        q_rot = apply_rope_phys(qg, A_invT, freqs_row) @ A.T  # gauge
        k_rot = apply_rope_phys(kg, A_invT, freqs_row) @ A.T  # gauge

        # ---------------- Core factors ----------------------------
        p_q = q_rot @ P_r                     # (B , r_k)
        p_k = k_rot @ P_r                     # (B , r_k)

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
        cache.S[h_idx] += phi_k.sum(0).float()

        # During the first token there is nothing to attend to.
        if step == 0:
            return None, None

        # --------------- Read history ----------------------------
        a_hist = cache.A[:step, h_idx].to(dtype)          # (S , r_a)
        b_hist = cache.B[:step, h_idx].to(dtype)          # (S , r_b)
        v_hist = cache.V[:step, h_idx].to(dtype) @ Z_r.T   # (S , d_v)

        # Core logits using classic TPA formula
        core_log = (a_k @ a_hist.T) * (b_k @ b_hist.T)    # (B , S)
        core_log = core_log / math.sqrt(self.d_k)

        # Blanket logits
        phi_q   = self.powmap(core_residual(q_rot, P_r))   # (B , d_k)
        tail_log = (self.lam / math.sqrt(self.d_k)) * (phi_q @ cache.S[h_idx].T)

        # broadcast tail_log over sequence length
        return core_log + tail_log.unsqueeze(-1), v_hist