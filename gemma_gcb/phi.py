# -*- coding: utf-8 -*-
"""Hadamard-based power-law feature map  φ_α  used by the Blanket."""
import math, torch
from torch import Tensor
from typing import Final

def hadamard(n: int, *, dtype=None, device=None):
    """Create a Hadamard matrix of size n x n (n must be a power of 2)."""
    if n == 1:
        return torch.ones((1, 1), dtype=dtype, device=device)
    h = hadamard(n // 2, dtype=dtype, device=device)
    top = torch.cat((h, h), 1)
    bot = torch.cat((h, -h), 1)
    return torch.cat((top, bot), 0)

def _hadamard_full(x: Tensor) -> Tensor:
    """Hadamard transform along the last axis (len must be power-of-2)."""
    n = x.size(-1)
    h = hadamard(n, dtype=x.dtype, device=x.device) / math.sqrt(n)
    return x @ h.T

# --------------------------------------------------------------------
def core_residual(x: Tensor, P_r: Tensor) -> Tensor:
    """Remove the rank-r core component from vector x (x, P_r in same gauge)."""
    # (… , d_k)  –  (d_k , r_k)  →  (… , r_k)
    coeff = x @ P_r
    # (… , r_k)  –>  (… , d_k)
    return x - coeff @ P_r.T

# --------------------------------------------------------------------
class PowerMap(torch.nn.Module):
    """φ_α(x)  ≔ |H x| ⊙ scale   where scale_i = i^{-α/2}  (tail only)."""

    def __init__(self, alpha: float, r_k: int, d_k: int):
        super().__init__()
        self.register_buffer(
            "scale",
            torch.arange(d_k, dtype=torch.float32) ** (-alpha / 2.0)
        )
        self.scale[:r_k] = 0.0     # zero-out the core part
        self.d_k: Final = d_k
        self.r_k: Final = r_k

    def forward(self, x_tail: Tensor) -> Tensor:
        # x_tail shape (… , d_k)  where first r_k coords are (approx.) 0
        h = _hadamard_full(x_tail.to(self.scale.device))  # ensure device match before matmul
        # --- FIX: Mask core components AFTER Hadamard ---
        h[..., :self.r_k] = 0.0
        return h.abs() * self.scale