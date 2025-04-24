# -*- coding: utf-8 -*-
"""Helpers for applying RoPE in the *physical* basis while caching in gauge."""
import torch
from gemma.model import apply_rotary_emb      # reuse original util

def apply_rope_phys(x_g, A_invT, freqs_row):
    """
    x_g       : (B , d_k) in the gauge basis
    A_invT    : (d_k , d_k)   gauge → phys
    freqs_row : (1 , d_k/2 , 2)   row from the usual RoPE table
    returns   : (B , d_k) in the *rotated gauge* basis
    """
    x_phys = x_g @ A_invT                # back to physical
    x_phys = x_phys.view(*x_phys.shape[:-1], 1, -1)   # reshape for util
    x_rot  = apply_rotary_emb(x_phys, freqs_row)
    return x_rot.view_as(x_g)            # keep gauge-shape interface