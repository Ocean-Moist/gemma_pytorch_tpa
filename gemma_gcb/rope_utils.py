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
    #  (B , d_k)  →  (B , 1 , 1 , d_k)  so that
    #  dim order matches  (batch , heads , seq , head_dim)
    x_phys = (x_g @ A_invT).view(x_g.size(0), 1, 1, -1)
    
    #  apply RoPE in the physical basis
    x_rot  = apply_rotary_emb(x_phys, freqs_row)      # still 4-D
    
    #  flatten back to (B , d_k) before returning to the gauge flow
    return x_rot.view(x_rot.size(0), -1)