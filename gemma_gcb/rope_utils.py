# -*- coding: utf-8 -*-
"""Helpers for applying RoPE in the physical basis with proper gauge transformations."""
import torch
from gemma.model import apply_rotary_emb      # reuse original util

def apply_rope_query(q_phys, freqs_row):
    """
    Apply RoPE to the query in physical space
    q_phys     : (B , d_k) in the physical basis
    freqs_row  : (1 , d_k/2 , 2)   row from the usual RoPE table
    returns    : (B , d_k) in the physical basis with RoPE applied
    """
    #  (B , d_k)  →  (B , 1 , 1 , d_k)  to match
    #  (batch , heads , seq , head_dim) expected by apply_rotary_emb
    q_phys_reshaped = q_phys.view(q_phys.size(0), 1, 1, -1)
    
    #  apply RoPE in the physical basis
    q_rot_phys = apply_rotary_emb(q_phys_reshaped, freqs_row)      # still 4-D
    
    #  flatten back to (B , d_k)
    return q_rot_phys.view(q_rot_phys.size(0), -1)

def apply_rope_key(k_phys, freqs_row):
    """
    Apply RoPE to the key in physical space
    k_phys     : (B , d_k) in the physical basis
    freqs_row  : (1 , d_k/2 , 2)   row from the usual RoPE table
    returns    : (B , d_k) in the physical basis with RoPE applied
    """
    #  (B , d_k)  →  (B , 1 , 1 , d_k)  to match
    #  (batch , heads , seq , head_dim) expected by apply_rotary_emb
    k_phys_reshaped = k_phys.view(k_phys.size(0), 1, 1, -1)
    
    #  apply RoPE in the physical basis
    k_rot_phys = apply_rotary_emb(k_phys_reshaped, freqs_row)      # still 4-D
    
    #  flatten back to (B , d_k)
    return k_rot_phys.view(k_rot_phys.size(0), -1)