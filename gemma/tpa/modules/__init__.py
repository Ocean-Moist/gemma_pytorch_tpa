"""
Modules for Gemma3 TPA implementation.

This package provides specialized modules for Tensor Product Attention (TPA)
implementations for Gemma3 models, including tensor factorization, SVD utilities,
and model conversion utilities.
"""
from .gqa_to_tpa import split_combined_qkv_weights
from .gqa_to_tpa import prepare_isp_kv_components