"""
Modules for Gemma3 TPA implementation.

This package provides specialized modules for Tensor Product Attention (TPA)
implementations for Gemma3 models, including tensor factorization, SVD utilities,
and model conversion utilities.
"""

from .gqa_to_tpa import compute_svd_tpa_factors