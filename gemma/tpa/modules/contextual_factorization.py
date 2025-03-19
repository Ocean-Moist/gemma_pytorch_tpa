"""
T6-style contextual factorization methods for Tensor Product Attention.

This module serves as the main entry point for tensor factorization in TPA models,
importing and exposing functionality from specialized submodules.
"""

import torch
from typing import Dict, List, Tuple, Optional, Union, Any

# Import from specialized modules
from .svd_utils import patched_svd, randomized_svd_low_memory, randomized_svd_tiled, HAS_TENSORLY
from .tucker_decomposition import memory_efficient_tucker, tile_based_tucker, unfold_tensor, mode_dot
from .tensor_factorization import (
    contextual_tensor_decomposition,
    tucker_tensor_decomposition,
    direct_tensorly_tucker_decomposition,
    shared_factors_tucker_decomposition,
    _init_contextual_factorization
)
from .model_conversion import apply_contextual_tensor_decomposition, convert_from_standard_weights

# Export all important functions
__all__ = [
    # SVD utilities
    "patched_svd",
    "randomized_svd_low_memory",
    "randomized_svd_tiled",
    "HAS_TENSORLY",
    
    # Tucker decomposition
    "memory_efficient_tucker",
    "tile_based_tucker",
    "unfold_tensor",
    "mode_dot",
    
    # Core factorization algorithms
    "contextual_tensor_decomposition",
    "tucker_tensor_decomposition",
    "direct_tensorly_tucker_decomposition",
    "shared_factors_tucker_decomposition",
    "_init_contextual_factorization",
    
    # Model conversion
    "apply_contextual_tensor_decomposition",
    "convert_from_standard_weights"
]