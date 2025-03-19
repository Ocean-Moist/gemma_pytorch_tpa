"""
Modules for Gemma3 TPA implementation.

This package provides specialized modules for Tensor Product Attention (TPA)
implementations for Gemma3 models, including tensor factorization, SVD utilities,
and model conversion utilities.
"""

# Import all important functions from main module for backward compatibility
from .contextual_factorization import (
    contextual_tensor_decomposition,
    tucker_tensor_decomposition,
    _init_contextual_factorization,
    apply_contextual_tensor_decomposition,
    convert_from_standard_weights,
    memory_efficient_tucker,
    tile_based_tucker,
    unfold_tensor,
    mode_dot
)

# SVD utilities
from .svd_utils import (
    patched_svd,
    randomized_svd_low_memory,
    randomized_svd_tiled
)

# Utility functions from tensor_product_utils
from .tensor_product_utils import (
    register_freqs_cis,
    create_attention_mask,
    populate_image_embeddings
)

__all__ = [
    # Core tensor factorization
    "contextual_tensor_decomposition",
    "tucker_tensor_decomposition",
    "_init_contextual_factorization",
    
    # Tucker decomposition utilities
    "memory_efficient_tucker",
    "tile_based_tucker",
    "unfold_tensor",
    "mode_dot",
    
    # SVD utilities
    "patched_svd",
    "randomized_svd_low_memory",
    "randomized_svd_tiled",
    
    # Model conversion
    "apply_contextual_tensor_decomposition",
    "convert_from_standard_weights",
    
    # Tensor product utilities
    "register_freqs_cis",
    "create_attention_mask",
    "populate_image_embeddings"
]