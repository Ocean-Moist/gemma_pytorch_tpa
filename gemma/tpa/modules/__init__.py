"""
Modules for Gemma3 TPA implementation.

This package provides specialized modules for Tensor Product Attention (TPA)
implementations for Gemma3 models.
"""

from .contextual_factorization import (
    contextual_tensor_decomposition,
    _init_contextual_factorization,
    apply_contextual_tensor_decomposition,
    convert_from_standard_weights
)

from .tensor_product_utils import (
    register_freqs_cis,
    create_attention_mask,
    populate_image_embeddings
)

__all__ = [
    "contextual_tensor_decomposition",
    "_init_contextual_factorization",
    "apply_contextual_tensor_decomposition",
    "convert_from_standard_weights",
    "register_freqs_cis",
    "create_attention_mask",
    "populate_image_embeddings"
]