"""
Modules for Gemma3 TPA implementation.

This package provides specialized modules for Tensor Product Attention (TPA)
implementations for Gemma3 models.
"""

from .tucker_factorization import (
    _factorize_mha_weights_with_shared_factors,
    factorize_all_layers_with_shared_factors,
    adaptive_rank_selection,
    _factorize_and_set_weights
)

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
    "_factorize_mha_weights_with_shared_factors",
    "factorize_all_layers_with_shared_factors",
    "adaptive_rank_selection",
    "_factorize_and_set_weights",
    "contextual_tensor_decomposition",
    "_init_contextual_factorization",
    "apply_contextual_tensor_decomposition",
    "convert_from_standard_weights",
    "register_freqs_cis",
    "create_attention_mask",
    "populate_image_embeddings"
]