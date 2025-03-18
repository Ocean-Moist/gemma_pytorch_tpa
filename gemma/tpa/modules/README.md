# Modular TPA Implementation for Gemma Models

This directory contains modular components for implementing Tensor Product Attention (TPA)
in Gemma models. The implementation follows the approaches from the "Tensor Product Attention Is All You Need" (T6) paper
and the TensorLLM approach to tensor factorization.

## Module Structure

The TPA implementation is organized into the following modules:

### 1. Tucker Factorization (`tucker_factorization.py`)

This module implements the TensorLLM approach to attention weight factorization using Tucker decomposition
with shared factor matrices. Key functions include:

- `factorize_all_layers_with_shared_factors`: Apply Tucker decomposition to all transformer layers
- `_factorize_mha_weights_with_shared_factors`: Factorize multi-head attention weights using shared factors
- `adaptive_rank_selection`: Automatically determine optimal ranks for Tucker decomposition
- `_factorize_and_set_weights`: Factorize a weight matrix and set the factorized weights

### 2. Contextual Factorization (`contextual_factorization.py`)

This module implements the T6-style contextual tensor factorization for queries, keys, and values.
Key functions include:

- `contextual_tensor_decomposition`: Factorize weight matrices using the T6 approach
- `_init_contextual_factorization`: Initialize and optimize factor matrices
- `apply_contextual_tensor_decomposition`: Apply factorization to all model layers
- `convert_from_standard_weights`: Convert standard Gemma weights to TPA format

### 3. Tensor Product Utilities (`tensor_product_utils.py`)

This module provides utility functions for working with tensor product operations:

- `register_freqs_cis`: Create and register rotary position embedding frequencies
- `create_attention_mask`: Create attention masks with support for sliding windows
- `populate_image_embeddings`: Insert image embeddings into hidden states
- `reshape_for_broadcast`: Reshape tensors for efficient broadcasting

## Usage

The modular implementation can be used through the `Gemma3ForMultimodalLMwithTPA` class in `gemma3_tpa_model_modular.py`,
which integrates these components. The main class exposes methods for factorizing model weights and
performing inference with TPA.

Example:

```python
from gemma.tpa.gemma3_tpa_model_modular import Gemma3ForMultimodalLMwithTPA

# Initialize model
model = Gemma3ForMultimodalLMwithTPA(config)

# Load pre-trained weights
model.load_weights("path/to/weights")

# Apply TPA factorization
model.factorize_all_layers_with_shared_factors()

# Generate text
outputs = model.generate(["Your prompt here"], max_tokens=128)
```

## Advantages of Modular Structure

1. **Better Code Organization**: Clear separation of concerns makes the code easier to understand and maintain
2. **Improved Testing**: Modular functions can be tested independently
3. **Easier Extensions**: New factorization approaches can be added without modifying the core model
4. **Reduced File Size**: Smaller, focused files instead of a single large file
5. **Clearer API**: Well-defined interfaces between components