# Tensor Product Attention (TPA) for Gemma Models

This implementation adds Tensor Product Attention (TPA) support to Gemma models. TPA is a matrix factorization technique that significantly reduces KV cache memory usage during inference, while maintaining model quality.

## Overview

TPA works by factorizing the attention matrices (Q, K, V) into lower-rank representations:

- Instead of storing full K and V matrices of size `[batch_size, seq_len, num_heads, head_dim]`
- TPA stores factorized components of size `[batch_size, seq_len, num_heads, k_rank]` and `[batch_size, seq_len, k_rank, head_dim]`

This approach can reduce KV cache memory usage by 5-10x, enabling much longer context lengths with the same memory budget.

## Features

- Support for Gemma 1B, 4B, 12B, and 27B models
- Configurable rank parameters for Q, K, and V matrices
- Compatible with both text-only and multimodal Gemma models
- Preserves Rotary Positional Embedding (RoPE) which is critical for performance

## Usage

### Testing TPA with Gemma 1B

The simplest way to test TPA is with the `run_test_tpa_1b.py` script:

```bash
python run_test_tpa_1b.py \
  --ckpt /path/to/gemma_1b_model.ckpt \
  --tokenizer /path/to/tokenizer.model \
  --output tpa_model.pt \
  --device cuda \
  --prompt "Your test prompt here" \
  --q-rank 6 \
  --k-rank 2 \
  --v-rank 2
```

This will:
1. Load a standard Gemma 1B model
2. Convert it to TPA format
3. Save the TPA model
4. Run a test inference
5. Report memory usage and speed metrics

### For Production Use

For more advanced usage, you can use the `run_tpa.py` script in the `scripts` directory:

```bash
python scripts/run_tpa.py \
  --ckpt /path/to/gemma_model.ckpt \
  --variant 1b \  # Options: 1b, 4b, 12b, 27b
  --prompt "Write a poem about mathematics" \
  --convert \
  --save_tpa /path/to/save/tpa_model.pt \
  --device cuda \
  --q-rank 6 \
  --k-rank 2 \
  --v-rank 2
```

## Rank Configuration

The rank parameters control the compression level and quality tradeoff:

- `q_rank`: Rank for query matrices (default: 6)
- `k_rank`: Rank for key matrices (default: 2)
- `v_rank`: Rank for value matrices (default: 2)

Higher ranks provide better quality but less compression, while lower ranks offer more compression but may impact quality.

## Technical Implementation

The TPA implementation factorizes attention matrices using Singular Value Decomposition (SVD):

1. At model conversion time, the QKV projection weights are factorized using SVD
2. During inference, the factorized components A and B are used to compute attention
3. KV cache stores the compact factorized representations instead of full matrices

## Performance Considerations

- Memory usage is reduced by approximately `head_dim / ((k_rank + v_rank) / 2)` times
- For a typical Gemma model with 128 head dimensions and ranks of k_rank=2, v_rank=2, the reduction is ~32x
- The main performance cost is in the SVD factorization during model conversion
- Inference can actually be faster due to reduced memory movement

## References

- [TPA Paper: "Tensor Product Attention Is All You Need"](https://arxiv.org/abs/2501.06425)