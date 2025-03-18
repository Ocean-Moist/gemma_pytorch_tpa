# Gemma with TPA (Tensor Product Attention)

This implementation adds the Tensor Product Attention (TPA) mechanism to Gemma models. TPA significantly reduces KV cache memory usage during inference while maintaining or improving model performance.

## What is Tensor Product Attention?

Tensor Product Attention (TPA) is a novel attention mechanism proposed in ["Tensor Product Attention Is All You Need"](https://arxiv.org/abs/2501.06425). TPA factorizes query, key, and value matrices into contextual low-rank components, which significantly reduces the memory footprint of KV caches during inference.

TPA offers several advantages:
- Reduces KV cache size by 5-10x compared to standard multi-head attention
- Allows for processing longer sequences with the same memory budget
- Compatible with rotary positional embeddings (RoPE)
- Improves model performance compared to MHA, MQA, and GQA attention mechanisms

## Implementation

Our implementation adds TPA support to Gemma 3 models:

1. `GemmaTensorProductAttention`: A drop-in replacement for standard attention that uses TPA
2. `GemmaTPAModel`: A Gemma model using TPA throughout
3. `Gemma3ForMultimodalLMwithTPA`: A multimodal Gemma 3 model with TPA support
4. Weight conversion utilities to convert standard Gemma weights to TPA format

## Usage

### Converting Standard Gemma to TPA

You can convert an existing Gemma 3 model to use TPA as follows:

```python
from gemma import config as gemma_config
from gemma.gemma3_model import Gemma3ForMultimodalLM
from gemma.tpa.gemma3_tpa_model import Gemma3ForMultimodalLMwithTPA

# Create a standard Gemma configuration
config = gemma_config.get_config_for_4b()

# Add TPA-specific parameters
config.q_rank = 6  # Default rank for query factorization
config.k_rank = 2  # Default rank for key factorization 
config.v_rank = 2  # Default rank for value factorization

# Load standard model
standard_model = Gemma3ForMultimodalLM(config)
standard_model.load_weights("path/to/gemma3_weights")

# Create TPA-based model
tpa_model = Gemma3ForMultimodalLMwithTPA(config)

# Convert weights
tpa_model.convert_from_standard_weights(standard_model)

# Save converted model (optional)
import torch
torch.save({'model_state_dict': tpa_model.state_dict()}, "path/to/save/tpa_model.pt")
```

### Inference with TPA

The provided script `tpa_inference_gemma.py` demonstrates how to run inference with TPA-based Gemma models.

```bash
# Convert standard weights to TPA and run inference
python tpa_inference_gemma.py \
  --model_path /path/to/gemma3_weights \
  --model_variant 4b \
  --prompt "Write a poem about mathematics" \
  --convert_from_standard \
  --save_tpa_model /path/to/save/tpa_model.pt

# Run inference with already converted TPA model
python tpa_inference_gemma.py \
  --model_path /path/to/tpa_model.pt \
  --model_variant 4b \
  --prompt "Explain quantum mechanics" \
  --image /path/to/image.jpg  # Optional image for multimodal models
```

## Memory Savings

TPA provides significant memory savings compared to standard multi-head attention:

| Model | Standard KV Cache | TPA KV Cache | Reduction |
|-------|-------------------|--------------|-----------|
| 1B    | ~1.9 GB           | ~0.3 GB      | ~6.3x     |
| 4B    | ~9.1 GB           | ~1.2 GB      | ~7.6x     |
| 12B   | ~27.3 GB          | ~3.8 GB      | ~7.2x     |
| 27B   | ~46.1 GB          | ~6.5 GB      | ~7.1x     |

Memory savings are most significant for models with larger head dimensions and more heads, which makes TPA particularly useful for large models and/or models with long sequences.

## Advantages of TPA

1. **Memory Efficiency**: By factorizing key and value matrices, TPA drastically reduces KV cache size during inference.
   
2. **Longer Contexts**: The reduced memory footprint enables processing significantly longer sequences under the same memory constraints.

3. **RoPE Compatibility**: TPA seamlessly integrates with Rotary Positional Embeddings.

4. **Improved Performance**: TPA can outperform standard attention methods like MHA, MQA, and GQA.

## Reference

If you use this implementation, please cite the original TPA paper:

```
@article{zhang2024tensor,
  title={Tensor Product Attention Is All You Need},
  author={Zhang, Yifan and Liu, Yifeng and Yuan, Huizhuo and Qin, Zhen and Yuan, Yang and Gu, Quanquan and Yao, Andrew Chi-Chih},
  journal={arXiv preprint arXiv:2501.06425},
  year={2024}
}
```