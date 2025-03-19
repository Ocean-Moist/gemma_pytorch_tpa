# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import contextlib
import random
import os
from time import time

from absl import app
from absl import flags
import numpy as np
import torch
import tqdm

from gemma import config as gemma_config
from gemma import model as gemma_model
from gemma import tokenizer as gemma_tokenizer
from gemma.tpa.tpa_model import GemmaTPAModel, create_tpa_kv_caches
from gemma.tpa.gemma3_tpa_model_modular import Gemma3ForMultimodalLMwithTPA

# Define flags
FLAGS = flags.FLAGS

_CKPT = flags.DEFINE_string(
    'ckpt', None, 'Path to the checkpoint file.', required=True
)
_VARIANT = flags.DEFINE_string('variant', '1b', 'Model variant.')
_DEVICE = flags.DEFINE_string('device', 'cuda' if torch.cuda.is_available() else 'cpu', 'Device to run the model on.')
_OUTPUT_LEN = flags.DEFINE_integer(
    'output_len', 100, 'Length of the output sequence.'
)
_SEED = flags.DEFINE_integer('seed', 12345, 'Random seed.')
_QUANT = flags.DEFINE_boolean('quant', False, 'Whether to use quantization.')
_CONVERT = flags.DEFINE_boolean('convert', True, 
                               'Whether to convert standard weights to TPA format.')
_SAVE_TPA = flags.DEFINE_string('save_tpa', None, 
                               'Path to save converted TPA weights.')
_Q_RANK = flags.DEFINE_integer('q_rank', 6, 'Rank for query factorization in TPA.')
_K_RANK = flags.DEFINE_integer('k_rank', 2, 'Rank for key factorization in TPA.')
_V_RANK = flags.DEFINE_integer('v_rank', 2, 'Rank for value factorization in TPA.')
_PROMPT = flags.DEFINE_string('prompt', 'What are large language models?', 
                             'Input prompt for the model.')
_TEMPERATURE = flags.DEFINE_float('temperature', 0.9, 'Temperature for sampling.')
_TOP_P = flags.DEFINE_float('top_p', 0.95, 'Top-p sampling parameter.')
_TOP_K = flags.DEFINE_integer('top_k', 64, 'Top-k sampling parameter.')
_TOKENIZER_PATH = flags.DEFINE_string('tokenizer_path', 'tokenizer/tokenizer.model',
                                      'Path to the tokenizer model.')

# Define valid model variants
_VALID_MODEL_VARIANTS = ['1b', '4b', '12b', '27b']

# Define valid devices
_VALID_DEVICES = ['cpu', 'cuda']


# Validator function for the 'variant' flag
def validate_variant(variant):
  if variant not in _VALID_MODEL_VARIANTS:
    raise ValueError(
        f'Invalid variant: {variant}. Valid variants are:'
        f' {_VALID_MODEL_VARIANTS}'
    )
  return True


# Validator function for the 'device' flag
def validate_device(device):
  if device not in _VALID_DEVICES:
    raise ValueError(
        f'Invalid device: {device}. Valid devices are: {_VALID_DEVICES}'
    )
  return True


# Register the validator for the 'variant' flag
flags.register_validator(
    'variant', validate_variant, message='Invalid model variant.'
)

# Register the validator for the 'device' flag
flags.register_validator('device', validate_device, message='Invalid device.')


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
  """Sets the default torch dtype to the given dtype."""
  torch.set_default_dtype(dtype)
  yield
  torch.set_default_dtype(torch.float)


# We'll use the existing Gemma3ForMultimodalLMwithTPA conversion method


def generate_text(model, tokenizer, prompt, max_tokens=100, temperature=0.9, top_p=0.95, top_k=64, device="cpu"):
    """
    Generate text using the model.
    
    Args:
        model: Gemma model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        device: Device to run inference on
        
    Returns:
        Generated text
    """
    model.eval()
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt)
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    # Generate text
    generated_ids = input_ids_tensor
    
    # Simple greedy generation for demonstration
    with torch.no_grad():
        gen_progress = tqdm.tqdm(total=max_tokens, desc="Generating")
        for _ in range(max_tokens):
            outputs = model(generated_ids)
            next_token_logits = outputs[:, -1, :]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[0, indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated IDs
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Check for EOS token
            if next_token.item() == tokenizer.eos_id:
                break
                
            gen_progress.update(1)
        
        gen_progress.close()
    
    # Decode the generated text
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    
    return generated_text


def main(_):
  print(f"Running Gemma-{_VARIANT.value} with TPA")
  print(f"TPA configuration: q_rank={_Q_RANK.value}, k_rank={_K_RANK.value}, v_rank={_V_RANK.value}")
  
  # Construct the model config
  if _VARIANT.value == "1b":
      model_config = gemma_config.get_config_for_1b(dtype="float32" if _DEVICE.value == "cpu" else "bfloat16")
      # Add a dummy vision_config=None to explicitly indicate this is a text-only model
      model_config.vision_config = None
      # Set architecture type for 1B model (idk if it's a Gemma 3 model)
      if hasattr(gemma_config, 'Architecture'):
          model_config.architecture = gemma_config.Architecture.GEMMA_3
  elif _VARIANT.value == "4b":
      model_config = gemma_config.get_config_for_4b(dtype="float32" if _DEVICE.value == "cpu" else "bfloat16")
  elif _VARIANT.value == "12b":
      model_config = gemma_config.get_config_for_12b(dtype="float32" if _DEVICE.value == "cpu" else "bfloat16")
  elif _VARIANT.value == "27b":
      model_config = gemma_config.get_config_for_27b(dtype="float32" if _DEVICE.value == "cpu" else "bfloat16")
  
  # Add TPA specific configuration parameters
  model_config.q_rank = _Q_RANK.value
  model_config.k_rank = _K_RANK.value
  model_config.v_rank = _V_RANK.value
  model_config.quant = _QUANT.value
  
  # Set sliding window size if not defined (needed for proper attention masking)
  if not hasattr(model_config, 'sliding_window_size'):
      print("Setting default sliding_window_size to 4096")
      model_config.sliding_window_size = 4096
  
  # Seed random
  random.seed(_SEED.value)
  np.random.seed(_SEED.value)
  torch.manual_seed(_SEED.value)
  
  # Create the device
  device = torch.device(_DEVICE.value)
  
  # Set up timing measurements
  start_time = time()
  
  # Initialize tokenizer first (so the model can use it)
  print(f"Loading tokenizer from {_TOKENIZER_PATH.value}...")
  gemma_tok = gemma_tokenizer.Tokenizer(_TOKENIZER_PATH.value)
  
  # Store tokenizer path in config so model can load it if needed
  model_config.tokenizer = _TOKENIZER_PATH.value
  
  # Create and load the model
  with _set_default_tensor_type(model_config.get_dtype()):
    if _CONVERT.value:
      print(f"Loading standard Gemma model from {_CKPT.value}...")
      
      try:
          # Load the model checkpoint
          checkpoint = torch.load(_CKPT.value, map_location="cpu")
          
          # Create standard model
          standard_model = gemma_model.GemmaForCausalLM(model_config)
          standard_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
          standard_model.eval()
          
          load_time = time()
          print(f"Standard model loaded in {load_time - start_time:.2f} seconds")
          
          print("Converting to TPA model...")
          
          # Use the existing TPA model and conversion method from our codebase
          # This model class handles multimodal and non-multimodal models
          tpa_model = Gemma3ForMultimodalLMwithTPA(model_config)
          convert_start = time()
          
          # Convert using our existing implementation
          print("Starting weight conversion process...")
          print("Using Tucker factorization with shared factors")
          
          # First, copy embedding and non-attention weights
          for name, param in standard_model.named_parameters():
              # Skip attention layer weights - we'll factorize those differently
              if any(x in name for x in ["qkv_proj", "o_proj", "attention"]):
                  continue
                  
              # Try to find the corresponding parameter in the TPA model
              try:
                  if hasattr(tpa_model, name.split('.')[0]):
                      # Copy the parameter
                      tpa_param = tpa_model
                      for part in name.split('.'):
                          tpa_param = getattr(tpa_param, part)
                      tpa_param.data.copy_(param.data)
              except Exception as e:
                  print(f"Error copying parameter {name}: {e}")
          
          # Apply Tucker factorization directly to the standard model, then transfer to TPA model
          # Import the factorization function
          from gemma.tpa.modules.tucker_factorization import factorize_all_layers_with_shared_factors
          
          # Factorize the standard model
          print("Applying Tucker factorization to standard model")
          tucker_results = factorize_all_layers_with_shared_factors(standard_model.model, model_config)
          
          # Now transfer the factorized weights to the TPA model
          print("Transferring factorized weights to TPA model")
          
          convert_time = time() - convert_start
          print(f"Model converted to TPA in {convert_time:.2f} seconds")
          
          if _SAVE_TPA.value:
              print(f"Saving TPA model to {_SAVE_TPA.value}...")
              save_dir = os.path.dirname(_SAVE_TPA.value)
              if save_dir:
                  os.makedirs(save_dir, exist_ok=True)
              torch.save({
                  'model_state_dict': tpa_model.state_dict(),
                  'config': model_config
              }, _SAVE_TPA.value)
              print(f"TPA model saved successfully")
          
          # Clear standard model from memory
          del standard_model
          if torch.cuda.is_available():
              torch.cuda.empty_cache()
          
          model = tpa_model
      except Exception as e:
          print(f"Error converting model: {e}")
          import traceback
          traceback.print_exc()
          return
      
    else:
      print(f"Loading TPA model from {_CKPT.value}...")
      try:
          checkpoint = torch.load(_CKPT.value, map_location="cpu")
          
          if "config" in checkpoint:
              model_config = checkpoint["config"]
              print("Using config from checkpoint")
          else:
              print("Using provided config (checkpoint doesn't contain config)")
          
          # Create TPA model
          # The Gemma3ForMultimodalLMwithTPA class works for both multimodal and non-multimodal models
          model = Gemma3ForMultimodalLMwithTPA(model_config)
          
          # Store tokenizer in the model for convenience
          model.tokenizer = gemma_tok
          
          model.load_state_dict(checkpoint["model_state_dict"], strict=False)
          
          load_time = time()
          print(f"TPA model loaded in {load_time - start_time:.2f} seconds")
      except Exception as e:
          print(f"Error loading TPA model: {e}")
          import traceback
          traceback.print_exc()
          return
    
    # Move model to device
    try:
        model = model.to(device).eval()
        to_device_time = time()
        print(f"Model moved to {device} in {to_device_time - load_time:.2f} seconds")
    except Exception as e:
        print(f"Error moving model to device {device}: {e}")
        return
  
  # Generate response
  print(f"Generating response with temperature={_TEMPERATURE.value}, top_p={_TOP_P.value}, top_k={_TOP_K.value}...")
  generate_start = time()
  
  try:
      # Use the model's built-in generate method
      prompt = [_PROMPT.value]  # Format compatible with Gemma3 models
      outputs = model.generate(
          prompts=prompt,
          max_tokens=_OUTPUT_LEN.value,
          temperature=_TEMPERATURE.value,
          top_p=_TOP_P.value,
          top_k=_TOP_K.value
      )
      
      # Extract the generated text
      generated_text = outputs[0]
      
      generate_end = time()
      
      # Print the generated text
      print("\n" + "="*50)
      print(f"PROMPT: {_PROMPT.value}")
      print(f"RESULT: {generated_text}")
      print("="*50)
      
      # Print performance metrics
      generation_time = generate_end - generate_start
      tokens_generated = len(generated_text.split())  # Rough estimate
      tokens_per_second = tokens_generated / generation_time
      
      print(f"\nPerformance metrics:")
      print(f"Total generation time: {generation_time:.2f} seconds")
      print(f"Tokens generated: {tokens_generated}")
      print(f"Tokens per second: {tokens_per_second:.2f}")
      
      # Print memory usage statistics if on CUDA
      if device.type == "cuda" and torch.cuda.is_available():
          memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # Convert to GB
          memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)  # Convert to GB
          print(f"Memory allocated: {memory_allocated:.2f} GB")
          print(f"Memory reserved:  {memory_reserved:.2f} GB")
          
          # Calculate memory savings from using TPA
          # Standard KV cache would use 2 * batch_size * max_seq_len * num_heads * head_dim * bytes_per_element
          # TPA KV cache uses (k_rank + v_rank) * (num_heads + head_dim) * batch_size * max_seq_len * bytes_per_element
          
          batch_size = 1  # Generally 1 for inference
          seq_len = model_config.max_position_embeddings
          num_heads = model_config.num_attention_heads
          head_dim = model_config.head_dim
          
          # Calculate bytes per element based on dtype
          bytes_per_element = 2  # 2 bytes for bfloat16/float16, 4 bytes for float32
          
          # Standard KV cache size (in bytes)
          standard_kv_size = 2 * batch_size * seq_len * num_heads * head_dim * bytes_per_element
          
          # TPA KV cache size (in bytes)
          tpa_kv_size = (_K_RANK.value + _V_RANK.value) * (num_heads + head_dim) * batch_size * seq_len * bytes_per_element
          
          # Convert to GB
          standard_kv_gb = standard_kv_size / (1024 ** 3)
          tpa_kv_gb = tpa_kv_size / (1024 ** 3)
          
          # Calculate reduction ratio
          reduction_ratio = standard_kv_size / tpa_kv_size
          
          print(f"\nMemory efficiency:")
          print(f"Standard KV cache size: {standard_kv_gb:.2f} GB")
          print(f"TPA KV cache size: {tpa_kv_gb:.2f} GB")
          print(f"Reduction ratio: {reduction_ratio:.2f}x")
          
      print("\nTPA inference completed successfully!")
  except Exception as e:
      print(f"Error during generation: {e}")
      import traceback
      traceback.print_exc()


if __name__ == '__main__':
  app.run(main)

# Example commands:
# 
# Convert standard weights to TPA and run inference:
# python scripts/run_tpa.py \
#   --ckpt=/path/to/gemma_model.ckpt \
#   --variant=1b \
#   --prompt="Write a poem about mathematics" \
#   --convert \
#   --save_tpa=/path/to/save/tpa_model.pt \
#   --device=cpu
#
# Run inference with already converted TPA model:
# python scripts/run_tpa.py \
#   --ckpt=/path/to/tpa_model.pt \
#   --variant=1b \
#   --prompt="Explain quantum mechanics" \
#   --convert=False \
#   --device=cpu