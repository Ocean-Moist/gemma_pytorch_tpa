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
from gemma.tpa.gemma3_tpa_model import Gemma3ForMultimodalLMwithTPA

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
_CUDA_LAUNCH_BLOCKING = flags.DEFINE_boolean('cuda_launch_blocking', False, 
                               'Whether to set CUDA_LAUNCH_BLOCKING=1 (debugging synchronous CUDA operations).')
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
_EXTRA_CONFIG = flags.DEFINE_string('extra_config', None, 
                                   'Extra configuration for the model in JSON format. E.g. \'{"use_tensorly": true}\'.')
_FAT_RANKS = flags.DEFINE_boolean('fat_ranks', False, 
                                 'Whether to use much larger ranks (240) for higher accuracy.')

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
          # Load the model checkpoint directly to the target device
          checkpoint = torch.load(_CKPT.value, map_location=device)
          
          # Create standard model
          standard_model = gemma_model.GemmaForCausalLM(model_config)
          standard_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
          standard_model = standard_model.to(device)  # Move to device immediately
          standard_model.eval()
          print(f"Standard model loaded and moved to {device}")
          
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
          # Special handling for embedding layer name mismatch
          if hasattr(standard_model, 'embedder') and hasattr(tpa_model, 'text_token_embedder'):
              print("Copying embedding weights from 'embedder' to 'text_token_embedder'")
              # Copy the weight tensor
              tpa_model.text_token_embedder.weight.data.copy_(standard_model.embedder.weight.data)
              # If using quantization, also copy the weight scaler
              if hasattr(standard_model.embedder, 'weight_scaler') and hasattr(tpa_model.text_token_embedder, 'weight_scaler'):
                  tpa_model.text_token_embedder.weight_scaler.data.copy_(standard_model.embedder.weight_scaler.data)
          
          # Now copy all other non-attention weights as before
          for name, param in standard_model.named_parameters():
              # Skip attention layer weights - we'll factorize those differently
              if any(x in name for x in ["qkv_proj", "o_proj", "attention"]):
                  continue
              # Skip embedder weights which we already copied manually
              if name.startswith("embedder"):
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
          
          # Import the factorization

          # Use the built-in convert_from_standard_weights method which now supports Tucker factorization
          q_rank = _Q_RANK.value
          k_rank = _K_RANK.value
          v_rank = _V_RANK.value
          print(f"Ranks: Q={q_rank}, K={k_rank}, V={v_rank}")
          
          # Use shared factors approach with TensorLy
          if True:
              # Parse extra configuration if provided
              extra_config = {}
              if _EXTRA_CONFIG.value:
                  try:
                      import json
                      extra_config = json.loads(_EXTRA_CONFIG.value)
                      print(f"Using extra configuration: {extra_config}")
                  except json.JSONDecodeError as e:
                      print(f"Error parsing extra_config: {e}, using default configuration")
              
              # Get factorization method from extra_config or use default
              factorization_method = extra_config.get("factorization_method", "shared_factors")
              
              if factorization_method == "direct_tensorly":
                  print("Using direct TensorLy Tucker decomposition")
                  # Set up target ranks for direct TensorLy implementation
                  tpa_model.target_ranks = {
                      "use_tensorly": True,
                      "use_shared_factors": False,
                      "q_rank": q_rank,
                      "k_rank": k_rank,
                      "v_rank": v_rank
                  }
              elif factorization_method == "shared_factors":
                  print("Using shared factors approach for Tucker decomposition")
                  # Set up target ranks with shared factors configuration
                  tpa_model.target_ranks = {
                      "use_shared_factors": True,
                      "hidden_rank": extra_config.get("hidden_rank", 8),
                      "head_rank": extra_config.get("head_rank", 4),
                      "dim_rank": extra_config.get("dim_rank", 4),
                      "q_rank": q_rank,
                      "k_rank": k_rank,
                      "v_rank": v_rank
                  }
              elif factorization_method == "contextual":
                  print("Using original contextual factorization (T6-style)")
                  # No target_ranks needed for contextual factorization
                  # Will fall back to this automatically if TensorLy isn't available
                  pass
              elif factorization_method == "gqa_to_tpa":
                  print("Using GQA to TPA conversion via Tucker decomposition")
                  # Import the GQA to TPA conversion
                  from gemma.tpa.modules.gqa_to_tpa import convert_gqa_model_to_tpa
                  # Set CUDA device explicitly for this process
                  if torch.cuda.is_available():
                      # Get the CUDA device index
                      if isinstance(device, torch.device) and device.index is not None:
                          cuda_device_idx = device.index
                      else:
                          cuda_device_idx = 0  # Default to first CUDA device
                      torch.cuda.set_device(cuda_device_idx)
                      print(f"Explicitly set CUDA device to cuda:{cuda_device_idx}")
                  
                  # Flag to use this special conversion
                  tpa_model.use_gqa_to_tpa = True
                  
                  # Specific conversion will be applied after normal model loading
                  tpa_model.gqa_config = {
                      "q_rank": q_rank,
                      "k_rank": k_rank,
                      "v_rank": v_rank
                  }
              else:
                  print(f"Unknown factorization method: {factorization_method}, using shared factors")
                  tpa_model.target_ranks = {
                      "use_shared_factors": True,
                      "q_rank": q_rank,
                      "k_rank": k_rank,
                      "v_rank": v_rank
                  }
          
          # Make sure model config has correct ranks
          tpa_model.config.q_rank = q_rank
          tpa_model.config.k_rank = k_rank
          tpa_model.config.v_rank = v_rank
          
          # Check if we should use the special GQA to TPA conversion
          if hasattr(tpa_model, 'use_gqa_to_tpa') and tpa_model.use_gqa_to_tpa:
              # First, do a standard conversion to copy non-attention weights
              try:
                  # Temporarily assign a basic target_ranks to avoid errors
                  tpa_model.target_ranks = {
                      "use_shared_factors": True,
                      "q_rank": _Q_RANK.value,
                      "k_rank": _K_RANK.value,
                      "v_rank": _V_RANK.value 
                  }
                  
                  # Copy non-attention weights
                  for tpa_name, tpa_param in tpa_model.named_parameters():
                      if "attention" not in tpa_name:
                          # Find corresponding parameter in standard model
                          std_name = tpa_name
                          if std_name in standard_model.state_dict():
                              tpa_param.data.copy_(standard_model.state_dict()[std_name])
                  
                  # Now apply the GQA to TPA conversion
                  from gemma.tpa.modules.gqa_to_tpa import convert_gqa_model_to_tpa
                  
                  # Extract conversion parameters
                  q_rank = tpa_model.gqa_config["q_rank"]
                  k_rank = tpa_model.gqa_config["k_rank"]
                  v_rank = tpa_model.gqa_config["v_rank"]
                  
                  # Apply the actual conversion
                  print("Applying GQA to TPA conversion...")
                  # Check if dynamic ranks should be used
                  use_dynamic_ranks = extra_config.get("use_dynamic_ranks", True)
                  print(f"Using dynamic ranks: {use_dynamic_ranks}")
                  
                  # Import the new standalone function
                  from gemma.tpa.modules.gqa_to_tpa import create_tpa_model_from_standard
                  
                  # Print device details before conversion
                  if torch.cuda.is_available():
                      print(f"CUDA memory before conversion: {torch.cuda.memory_allocated(device)/(1024**3):.2f} GB allocated")
                      print(f"CUDA memory reserved: {torch.cuda.memory_reserved(device)/(1024**3):.2f} GB")
                  
                  # Create a new TPA model from the standard model using our standalone function
                  # This function handles all the complexities of creating a proper TPA model
                  print(f"INFO: standard_model.config hidden_size = {standard_model.config.hidden_size}")
                  print(f"INFO: tpa_model.config hidden_size = {tpa_model.config.hidden_size}")
                  
                  # Force preserving the config's hidden_size during conversion
                  if standard_model.config.hidden_size != tpa_model.config.hidden_size:
                      print(f"CRITICAL ERROR: hidden_size mismatch: {standard_model.config.hidden_size} vs {tpa_model.config.hidden_size}")
                      print(f"Forcing standard_model.config.hidden_size to match tpa_model.config.hidden_size")
                      standard_model.config.hidden_size = tpa_model.config.hidden_size
                  
                  # Ensure device is correct before passing to create_tpa_model_from_standard
                  if torch.cuda.is_available():
                      print(f"Setting device to CUDA before conversion")
                      # Get device index for specificity
                      cuda_device_idx = 0  # Default to first CUDA device
                      device = torch.device(f'cuda:{cuda_device_idx}')  # Ensure we're using CUDA with specific index
                      # Force standard_model to CUDA
                      standard_model = standard_model.to(device)
                      
                  # Check if fat_ranks mode is enabled
                  if _FAT_RANKS.value:
                      print("Using FAT RANKS MODE with ranks of 240 for higher accuracy but more memory usage")
                      print("Warning: This will consume significantly more memory and computation time")
                  
                  # Convert model with explicit parameters
                  new_tpa_model = create_tpa_model_from_standard(
                      standard_model, 
                       q_rank=96,
                       k_rank=48,
                       v_rank=48,
                      dtype=tpa_model.dtype,
                      device=device,
                      use_dynamic_ranks=use_dynamic_ranks,
                      fat_ranks=_FAT_RANKS.value  # Pass the fat_ranks parameter
                  )
                  
                  # Replace the original TPA model with our properly created one
                  del tpa_model
                  torch.cuda.empty_cache()
                  tpa_model = new_tpa_model
                  
                  print("Successfully created TPA model with factorized weights")
                  
              except Exception as gqa_error:
                  print(f"Error in GQA to TPA conversion: {gqa_error}")
                  import traceback
                  traceback.print_exc()
                  
                  print("Falling back to standard conversion...")
                  tpa_model = tpa_model.convert_from_standard_weights(standard_model)
          else:
              # Convert using the built-in method that now uses Tucker factorization when available
              try:
                  tpa_model = tpa_model.convert_from_standard_weights(standard_model)
              except Exception as convert_error:
                  print(f"Error using Tucker factorization: {convert_error}")
                  print("Falling back to standard contextual factorization")
                  
                  # Force the use of standard factorization by temporarily disabling tensorly
                  import sys
                  orig_modules = sys.modules.copy()
                  if 'gemma.tpa.modules.contextual_factorization' in sys.modules:
                      mod = sys.modules['gemma.tpa.modules.contextual_factorization']
                      orig_has_tensorly = getattr(mod, 'HAS_TENSORLY', False)
                      setattr(mod, 'HAS_TENSORLY', False)
                  
                  # Retry conversion
                  try:
                      tpa_model = tpa_model.convert_from_standard_weights(standard_model)
                  except Exception as fallback_error:
                      print(f"Error in fallback conversion: {fallback_error}")
                      import traceback
                      traceback.print_exc()
                      raise
                  finally:
                      # Restore original HAS_TENSORLY value
                      if 'gemma.tpa.modules.contextual_factorization' in sys.modules:
                          mod = sys.modules['gemma.tpa.modules.contextual_factorization']
                          setattr(mod, 'HAS_TENSORLY', orig_has_tensorly)
          
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
      # Use the model's built-in generate method with the correct parameters
      # Format prompt for chat/instruction model using proper chat format
      formatted_prompt = f"<start_of_turn>user {_PROMPT.value}<end_of_turn>\n<start_of_turn>model"
      # Alternative simple prompt for debugging
      simple_prompt = _PROMPT.value
      
      # Choose the simpler prompt format for debugging
      print("Using simple prompt format for debugging")
      prompt = [simple_prompt]  # More likely to work with possibly broken tokenizer
      
      if hasattr(model, 'generate'):
          # Check which class the model is and use appropriate parameter names
          if isinstance(model, Gemma3ForMultimodalLMwithTPA):
              # For the TPA modular model which takes max_tokens
              print("Using Gemma3ForMultimodalLMwithTPA generate() interface")
              outputs = model.generate(
                  prompts=prompt,
                  max_tokens=_OUTPUT_LEN.value,
                  temperature=_TEMPERATURE.value,
                  top_p=_TOP_P.value,
                  top_k=_TOP_K.value
              )
          else:
              # For the standard GemmaForCausalLM model which takes output_len
              print("Using GemmaForCausalLM generate() interface")
              outputs = model.generate(
                  prompts=prompt,
                  device=device,
                  output_len=_OUTPUT_LEN.value,
                  temperature=_TEMPERATURE.value,
                  top_p=_TOP_P.value,
                  top_k=_TOP_K.value
              )
      else:
          raise ValueError("Model does not have a generate method")
      
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
# Convert standard weights to TPA using Tucker factorization and run inference:
# python scripts/run_tpa.py \
#   --ckpt=/path/to/gemma_model.ckpt \
#   --variant=1b \
#   --prompt="Write a poem about mathematics" \
#   --convert \
#   --save_tpa=/path/to/save/tpa_model.pt \
#   --q_rank=6 \
#   --k_rank=2 \
#   --v_rank=2 \
#   --device=cuda
#
# Run inference with already converted TPA model:
# python scripts/run_tpa.py \
#   --ckpt=/path/to/tpa_model.pt \
#   --variant=1b \
#   --prompt="Explain quantum mechanics" \
#   --convert=False \
#   --device=cuda
#
# Use direct TensorLy Tucker decomposition:
# python scripts/run_tpa.py \
#   --ckpt=/path/to/gemma_model.ckpt \
#   --variant=1b \
#   --prompt="What is artificial intelligence?" \
#   --convert \
#   --save_tpa=/path/to/save/tpa_model.pt \
#   --extra_config='{"factorization_method": "direct_tensorly"}' \
#   --device=cuda
#
# Use GQA to TPA conversion via Tucker decomposition:
# python scripts/run_tpa.py \
#   --ckpt=/path/to/gemma_model.ckpt \
#   --variant=1b \
#   --prompt="Describe the solar system" \
#   --convert \
#   --save_tpa=/path/to/save/tpa_model.pt \
#   --extra_config='{"factorization_method": "gqa_to_tpa"}' \
#   --device=cuda
#
# Use original T6-style contextual factorization:
# python scripts/run_tpa.py \
#   --ckpt=/path/to/gemma_model.ckpt \
#   --variant=1b \
#   --prompt="How do computers work?" \
#   --convert \
#   --save_tpa=/path/to/save/tpa_model.pt \
#   --extra_config='{"factorization_method": "contextual"}' \
#   --device=cuda