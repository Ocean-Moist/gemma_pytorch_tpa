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
from PIL import Image
import torch

from gemma import config
from gemma import gemma3_model
from gemma.tpa.gemma3_tpa_model import Gemma3ForMultimodalLMwithTPA

# Define flags
FLAGS = flags.FLAGS

_CKPT = flags.DEFINE_string(
    'ckpt', None, 'Path to the checkpoint file.', required=True
)
_VARIANT = flags.DEFINE_string('variant', '4b', 'Model variant.')
_DEVICE = flags.DEFINE_string('device', 'cuda', 'Device to run the model on.')
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
_IMAGE = flags.DEFINE_string('image', None, 'Path to image file for multimodal prompting.')
_PROMPT = flags.DEFINE_string('prompt', 'What are large language models?', 
                             'Input prompt for the model.')
_TEMPERATURE = flags.DEFINE_float('temperature', 0.9, 'Temperature for sampling.')
_TOP_P = flags.DEFINE_float('top_p', 0.95, 'Top-p sampling parameter.')
_TOP_K = flags.DEFINE_integer('top_k', 64, 'Top-k sampling parameter.')

# Define valid multimodal model variants
_VALID_MODEL_VARIANTS = ['4b', '12b', '27b_v3', '1b']

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


def main(_):
  print(f"Running Gemma-{_VARIANT.value} with TPA")
  print(f"TPA configuration: q_rank={_Q_RANK.value}, k_rank={_K_RANK.value}, v_rank={_V_RANK.value}")
  
  # Load image if specified
  image_data = None
  if _IMAGE.value:
    try:
      print(f"Loading image from {_IMAGE.value}")
      image_data = Image.open(_IMAGE.value)
      print(f"Image loaded successfully: {image_data.size}")
    except Exception as e:
      print(f"Error loading image: {e}")
      return
  
  # Construct the model config
  model_config = config.get_model_config(_VARIANT.value)
  model_config.dtype = 'float16' if torch.cuda.is_available() and _DEVICE.value == 'cuda' else 'float32'
  model_config.quant = _QUANT.value
  
  # Add TPA specific configuration parameters
  model_config.q_rank = _Q_RANK.value
  model_config.k_rank = _K_RANK.value
  model_config.v_rank = _V_RANK.value
  
  # Seed random
  random.seed(_SEED.value)
  np.random.seed(_SEED.value)
  torch.manual_seed(_SEED.value)
  
  # Create the device
  device = torch.device(_DEVICE.value)
  
  # Set up timing measurements
  start_time = time()
  
  # Create and load the model
  with _set_default_tensor_type(model_config.get_dtype()):
    if _CONVERT.value:
      print(f"Loading standard Gemma model from {_CKPT.value}...")
      standard_model = gemma3_model.Gemma3ForMultimodalLM(model_config)
      standard_model.load_weights(_CKPT.value)
      standard_model.eval()
      
      load_time = time()
      print(f"Standard model loaded in {load_time - start_time:.2f} seconds")
      
      print("Converting to TPA model...")
      model = Gemma3ForMultimodalLMwithTPA(model_config)
      model.convert_from_standard_weights(standard_model)
      
      convert_time = time()
      print(f"Model converted to TPA in {convert_time - load_time:.2f} seconds")
      
      if _SAVE_TPA.value:
        print(f"Saving TPA model to {_SAVE_TPA.value}...")
        save_dir = os.path.dirname(_SAVE_TPA.value)
        if save_dir:
          os.makedirs(save_dir, exist_ok=True)
        torch.save({'model_state_dict': model.state_dict()}, _SAVE_TPA.value)
        print(f"TPA model saved successfully")
      
      # Clear standard model from memory
      del standard_model
      if torch.cuda.is_available():
        torch.cuda.empty_cache()
      
    else:
      print(f"Loading TPA model from {_CKPT.value}...")
      model = Gemma3ForMultimodalLMwithTPA(model_config)
      model.load_weights(_CKPT.value)
      load_time = time()
      print(f"TPA model loaded in {load_time - start_time:.2f} seconds")
    
    # Move model to device
    model = model.to(device).eval()
    to_device_time = time()
    print(f"Model moved to {device} in {to_device_time - load_time:.2f} seconds")
  
  # Prepare prompt
  if image_data:
    prompt = [(_PROMPT.value, image_data)]
  else:
    prompt = [(_PROMPT.value,)]
  
  # Generate response
  print(f"Generating response with temperature={_TEMPERATURE.value}, top_p={_TOP_P.value}, top_k={_TOP_K.value}...")
  generate_start = time()
  
  outputs = model.generate(
    prompts=prompt,
    device=device,
    output_len=_OUTPUT_LEN.value,
    temperature=_TEMPERATURE.value,
    top_p=_TOP_P.value,
    top_k=_TOP_K.value
  )
  
  generate_end = time()
  
  # Print the generated text
  print("\n" + "="*50)
  print(f"PROMPT: {_PROMPT.value}")
  if image_data:
    print(f"[Image: {_IMAGE.value}]")
  print(f"RESULT: {outputs[0]}")
  print("="*50)
  
  # Print performance metrics
  generation_time = generate_end - generate_start
  tokens_generated = len(outputs[0].split())
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


if __name__ == '__main__':
  app.run(main)

# Example commands:
# 
# Convert standard weights to TPA and run inference:
# python scripts/run_tpa.py \
#   --ckpt=/path/to/gemma3_weights \
#   --variant=4b \
#   --prompt="Write a poem about mathematics" \
#   --convert \
#   --save_tpa=/path/to/save/tpa_model.pt
#
# Run inference with already converted TPA model:
# python scripts/run_tpa.py \
#   --ckpt=/path/to/tpa_model.pt \
#   --variant=4b \
#   --prompt="Explain quantum mechanics" \
#   --convert=False \
#   --image=/path/to/image.jpg