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

"""Runs inference with a Gemma model, potentially with DK-SVD compression."""

import contextlib
import random
import sys
import os

import numpy as np
import torch
from absl import app, flags
import gemma.model_dksv as gemma_model

# Ensure the gemma module can be found if script is run from a different directory
# Assuming standard project structure where 'scripts' is a sibling of 'gemma'
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


from gemma import config as gemma_config
# This script will use the DK-SVD enabled model definition


# Define flags
FLAGS = flags.FLAGS

flags.DEFINE_string('ckpt', None, 'Path to the checkpoint file (can be original or DK-SVD converted).', required=True)
flags.DEFINE_string('variant', '1b', 'Base model variant (e.g., 1b for Gemma 1.1-1b).')
flags.DEFINE_string('device', 'cpu', 'Device to run the model on (cpu or cuda).')
flags.DEFINE_integer('output_len', 100, 'Length of the output sequence.')
flags.DEFINE_integer('seed', 12345, 'Random seed.')
flags.DEFINE_boolean('quant', False,
                     'Whether the loaded checkpoint expects quantization for some layers. '
                     'If using DK-SVD, this flag pertains to non-DK-SVD modified parts or if DK-SVD parts were re-quantized.')
flags.DEFINE_string('prompt', 'The best thing about DK-SVD is', 'Input prompt for the model.')
flags.DEFINE_float('temperature', 1.0, 'Temperature for sampling. Set to 0 for greedy decoding.')
flags.DEFINE_float('top_p', 0.95, 'Top-p for nucleus sampling.')
flags.DEFINE_integer('top_k', 64, 'Top-k for sampling.')


# DK-SVD specific flags
flags.DEFINE_boolean('use_dksvd', False, 'Whether to load the model in DK-SVD mode. '
                                         'If True, --dksvd_rank must be specified. '
                                         'The checkpoint should be a DK-SVD converted one.')
flags.DEFINE_integer('dksvd_rank', None, 'Target rank r_g for DK-SVD QK compression. Required if --use_dksvd is True.')


# Define valid text only model variants
# Attempt to get from gemma_config, add '1b' specifically for the user's case if not already present.
_VALID_MODEL_VARIANTS = list(getattr(gemma_config, 'MODEL_CONFIGS', {}).keys())
if '1b' not in _VALID_MODEL_VARIANTS:
    _VALID_MODEL_VARPTS_CUSTOM = [
        config_name for config_name in dir(gemma_config) if config_name.startswith('get_config_for_')
    ]
    if any('1b' in name for name in _VALID_MODEL_VARPTS_CUSTOM):
        _VALID_MODEL_VARIANTS.append('1b')
    elif not _VALID_MODEL_VARIANTS: # If MODEL_CONFIGS was empty and no custom found
        _VALID_MODEL_VARIANTS = ['1b', '2b', '7b', '9b', '27b'] # Fallback list


# Define valid devices
_VALID_DEVICES = ['cpu', 'cuda']

# Validator function for the 'variant' flag
def validate_variant(variant):
    if variant not in _VALID_MODEL_VARIANTS:
        # Check if a specific config function exists for the variant (e.g., get_config_for_1b)
        if not hasattr(gemma_config, f'get_config_for_{variant}'):
            raise flags.FlagsError(f'Invalid variant: {variant}. Valid variants are: {_VALID_MODEL_VARIANTS} or must have a corresponding get_config_for_{variant} function in gemma.config.')
    return True

# Validator function for the 'device' flag
def validate_device(device):
    if device not in _VALID_DEVICES:
        raise flags.FlagsError(f'Invalid device: {device}. Valid devices are: {_VALID_DEVICES}')
    return True

flags.register_validator('variant', validate_variant, message='Invalid model variant.')
flags.register_validator('device', validate_device, message='Invalid device.')

@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float) # Reset to default float

def main(_argv):
    if FLAGS.use_dksvd and FLAGS.dksvd_rank is None:
        raise ValueError("--dksvd_rank must be specified when --use_dksvd is True.")
    if FLAGS.use_dksvd and FLAGS.dksvd_rank <= 0:
        raise ValueError("--dksvd_rank must be a positive integer.")

    # Construct the model config.
    # This assumes gemma_config.get_model_config can handle FLAGS.variant
    # (e.g. by looking up '1b' for get_config_for_1b)
    model_config = gemma_config.get_model_config(FLAGS.variant)

    # Set quantization status from flags. This will be used by the model's Linear/Embedding layers.
    model_config.quant = FLAGS.quant

    # Apply DK-SVD specific configurations to the config object.
    # The model_dksvd.py's GemmaForCausalLM and GemmaAttention will use these.
    if FLAGS.use_dksvd:
        setattr(model_config, 'use_dksvd', True)
        setattr(model_config, 'dksvd_rank', FLAGS.dksvd_rank)
        print(f"Running in DK-SVD mode with rank: {FLAGS.dksvd_rank}")
    else:
        setattr(model_config, 'use_dksvd', False)
        setattr(model_config, 'dksvd_rank', None)
        print("Running in standard GQA mode (or as per original checkpoint).")

    # Seed random number generators for reproducibility.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    # Determine the device for computation.
    if FLAGS.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device(FLAGS.device)

    if FLAGS.device == 'cuda':
        torch.cuda.manual_seed_all(FLAGS.seed)


    # Determine the torch dtype for model operations from the model_config.
    # model_config.get_dtype() should return the torch.dtype (e.g., torch.bfloat16).
    torch_dtype_for_model = model_config.get_dtype()
    print(f"Using PyTorch dtype: {torch_dtype_for_model} for model operations.")


    with _set_default_tensor_type(torch_dtype_for_model):
        # Instantiate the DK-SVD capable model
        # model_dksvd.GemmaForCausalLM should be designed to read use_dksvd and dksvd_rank from model_config
        print("Initializing model...")
        model = gemma_model.GemmaForCausalLM(model_config)

        print(f"Loading checkpoint from: {FLAGS.ckpt}")
        # The load_weights method in model_dksvd.py should be able to handle
        # DK-SVD specific weight names if use_dksvd is True in config.
        model.load_weights(FLAGS.ckpt)

        model = model.to(device).eval()
    print("Model loading and setup complete.")

    # Generate the response.
    print(f"\nGenerating response for prompt: '{FLAGS.prompt}'")
    print(f"Output length: {FLAGS.output_len}, Temperature: {FLAGS.temperature}, Top-p: {FLAGS.top_p}, Top-k: {FLAGS.top_k}\n")

    # Handle temperature for greedy decoding
    current_temperature = FLAGS.temperature if FLAGS.temperature > 0 else None

    result = model.generate(
        prompts=FLAGS.prompt,
        device=device,
        output_len=FLAGS.output_len,
        temperature=current_temperature,
        top_p=FLAGS.top_p,
        top_k=FLAGS.top_k,
    )

    # Print the prompts and results.
    print('======================================')
    print(f'PROMPT: {FLAGS.prompt}')
    print(f'RESULT: {result}')
    print('======================================')

if __name__ == "__main__":
    # It's good practice to parse flags explicitly for absl.
    # FLAGS(sys.argv) is not needed if app.run is the entry point.
    app.run(main)