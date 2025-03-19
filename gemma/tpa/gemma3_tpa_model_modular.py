"""
Modular Gemma3 model implementation with Tensor Product Attention (TPA).

This file contains the main model class (Gemma3ForMultimodalLMwithTPA) which
integrates components from the modules directory for TPA functionality.
"""

import math
import time
import torch
import os
import json
import gc
from torch import nn
from PIL import Image
from typing import Any, List, Sequence, Tuple, Union, Optional, Mapping

from .. import model as gemma_model
from .. import config as gemma_config
from .. import gemma3_preprocessor
from .. import tokenizer
from ..siglip_vision import siglip_vision_model
from .tpa_model import GemmaTPAModel, create_tpa_kv_caches
import torch.nn.functional as F

# Import modular components
from .modules.tucker_factorization import (
    factorize_all_layers_with_shared_factors,
    _factorize_mha_weights_with_shared_factors,
    adaptive_rank_selection,
    _factorize_and_set_weights
)
from .modules.contextual_factorization import (
    contextual_tensor_decomposition,
    apply_contextual_tensor_decomposition,
    convert_from_standard_weights as cf_convert_from_standard_weights
)
from .modules.tensor_product_utils import (
    register_freqs_cis,
    create_attention_mask,
    populate_image_embeddings
)


class Gemma3ForMultimodalLMwithTPA(nn.Module):
    """Gemma3 model for multimodal causal LM with Tensor Product Attention."""
    
    def __init__(
        self,
        config: gemma_config.GemmaConfig,
    ):
        super().__init__()
        self.dtype = config.get_dtype()
        # Check for multimodal model by detecting vision_config
        # Remove architecture check to allow non-Gemma3 models (like 1B text-only)
        self.is_multimodal = (hasattr(config, 'vision_config') and 
                             config.vision_config is not None)
        
        self.config = config
        
        # Set proper defaults if not present
        if not hasattr(config, 'max_position_embeddings'):
            print("Setting default max_position_embeddings to 2048")
            config.max_position_embeddings = 2048
        
        if not hasattr(config, 'head_dim'):
            print(f"Setting default head_dim to {config.hidden_size // config.num_attention_heads}")
            config.head_dim = config.hidden_size // config.num_attention_heads
            
        # Add TPA ranks if not defined
        if not hasattr(config, 'q_rank'):
            print("Setting default q_rank to 6")
            config.q_rank = 6
        if not hasattr(config, 'k_rank'):
            print("Setting default k_rank to 2")
            config.k_rank = 2
        if not hasattr(config, 'v_rank'):
            print("Setting default v_rank to 2")
            config.v_rank = 2
        
        # For non-MQA/GQA models, ensure num_key_value_heads is set
        if not hasattr(config, 'num_key_value_heads'):
            print(f"Setting default num_key_value_heads to {config.num_attention_heads}")
            config.num_key_value_heads = config.num_attention_heads
            
        max_seq_len = config.max_position_embeddings
        head_dim = config.head_dim
        vocab_size = config.vocab_size

        # Initialize tokenizer if available
        if hasattr(config, 'tokenizer'):
            self.tokenizer = tokenizer.Tokenizer(config.tokenizer)
        else:
            self.tokenizer = None
            print("No tokenizer configuration found in model config")
            
        # Initialize embedder - match naming with standard Gemma model
        self.embedder = gemma_model.Embedding(vocab_size, config.hidden_size, 
                                           getattr(config, 'quant', False))
        # Also create an alias for compatibility with any code using Gemma3 naming convention
        self.text_token_embedder = self.embedder
        
        # Initialize TPA model
        self.model = GemmaTPAModel(config)
        
        # Initialize output sampler
        self.sampler = gemma_model.Sampler(vocab_size, config)

        # Handle multimodal components only if the model is multimodal
        if self.is_multimodal:
            print("Initializing multimodal components")
            self.siglip_vision_model = siglip_vision_model.SiglipVisionModel(config.vision_config)
            # transformer/embedder/mm_soft_embedding_norm
            self.mm_soft_embedding_norm = gemma_model.RMSNorm(config.vision_config.embedding_dim,
                                                           eps = config.rms_norm_eps)
            # transformer/embedder/mm_input_projection
            self.mm_input_projection = gemma_model.Linear(config.vision_config.embedding_dim, 
                                                      config.hidden_size, 
                                                      getattr(config, 'quant', False))
        else:
            print("Initializing text-only model without vision components")

        # Set up RoPE frequencies, with different handling for different model types
        # Get scaling factor (if available)
        rope_scaling_factor = getattr(config, 'rope_scaling_factor', 1)
            
        # Handle RoPE configuration for different model types
        try:
            if hasattr(config, 'rope_wave_length') and config.rope_wave_length is not None:
                # Gemma3 style
                print("Using Gemma3-style RoPE configuration")
                
                # Make sure AttentionType constants are available
                if hasattr(gemma_config, 'AttentionType'):
                    defaults = {
                        gemma_config.AttentionType.LOCAL_SLIDING: 10_000,
                        gemma_config.AttentionType.GLOBAL: 10_000,
                    }
                    
                    rope_lengths = config.rope_wave_length
                    
                    # Generate local frequencies
                    local_theta = rope_lengths.get(
                        gemma_config.AttentionType.LOCAL_SLIDING, 
                        defaults[gemma_config.AttentionType.LOCAL_SLIDING]
                    )
                    self._register_freqs_cis('local_freqs_cis', head_dim, max_seq_len, 
                                           theta=local_theta, rope_scaling_factor=1)
                    
                    # Generate global frequencies
                    global_theta = rope_lengths.get(
                        gemma_config.AttentionType.GLOBAL, 
                        defaults[gemma_config.AttentionType.GLOBAL]
                    )
                    self._register_freqs_cis('global_freqs_cis', head_dim, max_seq_len, 
                                          theta=global_theta, rope_scaling_factor=rope_scaling_factor)
                else:
                    # Fallback if AttentionType not available
                    print("Warning: gemma_config.AttentionType not available, using standard RoPE")
                    theta = 10_000  # Default
                    self._register_freqs_cis('freqs_cis', head_dim, max_seq_len, 
                                          theta=theta, rope_scaling_factor=rope_scaling_factor)
            else:
                # Standard Gemma style
                print("Using standard RoPE configuration")
                theta = getattr(config, 'rope_theta', 10_000)
                self._register_freqs_cis('freqs_cis', head_dim, max_seq_len, 
                                      theta=theta, rope_scaling_factor=rope_scaling_factor)
        except Exception as e:
            # Catch any errors and use a simple fallback
            print(f"Error setting up RoPE frequencies: {e}")
            print("Using fallback RoPE configuration")
            self._register_freqs_cis('freqs_cis', head_dim, max_seq_len, 
                                  theta=10_000, rope_scaling_factor=1)

    def _register_freqs_cis(self, name: str, head_dim: int, max_seq_len: int, theta: int = 10_000, rope_scaling_factor: int = 1):
        """Register rotary position embedding frequencies."""
        # Delegate to module function
        register_freqs_cis(self, name, head_dim, max_seq_len, theta, rope_scaling_factor)

    @torch.no_grad()
    def forward(
        self,
        input_token_ids: torch.Tensor,
        image_patches: Optional[torch.Tensor] = None,
        image_presence_mask: Optional[torch.Tensor] = None,
        input_positions: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
        mask: Optional[torch.Tensor] = None,
        output_positions: Optional[torch.Tensor] = None,
        temperatures: Optional[torch.Tensor] = None,
        top_ps: Optional[torch.Tensor] = None,
        top_ks: Optional[torch.Tensor] = None,
        local_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Gemma3 TPA model.
        
        Handles both text-only and multimodal inputs, generating next tokens
        and logits for the specified positions.
        
        Args:
            input_token_ids: Input token ids [batch_size, seq_len]
            image_patches: Image patches [batch_size, num_images, C, H, W]
            image_presence_mask: Boolean mask indicating image presence [batch_size, num_images]
            input_positions: Position indices for input tokens
            kv_caches: KV caches for each layer
            mask: Attention mask
            output_positions: Positions to generate tokens for
            temperatures: Temperature for sampling
            top_ps: Top-p values for sampling
            top_ks: Top-k values for sampling
            local_mask: Local window attention mask
            
        Returns:
            tuple: (next_tokens, logits)
        """
        batch_size, seq_len = input_token_ids.shape
        device = input_token_ids.device
        
        # Default positions if not provided
        if input_positions is None:
            input_positions = torch.arange(seq_len, device=device)
        
        # Create rotary position embedding mapping
        freqs_cis = {}
        if hasattr(self, 'local_freqs_cis') and hasattr(self, 'global_freqs_cis'):
            # Gemma3 style RoPE
            freqs_cis[gemma_config.AttentionType.LOCAL_SLIDING] = (
                self.local_freqs_cis.index_select(0, input_positions)
            )
            freqs_cis[gemma_config.AttentionType.GLOBAL] = (
                self.global_freqs_cis.index_select(0, input_positions)
            )
        elif hasattr(self, 'freqs_cis'):
            # Standard style RoPE
            freqs_cis = self.freqs_cis.index_select(0, input_positions)
        
        # Get text embeddings
        hidden_states = self.embedder(input_token_ids)
        
        # Handle multimodal inputs
        if self.is_multimodal and image_patches is not None and image_presence_mask is not None:
            # Process images with vision model
            image_embeddings = self._process_images(image_patches, image_presence_mask)
            
            # Find image token positions - assuming image tokens are marked with a special token ID
            image_token_ids = getattr(self.config, 'image_token_id', 
                                     self.tokenizer.piece_to_id("<image>") if self.tokenizer else 32000)
            
            # Create image token indices tensor
            image_indices = []
            for b in range(batch_size):
                # Find all positions where the image token appears
                indices = torch.where(input_token_ids[b] == image_token_ids)[0]
                # Ensure we don't have more indices than images
                num_images = image_presence_mask[b].sum().item()
                indices = indices[:num_images]
                # Pad with -1 if needed
                if len(indices) < image_presence_mask.shape[1]:
                    padding = torch.full((image_presence_mask.shape[1] - len(indices),), 
                                        -1, device=device, dtype=indices.dtype)
                    indices = torch.cat([indices, padding])
                image_indices.append(indices)
            
            # Stack into a tensor [batch_size, max_num_images]
            image_token_indices = torch.stack(image_indices, dim=0)
            
            # Insert image embeddings at appropriate positions
            hidden_states = populate_image_embeddings(
                hidden_states, image_embeddings, image_token_indices)
        
        # Create attention mask if needed
        if mask is None:
            mask = create_attention_mask(
                input_token_ids,
                seq_len,
                getattr(self.config, 'sliding_window_size', None),
                is_causal=True,
                image_tokens_indices=image_token_indices if self.is_multimodal else None
            )
        
        # Create KV caches if needed
        if kv_caches is None:
            kv_caches = create_tpa_kv_caches(
                self.config, batch_size, self.config.max_position_embeddings, device)
        
        # Run model forward pass
        hidden_states = self.model(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=input_positions,
            kv_caches=kv_caches,
            mask=mask,
            local_mask=local_mask,
        )
        
        # Sample next tokens
        if output_positions is None:
            # Default to the last position
            output_positions = torch.tensor([seq_len - 1], device=device)
        
        # Get outputs only at the specified positions
        outputs = hidden_states.index_select(1, output_positions)
        
        # Get embedder weight for sampler
        embedder_weight = self.embedder.weight
        if getattr(self.config, 'quant', False):
            embedder_weight = embedder_weight * self.embedder.weight_scaler.unsqueeze(-1)
            
        # Get default sampling parameters if not provided
        temperatures = temperatures if temperatures is not None else torch.ones(batch_size, device=device)
        top_ps = top_ps if top_ps is not None else torch.ones(batch_size, device=device)
        top_ks = top_ks if top_ks is not None else torch.full((batch_size,), 50, device=device)
        
        # Call sampler with all required parameters
        next_tokens, logits = self.sampler(
            embedding=embedder_weight,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks
        )
        
        return next_tokens, logits
        
    def _process_images(self, image_patches: torch.Tensor, image_presence_mask: torch.Tensor) -> torch.Tensor:
        """Process image patches through vision model and prepare for transformer."""
        if not self.is_multimodal:
            return None
            
        batch_size, num_images = image_presence_mask.shape
        
        # Process each image through the vision model
        embeddings = []
        for b in range(batch_size):
            batch_embeddings = []
            for i in range(num_images):
                if image_presence_mask[b, i]:
                    # Process image through vision model
                    with torch.no_grad():
                        # Get single image
                        img = image_patches[b, i]
                        # Process through vision model
                        img_embedding = self.siglip_vision_model(img.unsqueeze(0))
                        # Apply norm and projection
                        img_embedding = self.mm_soft_embedding_norm(img_embedding)
                        img_embedding = self.mm_input_projection(img_embedding)
                        batch_embeddings.append(img_embedding.squeeze(0))
                else:
                    # Create zero embedding for missing images
                    batch_embeddings.append(torch.zeros(
                        self.config.hidden_size, device=image_patches.device))
            
            # Stack this batch's embeddings
            embeddings.append(torch.stack(batch_embeddings, dim=0))
        
        # Stack all batches
        return torch.stack(embeddings, dim=0)
        
    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 128,
        max_seq_len: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 50,
        image_patches: Optional[torch.Tensor] = None,
        image_presence_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate text from input prompts.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum number of tokens to generate
            max_seq_len: Maximum sequence length (default: model's max_position_embeddings)
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            image_patches: Image patches for multimodal generation
            image_presence_mask: Boolean mask indicating image presence
            
        Returns:
            List of generated text strings
        """
        # Ensure we have a tokenizer
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for generation")
        
        # Default max sequence length
        if max_seq_len is None:
            max_seq_len = self.config.max_position_embeddings
        
        # Set device based on model parameters
        device = next(self.parameters()).device
        
        # Tokenize prompts
        batch_size = len(prompts)
        tokenized = []
        for prompt in prompts:
            # Make sure prompt is a string
            if not isinstance(prompt, str):
                prompt = str(prompt)
            tokens = self.tokenizer.encode(prompt)
            tokenized.append(tokens)
        
        # Find max prompt length
        max_prompt_len = max(len(t) for t in tokenized)
        
        # Create batch tensor and pad
        input_ids = []
        for tokens in tokenized:
            # Pad with EOS token
            padded = tokens + [self.tokenizer.eos_id] * (max_prompt_len - len(tokens))
            input_ids.append(padded)
        
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        
        # Handle multimodal inputs
        has_images = self.is_multimodal and image_patches is not None and image_presence_mask is not None
        
        # Create KV caches
        kv_caches = create_tpa_kv_caches(
            self.config, batch_size, max_seq_len, device)
        
        # Create tensors for sampling parameters
        temperatures = torch.full((batch_size,), temperature, device=device)
        top_ps = torch.full((batch_size,), top_p, device=device)
        top_ks = torch.full((batch_size,), top_k, device=device)
        
        # Initial forward pass with prompt
        positions = torch.arange(max_prompt_len, device=device)
        
        # Create attention mask
        attn_mask = create_attention_mask(
            input_ids,
            max_seq_len,
            getattr(self.config, 'sliding_window_size', None),
            is_causal=True
        )
        
        # Process prompt tokens
        _, _ = self.forward(
            input_token_ids=input_ids,
            image_patches=image_patches if has_images else None,
            image_presence_mask=image_presence_mask if has_images else None,
            input_positions=positions,
            kv_caches=kv_caches,
            mask=attn_mask,
            output_positions=torch.tensor([max_prompt_len - 1], device=device),
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
        )
        
        # Autoregressive generation
        all_output_ids = input_ids.clone()
        cur_len = max_prompt_len
        
        # Check if we can use a single token position for efficiency
        use_single_pos = True
        
        for _ in range(max_tokens):
            # Position for the next token
            if use_single_pos:
                # More efficient when generating one token at a time
                position = torch.tensor([cur_len], device=device)
            else:
                # Full sequence positions
                position = torch.arange(cur_len + 1, device=device)
            
            # Get next token
            if cur_len < max_seq_len:
                # Simple case: append token to sequence
                next_token_ids, _ = self.forward(
                    input_token_ids=all_output_ids,
                    input_positions=position[-1:],  # Just the new position
                    kv_caches=kv_caches,
                    mask=attn_mask,
                    output_positions=torch.tensor([cur_len], device=device),
                    temperatures=temperatures,
                    top_ps=top_ps,
                    top_ks=top_ks,
                )
                
                # Add new tokens to the sequence
                all_output_ids = torch.cat([all_output_ids, next_token_ids.unsqueeze(1)], dim=1)
            else:
                # Handle case where sequence exceeds max length using sliding window
                print(f"Warning: Sequence length exceeds max_seq_len ({max_seq_len}), using sliding window")
                # TODO: Implement sliding window approach for very long sequences
                break
                
            cur_len += 1
            
            # Check for EOS tokens
            if (all_output_ids[:, -1] == self.tokenizer.eos_id).all():
                break
        
        # Decode generated tokens to text
        outputs = []
        for i in range(batch_size):
            # Remove initial prompt and convert to string
            generated = all_output_ids[i, max_prompt_len:].tolist()
            # Remove EOS token if present
            if self.tokenizer.eos_id in generated:
                generated = generated[:generated.index(self.tokenizer.eos_id)]
            # Decode to string
            text = self.tokenizer.decode(generated)
            outputs.append(text)
            
        return outputs
    
    def load_weights(self, weights_path: str):
        """Load model weights from a file or directory."""
        if os.path.isdir(weights_path):
            # Load from directory containing sharded weights
            print(f"Loading weights from directory: {weights_path}")
            # Check for a model index file
            model_index_path = os.path.join(weights_path, "model_index.json")
            if os.path.exists(model_index_path):
                with open(model_index_path, "r") as f:
                    model_index = json.load(f)
                # TODO: Handle model index for sharded weights
            else:
                # Simple weight loading from checkpoint
                ckpt_path = os.path.join(weights_path, "model.ckpt")
                if os.path.exists(ckpt_path):
                    self.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                    print(f"Loaded weights from {ckpt_path}")
                else:
                    raise ValueError(f"No model.ckpt found in {weights_path}")
        else:
            # Load from single file
            print(f"Loading weights from file: {weights_path}")
            self.load_state_dict(torch.load(weights_path, map_location="cpu"))
            
        # Clean up any references to save memory
        gc.collect()
        torch.cuda.empty_cache()
        
    # Delegate factorization methods to module functions
    def factorize_all_layers_with_shared_factors(self):
        """Factorize all layers using Tucker decomposition with shared factors."""
        return factorize_all_layers_with_shared_factors(self.model, self.config)
    
    def apply_contextual_tensor_decomposition(self):
        """Apply T6-style contextual tensor factorization to all layers."""
        return apply_contextual_tensor_decomposition(
            self.model, 
            q_rank=self.config.q_rank,
            k_rank=self.config.k_rank,
            v_rank=self.config.v_rank
        )
        
    def convert_from_standard_weights(self, standard_model):
        """
        Convert from standard Gemma model to TPA model.
        
        This method copies weights from a standard Gemma model to this TPA model,
        converting attention weights using contextual tensor factorization.
        
        Args:
            standard_model: Standard Gemma model to convert from
            
        Returns:
            self: The converted TPA model
        """
        return cf_convert_from_standard_weights(
            standard_model,
            self,
            q_rank=self.config.q_rank,
            k_rank=self.config.k_rank,
            v_rank=self.config.v_rank
        )