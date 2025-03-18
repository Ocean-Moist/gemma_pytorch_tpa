"""TPA-based Gemma3 multimodal model implementation."""

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
            
        # Initialize embedder - using 'text_token_embedder' for Gemma3 naming convention
        self.text_token_embedder = gemma_model.Embedding(vocab_size, config.hidden_size, 
                                                     getattr(config, 'quant', False))
        
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
        defaults = {
                gemma_config.AttentionType.LOCAL_SLIDING: 10_000,
                gemma_config.AttentionType.GLOBAL: 10_000,
            }
            
        # Handle RoPE configuration for different model types
        if hasattr(config, 'rope_wave_length') and config.rope_wave_length is not None:
            # Gemma3 style
            print("Using Gemma3-style RoPE configuration")
            rope_lengths = config.rope_wave_length
            rope_scaling_factor = getattr(config, 'rope_scaling_factor', 1)
            
            self._register_freqs_cis('local_freqs_cis', head_dim, max_seq_len, theta=rope_lengths.get(
                    gemma_config.AttentionType.LOCAL_SLIDING, defaults[gemma_config.AttentionType.LOCAL_SLIDING]
                ))
            self._register_freqs_cis('global_freqs_cis', head_dim, max_seq_len, theta=rope_lengths.get(
                    gemma_config.AttentionType.GLOBAL, defaults[gemma_config.AttentionType.GLOBAL]
                ), rope_scaling_factor=rope_scaling_factor)
        else:
            # Standard Gemma style
            print("Using standard RoPE configuration")
            theta = getattr(config, 'rope_theta', 10_000)
            rope_scaling_factor = getattr(config, 'rope_scaling_factor', 1)
            
            self._register_freqs_cis('local_freqs_cis', head_dim, max_seq_len, theta=theta)
            self._register_freqs_cis('global_freqs_cis', head_dim, max_seq_len, theta=theta, 
                                  rope_scaling_factor=rope_scaling_factor)
                                  
        print(f"Initialized Gemma model with TPA: q_rank={config.q_rank}, k_rank={config.k_rank}, v_rank={config.v_rank}")

    def _register_freqs_cis(
        self, name: str, head_dim: int, max_seq_len: int, theta: int = 10_000, rope_scaling_factor: int = 1
    ):
        # Ensure rope_scaling_factor is not None
        if rope_scaling_factor is None:
            rope_scaling_factor = 1
            
        self.register_buffer(
                name, gemma_model.precompute_freqs_cis(head_dim, max_seq_len * 2, theta=theta, rope_scaling_factor=rope_scaling_factor)
            )

    @torch.no_grad()
    def forward(self,
            input_token_ids: torch.Tensor, # B x L
            image_patches: Optional[torch.Tensor] = None, # B x N x C x H x W (3x896x896)
            image_presence_mask: Optional[torch.Tensor] = None, # B x N
            input_positions: Optional[torch.Tensor] = None,
            kv_caches: Optional[List[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]]] = None,
            mask: Optional[torch.Tensor] = None,
            output_positions: Optional[torch.Tensor] = None,
            temperatures: Optional[Union[torch.Tensor, None]] = None,
            top_ps: Optional[torch.Tensor] = None,
            top_ks: Optional[torch.Tensor] = None,
            local_mask: Optional[torch.Tensor] = None,
            **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with flexibility to handle both multimodal and text-only models."""
        # For standard text-only model inference (used with non-multimodal models)
        if input_positions is None:
            # Simple text-only inference case
            logits = self.text_token_embedder(input_token_ids)
            normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=logits.dtype, device=logits.device)
            logits = logits * normalizer
            
            # Forward through TPA model - need to create positional indices
            seq_len = input_token_ids.size(1)
            batch_size = input_token_ids.size(0)
            pos_indices = torch.arange(0, seq_len, device=input_token_ids.device)
            
            # Create freqs_cis dictionary
            freqs_cis = {}
            freqs_cis[gemma_config.AttentionType.LOCAL_SLIDING] = self.local_freqs_cis.index_select(0, pos_indices)
            freqs_cis[gemma_config.AttentionType.GLOBAL] = self.global_freqs_cis.index_select(0, pos_indices)
            
            # Create attention mask for causal decoding
            max_len = seq_len
            attn_mask = torch.tril(torch.ones((batch_size, 1, max_len, max_len), dtype=torch.bool, device=input_token_ids.device))
            min_dtype = torch.finfo(self.dtype).min
            attn_mask_tensor = torch.where(attn_mask, 0, torch.tensor(min_dtype, dtype=torch.float32, device=input_token_ids.device))
            
            # Create sliding window local mask if needed 
            if self.config.sliding_window_size is not None:
                # Create proper sliding window mask
                # Ensure the sliding window is created with the correct size and properly aligned
                window_size = self.config.sliding_window_size
                
                # Create the base causal mask
                local_mask_bool = torch.ones((batch_size, 1, max_len, max_len), dtype=torch.bool, device=input_token_ids.device)
                local_mask_bool = torch.tril(local_mask_bool)  # Lower triangular for causal attention
                
                # Create the sliding window upper bound mask
                # Allow attention to positions within window_size steps before current position
                if window_size > 0:
                    window_mask = torch.triu(torch.ones((1, 1, max_len, max_len), dtype=torch.bool, device=input_token_ids.device), 
                                          diagonal=-(window_size-1))
                    # Apply both masks - this limits attention to only the local window in the causal region
                    local_mask_bool = torch.logical_and(local_mask_bool, window_mask)
                
                # Convert bool mask to attention values (0 for attend, min_value for don't attend)
                local_mask_tensor = torch.where(
                    local_mask_bool, 0, torch.tensor(min_dtype, dtype=torch.float32, device=input_token_ids.device)
                )
                
                print(f"Created sliding window mask with shape {local_mask_tensor.shape}, window size: {window_size}")
            else:
                local_mask_tensor = attn_mask_tensor
                print(f"Using standard causal mask with shape {local_mask_tensor.shape}")
            
            # Create TPA KV caches if not provided
            if kv_caches is None:
                kv_caches = create_tpa_kv_caches(self.config, batch_size, seq_len, input_token_ids.device)
            
            # Simple write indices for full sequence
            kv_write_indices = torch.arange(0, seq_len, device=input_token_ids.device)
            
            # Forward through model
            hidden_states = self.model(
                hidden_states=logits,
                freqs_cis=freqs_cis,
                kv_write_indices=kv_write_indices,
                kv_caches=kv_caches,
                mask=attn_mask_tensor,
                local_mask=local_mask_tensor,
            )
            
            # Project to vocabulary
            embedder_weight = self.text_token_embedder.weight
            if self.config.quant:
                embedder_weight = embedder_weight * self.text_token_embedder.weight_scaler.unsqueeze(-1)
                
            # Compute logits
            return torch.matmul(hidden_states, embedder_weight.transpose(0, 1))
            
        # Regular multimodal forward pass
        freqs_cis = {}
        freqs_cis[gemma_config.AttentionType.LOCAL_SLIDING] = (
                self.local_freqs_cis.index_select(0, input_positions)
            )
        freqs_cis[gemma_config.AttentionType.GLOBAL] = (
                self.global_freqs_cis.index_select(0, input_positions)
            )
        hidden_states = self.text_token_embedder(input_token_ids)
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = hidden_states * normalizer
        
        # Handle image input for multimodal models
        if self.is_multimodal and image_patches is not None:
            # the input has images
            B, N, C, H, W = image_patches.shape
            # Flatten and Pass to SiglipVisionModel, and apply SiglipVisionModel Exit
            flattened_input = image_patches.reshape(B * N, C, H, W)  # (B*N)xCxHxW
            image_embeddings = self.siglip_vision_model(flattened_input)  # (B*N)xUxD
            image_embeddings = self.mm_soft_embedding_norm(image_embeddings)  # (B*N) x U x D
            image_embeddings = self.mm_input_projection(image_embeddings)  # (B*N) x U x model_dim
            hidden_states = self.populate_image_embeddings(
                hidden_states.clone(),
                image_embeddings.clone(),
                input_token_ids.clone(),
                image_presence_mask.clone(),
            )

        kv_write_indices = input_positions

        hidden_states = self.model(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis,
                kv_write_indices=kv_write_indices,
                kv_caches=kv_caches,
                mask=mask,
                local_mask=local_mask,
            )
        embedder_weight = self.text_token_embedder.weight
        if self.config.quant:
            embedder_weight = (
                    embedder_weight * self.text_token_embedder.weight_scaler.unsqueeze(-1))

        next_tokens, logits = self.sampler(
                embedding=embedder_weight,
                hidden_states=hidden_states,
                output_positions=output_positions,
                temperatures=temperatures,
                top_ps=top_ps,
                top_ks=top_ks,
            )
        return next_tokens, logits

    def populate_image_embeddings(self,
                            hidden_states: torch.Tensor, # B x L x model_dim
                            image_embeddings: torch.Tensor, # (B*N) x U x model_dim
                            input_token_ids: torch.Tensor, # B x L
                            image_presence_mask: torch.Tensor, # B x N
                            ):
        batch_size, seq_len, model_dim = hidden_states.shape
        
        # Return early if tokenizer is not set or doesn't support images
        if self.tokenizer is None or not hasattr(self.tokenizer, 'image_token_placeholder_id'):
            return hidden_states
            
        # Step 1 of 2: Fetch valid image embeddings
        # flatten indices of valid image embeddings
        valid_image_embeddings_indices = torch.nonzero(image_presence_mask.flatten(), as_tuple=False).squeeze()
        
        # Handle case where there are no valid image embeddings
        if valid_image_embeddings_indices.numel() == 0:
            return hidden_states
            
        # num_valid_images x model_dim
        valid_image_embeddings = image_embeddings.index_select(0, valid_image_embeddings_indices)

        # Step 2 of 2: Replace image embeddings at right places.
        image_placeholder_mask = input_token_ids == self.tokenizer.image_token_placeholder_id
        image_placeholder_indices = image_placeholder_mask.flatten().nonzero(as_tuple=False).squeeze()
        
        # Return early if there are no image placeholders
        if image_placeholder_indices.numel() == 0:
            return hidden_states

        hidden_states = hidden_states.reshape(-1, self.config.hidden_size)
        hidden_states[image_placeholder_indices] = valid_image_embeddings.reshape(-1, self.config.hidden_size)
        return hidden_states.reshape(batch_size, seq_len, model_dim).contiguous()

    def create_attention_mask(self, input_ids: torch.Tensor, sequence_length: int):
        """
        Create attention masks for both standard causal attention and local sliding window attention.
        
        Args:
            input_ids: Input token IDs tensor [batch_size, seq_len]
            sequence_length: Total sequence length
            
        Returns:
            tuple: (attention_mask, local_mask)
                - attention_mask: Global attention mask tensor
                - local_mask: Local sliding window attention mask tensor
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Create standard causal mask (lower triangular)
        causal_mask = torch.tril(
            torch.ones((batch_size, 1, sequence_length, sequence_length), 
                      dtype=torch.bool, device=device)
        )
        
        # For non-multimodal models or when tokenizer doesn't have image token placeholder
        if not self.is_multimodal or self.tokenizer is None or not hasattr(self.tokenizer, 'image_token_placeholder_id'):
            # Create sliding window local mask if needed
            if hasattr(self.config, 'sliding_window_size') and self.config.sliding_window_size is not None:
                sliding_window_size = self.config.sliding_window_size
                print(f"Creating sliding window mask with window size: {sliding_window_size}")
                
                # Create proper sliding window mask with correct dimensions
                # Ensure the causal mask has correct batch shape
                local_mask_bool = causal_mask.clone()
                
                # Add the window size constraint if it's positive
                if sliding_window_size > 0:
                    # Create upper triangular mask with -window_size offset diagonal
                    # This mask allows attention only within sliding_window_size tokens before current position
                    window_mask = torch.triu(
                        torch.ones((batch_size, 1, sequence_length, sequence_length), 
                                  dtype=torch.bool, device=device),
                        diagonal=-(sliding_window_size-1)
                    )
                    
                    # Combine causal mask with sliding window mask
                    local_mask = torch.logical_and(local_mask_bool, window_mask)
                else:
                    local_mask = local_mask_bool
                    
                print(f"Created sliding window local mask with shape {local_mask.shape}")
            else:
                # Without sliding window, local mask is just the causal mask
                local_mask = causal_mask
                print(f"Using standard causal mask with shape {causal_mask.shape} for local attention")
                
            return causal_mask, local_mask
        
        # For multimodal models with image tokens
        print("Creating multimodal attention mask with image token handling")
        
        try:
            # Get image token mask
            image_token_mask = input_ids == self.tokenizer.image_token_placeholder_id
            
            # Pad the mask to the left with 0 to detect boundaries
            padded_mask = nn.functional.pad(image_token_mask, (1, 0), value=0)
            
            # Find the boundary (starting index) of the image tokens patch
            # A boundary is detected when mask changes from 0 to 1
            boundary = padded_mask[:, 1:] > padded_mask[:, :-1]
            
            # Number the boundaries
            numbered_boundary = torch.cumsum(boundary, dim=-1)

            # Create block indices for query and key
            q_block_indices = image_token_mask * numbered_boundary
            kv_block_indices = q_block_indices
            
            # Create bidirectional mask for image tokens
            # This allows tokens within the same image to attend to each other
            bidirectional_mask = torch.logical_and(
                kv_block_indices[:, None, :] == q_block_indices.unsqueeze(-1),
                q_block_indices.unsqueeze(-1) > 0,
            )
            
            # Combine causal mask with bidirectional mask for image tokens
            attention_mask = torch.logical_or(causal_mask, bidirectional_mask.unsqueeze(1))
            
            # Apply sliding window constraint to the combined mask if needed
            sliding_window_size = getattr(self.config, 'sliding_window_size', None)
            if sliding_window_size is not None:
                diagonal_offset = -(sliding_window_size-1)
            else:
                diagonal_offset = -1  # Default to full sequence
                
            # Create local mask by applying sliding window constraint
            local_mask = torch.logical_and(
                attention_mask,
                torch.triu(
                    torch.ones((1, 1, sequence_length, sequence_length), 
                              dtype=torch.bool, device=device),
                    diagonal=diagonal_offset
                )
            )
            
            return attention_mask, local_mask
            
        except Exception as e:
            print(f"Error creating multimodal attention mask: {e}, falling back to standard mask")
            # Fallback to standard causal mask if there's an error
            if hasattr(self.config, 'sliding_window_size') and self.config.sliding_window_size is not None:
                sliding_window_size = self.config.sliding_window_size
                print(f"Creating fallback sliding window mask with window size: {sliding_window_size}")
                
                # Create proper sliding window mask with correct dimensions
                local_mask_bool = causal_mask.clone()
                
                # Add the window size constraint if it's positive
                if sliding_window_size > 0:
                    # Create upper triangular mask with -window_size offset diagonal
                    window_mask = torch.triu(
                        torch.ones((batch_size, 1, sequence_length, sequence_length), 
                                  dtype=torch.bool, device=device),
                        diagonal=-(sliding_window_size-1)
                    )
                    
                    # Combine causal mask with sliding window mask
                    local_mask = torch.logical_and(local_mask_bool, window_mask)
                else:
                    local_mask = local_mask_bool
                    
                print(f"Created fallback sliding window mask with shape {local_mask.shape}")
            else:
                local_mask = causal_mask
                print(f"Using standard causal mask as fallback with shape {causal_mask.shape}")
                
            return causal_mask, local_mask

    def generate(
        self,
        prompts: Sequence[Sequence[Union[str, Image.Image]]],
        device: Any,
        output_len: int = 100,
        temperature: Union[float, None] = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
    ) -> Sequence[str]:
        """Generates responses for given prompts using Gemma model with TPA."""
        # Handle different model types
        if self.is_multimodal:
            # Process multimodal input with Gemma3 preprocessor
            processing_result = gemma3_preprocessor.tokenize_raw_input(
                    self.tokenizer, prompts, self.config, output_len, device
                )
            batch_size = processing_result["batch_size"]
            user_input_token_ids = processing_result["user_input_token_ids"]
            image_batch = processing_result["image_batch"]
            min_prompt_len = processing_result["min_prompt_len"]
            max_prompt_len = processing_result["max_prompt_len"]
            total_seq_len = processing_result["max_seq_len"]
            image_presence_mask = processing_result["image_presence_mask"]
        else:
            # Handle text-only model case
            if self.tokenizer is None and hasattr(self.config, 'tokenizer'):
                self.tokenizer = tokenizer.Tokenizer(self.config.tokenizer)
            elif self.tokenizer is None:
                # Look for tokenizer in common locations
                tokenizer_path = "tokenizer/tokenizer.model"
                if os.path.exists("gemma_models/tokenizer.model"):
                    tokenizer_path = "gemma_models/tokenizer.model"
                self.tokenizer = tokenizer.Tokenizer(tokenizer_path)
                
            # Basic preprocessing for text-only prompts
            batch_size = len(prompts)
            
            # Extract text prompts
            text_prompts = []
            for p in prompts:
                if isinstance(p, str):
                    text_prompts.append(p)
                elif isinstance(p, tuple) and len(p) > 0 and isinstance(p[0], str):
                    text_prompts.append(p[0])
                else:
                    raise ValueError(f"Unsupported prompt type: {type(p)}")
            
            # Tokenize prompts
            tokenized_prompts = [torch.tensor(self.tokenizer.encode(p), dtype=torch.long) for p in text_prompts]
            max_prompt_len = max(len(p) for p in tokenized_prompts)
            
            # Create padded tensor
            user_input_token_ids = torch.full((batch_size, max_prompt_len), 
                                             self.tokenizer.pad_id,
                                             dtype=torch.long, device=device)
            
            # Fill padded tensor with tokenized prompts
            for i, tokens in enumerate(tokenized_prompts):
                user_input_token_ids[i, :len(tokens)] = tokens
                
            # Set generation parameters
            min_prompt_len = max_prompt_len
            total_seq_len = max_prompt_len + output_len
            
            # None for text-only model
            image_batch = None
            image_presence_mask = None

        # Create attention mask.
        min_dtype = torch.finfo(self.dtype).min
        
        # Set a default sliding window for non-multimodal models if needed
        if self.config.sliding_window_size is None and not hasattr(self.config, 'sliding_window_size'):
            self.config.sliding_window_size = 1024  # Default size for Gemma models
        boolean_mask, local_boolean_mask = self.create_attention_mask(user_input_token_ids, total_seq_len)
        mask_tensor = torch.where(boolean_mask, 0, torch.tensor(min_dtype, dtype=torch.float32, device=device)).contiguous()
        local_mask_tensor = torch.where(local_boolean_mask, 0, torch.tensor(min_dtype, dtype=torch.float32, device=device)).contiguous()

        # Create TPA KV caches
        kv_caches = create_tpa_kv_caches(self.config, batch_size, total_seq_len, device)

        input_token_ids_tensor = torch.full((batch_size, min_prompt_len),
                                            self.tokenizer.pad_id,
                                            dtype=torch.int64, device=device)
        token_ids_tensor = user_input_token_ids.to(device)
        for i in range(batch_size):
            p = user_input_token_ids[i]
            input_token_ids_tensor[i, :min_prompt_len] = p[:min_prompt_len]

        input_positions_tensor = torch.arange(0, min_prompt_len, dtype=torch.int64, device=device)
        prompt_mask_tensor = token_ids_tensor != self.tokenizer.pad_id
        curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
        # Let the TPA attention create its own mask
        curr_local_mask_tensor = None
        output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(device)
        temperatures_tensor = None if not temperature else torch.FloatTensor(
                [temperature] * batch_size).to(device)
        top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
        top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)
        output_index = torch.tensor(min_prompt_len, dtype=torch.int64, device=device)

        # Prefill up to min_prompt_len tokens, then treat other prefill as
        # decode and ignore output.
        for i in range(total_seq_len - min_prompt_len):
            next_token_ids, _ = self(
                    input_token_ids=input_token_ids_tensor,
                    image_patches=image_batch,
                    image_presence_mask=image_presence_mask,
                    input_positions=input_positions_tensor,
                    kv_caches=kv_caches,
                    mask=curr_mask_tensor,
                    output_positions=output_positions_tensor,
                    temperatures=temperatures_tensor,
                    top_ps=top_ps_tensor,
                    top_ks=top_ks_tensor,
                    local_mask=curr_local_mask_tensor,
                )
            curr_prompt_mask = prompt_mask_tensor.index_select(
                    1, output_index).squeeze(dim=1)
            curr_token_ids = token_ids_tensor.index_select(
                    1, output_index).squeeze(dim=1)
            output_token_ids = torch.where(curr_prompt_mask, curr_token_ids,
                                           next_token_ids).unsqueeze(dim=1)
            token_ids_tensor.index_copy_(1, output_index, output_token_ids)

            input_token_ids_tensor = output_token_ids
            input_positions_tensor = output_index.unsqueeze(dim=-1)
            curr_mask_tensor = mask_tensor.index_select(2,
                                                        input_positions_tensor)
            # Let the TPA attention create its own mask
            curr_local_mask_tensor = None
            output_positions_tensor = torch.tensor(0, dtype=torch.int64, device=device)
            output_index = output_index + 1
            image_batch = None
            image_presence_mask = None

        # Detokenization.
        token_ids = token_ids_tensor.tolist()
        results = []
        for i, tokens in enumerate(token_ids):
            output = tokens
            if self.tokenizer.eos_id in output:
                eos_index = output.index(self.tokenizer.eos_id)
                output = output[:eos_index]
            results.append(self.tokenizer.decode(output))

        return results

    def load_weights(self, model_path: str):
        if os.path.isfile(model_path):
            self.load_state_dict(
                    torch.load(
                        model_path, mmap=True, weights_only=True,
                    )['model_state_dict'],
                    strict=False,
                )
        else:
            index_path = os.path.join(model_path, 'pytorch_model.bin.index.json')
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            shard_files = list(set(index["weight_map"].values()))
            for shard_file in shard_files:
                shard_path = os.path.join(model_path, shard_file)
                state_dict = torch.load(shard_path, map_location="cpu", weights_only=True)
                self.load_state_dict(state_dict, strict=False)
                del state_dict  # Save memory.
                gc.collect()

    def convert_from_standard_weights(self, standard_model: nn.Module) -> None:
        """
        Convert standard Gemma3 weights to TPA-compatible weights.
        
        This method takes a standard Gemma3 model and converts its weights
        to be compatible with the TPA architecture through factorization.
        
        Args:
            standard_model: A standard Gemma3ForMultimodalLM model instance
        """
        try:
            # Import tqdm for progress tracking here to avoid dependency issues
            import tqdm
            import time
            has_tqdm = True
        except ImportError:
            has_tqdm = False
            print("Warning: tqdm not found, progress bars will not be displayed")
        
        # Copy non-attention weights directly
        print("Copying non-attention weights...")
        
        # Handle different attribute naming between model types
        if hasattr(standard_model, 'text_token_embedder'):
            # For Gemma3 multimodal models
            print("Using Gemma3 multimodal model attribute names...")
            self.text_token_embedder.load_state_dict(standard_model.text_token_embedder.state_dict())
        elif hasattr(standard_model, 'embedder'):
            # For standard GemmaForCausalLM models
            print("Using standard Gemma model attribute names...")
            self.text_token_embedder.load_state_dict(standard_model.embedder.state_dict())
        else:
            raise ValueError("Could not find a compatible embedding layer in the standard model")
            
        self.sampler.load_state_dict(standard_model.sampler.state_dict())
        
        if hasattr(standard_model, 'siglip_vision_model') and hasattr(self, 'siglip_vision_model'):
            print("Copying vision model weights...")
            self.siglip_vision_model.load_state_dict(standard_model.siglip_vision_model.state_dict())
        
        if hasattr(standard_model, 'mm_soft_embedding_norm') and hasattr(self, 'mm_soft_embedding_norm'):
            self.mm_soft_embedding_norm.load_state_dict(standard_model.mm_soft_embedding_norm.state_dict())
        
        if hasattr(standard_model, 'mm_input_projection') and hasattr(self, 'mm_input_projection'):
            self.mm_input_projection.load_state_dict(standard_model.mm_input_projection.state_dict())
        
        # Convert attention layers
        total_layers = len(standard_model.model.layers)
        print(f"Converting {total_layers} attention layers...")
        
        # Set up progress bar
        if has_tqdm:
            progress_bar = tqdm.tqdm(total=total_layers, desc="Converting layers")
        
        layer_total_time = 0
        
        for i, (std_layer, tpa_layer) in enumerate(zip(standard_model.model.layers, self.model.layers)):
            if has_tqdm:
                layer_start = time.time()
            
            # Copy non-attention weights (MLP, LayerNorms)
            tpa_layer.mlp.load_state_dict(std_layer.mlp.state_dict())
            tpa_layer.input_layernorm.load_state_dict(std_layer.input_layernorm.state_dict())
            tpa_layer.post_attention_layernorm.load_state_dict(std_layer.post_attention_layernorm.state_dict())
            
            if hasattr(tpa_layer, 'pre_feedforward_layernorm') and tpa_layer.pre_feedforward_layernorm is not None:
                if hasattr(std_layer, 'pre_feedforward_layernorm') and std_layer.pre_feedforward_layernorm is not None:
                    tpa_layer.pre_feedforward_layernorm.load_state_dict(std_layer.pre_feedforward_layernorm.state_dict())
            
            if hasattr(tpa_layer, 'post_feedforward_layernorm') and tpa_layer.post_feedforward_layernorm is not None:
                if hasattr(std_layer, 'post_feedforward_layernorm') and std_layer.post_feedforward_layernorm is not None:
                    tpa_layer.post_feedforward_layernorm.load_state_dict(std_layer.post_feedforward_layernorm.state_dict())
            
            # Get standard attention QKV projections
            qkv_weight = std_layer.self_attn.qkv_proj.weight
            q_size = self.config.num_attention_heads * self.config.head_dim
            kv_size = self.config.num_key_value_heads * self.config.head_dim
            
            # Split into Q, K, V
            q_weight = qkv_weight[:q_size]
            k_weight = qkv_weight[q_size:q_size+kv_size]
            v_weight = qkv_weight[q_size+kv_size:]
            
            # Factorize using SVD for TPA
            if not has_tqdm:
                print(f"Layer {i+1}/{total_layers}: Factorizing attention weights...")
                
            self._factorize_and_set_weights(q_weight, tpa_layer.self_attn.W_A_q, tpa_layer.self_attn.W_B_q, self.config.q_rank)
            self._factorize_and_set_weights(k_weight, tpa_layer.self_attn.W_A_k, tpa_layer.self_attn.W_B_k, self.config.k_rank)
            self._factorize_and_set_weights(v_weight, tpa_layer.self_attn.W_A_v, tpa_layer.self_attn.W_B_v, self.config.v_rank)
            
            # Copy output projection
            tpa_layer.self_attn.o_proj.load_state_dict(std_layer.self_attn.o_proj.state_dict())
            
            # Update progress bar
            if has_tqdm:
                layer_time = time.time() - layer_start
                layer_total_time += layer_time
                remaining_layers = total_layers - (i + 1)
                est_remaining_time = (layer_total_time / (i + 1)) * remaining_layers
                
                progress_bar.set_postfix({
                    "Layer": f"{i+1}/{total_layers}", 
                    "Est. time remaining": f"{est_remaining_time:.1f}s"
                })
                progress_bar.update(1)
        
        if has_tqdm:
            progress_bar.close()
        
        # Copy final layer norm
        print("Copying final layer norm...")
        if hasattr(standard_model.model, 'norm') and hasattr(self.model, 'norm'):
            self.model.norm.load_state_dict(standard_model.model.norm.state_dict())
        
        print("Model conversion completed successfully!")
    
    def _factorize_and_set_weights(self, weight: torch.Tensor, A_proj: nn.Module, B_proj: nn.Module, rank: int):
        """
        Factorize a weight matrix using SVD and set the factorized weights.
        
        Args:
            weight: Weight matrix to factorize
            A_proj: A projection module
            B_proj: B projection module
            rank: Rank for factorization
        """
        # Get target dimensions from projection modules
        hidden_size = self.config.hidden_size
        target_A_shape = A_proj.weight.shape
        target_B_shape = B_proj.weight.shape
        
        # Expected dimensions for the factorized matrices
        num_heads = self.config.num_attention_heads
        num_kv_heads = getattr(self.config, 'num_key_value_heads', num_heads)
        head_dim = self.config.head_dim
        
        print(f"Processing weight with shape {weight.shape}")
        print(f"Target shapes - A: {target_A_shape}, B: {target_B_shape}")
        
        # Convert weight to float32 for better numerical stability
        weight_float32 = weight.to(dtype=torch.float32)
        
        # Explicitly reshape weight for SVD based on whether it's Q, K, or V
        if weight.shape[0] == num_heads * head_dim:
            # This is a query weight
            print(f"Identified as query weight (num_heads={num_heads}, head_dim={head_dim})")
            weight_2d = weight_float32  # Keep as is for SVD
        else:
            # This is a key or value weight
            print(f"Identified as key/value weight (num_kv_heads={num_kv_heads}, head_dim={head_dim})")
            weight_2d = weight_float32  # Keep as is for SVD
        
        # Perform SVD on the 2D weight matrix
        try:
            print(f"Performing SVD on tensor of shape {weight_2d.shape}")
            U, S, Vh = torch.linalg.svd(weight_2d, full_matrices=False)
            
            # Limit to specified rank
            effective_rank = min(rank, min(U.shape[1], S.shape[0], Vh.shape[0]))
            print(f"Using effective rank: {effective_rank} (requested: {rank})")
            
            # Create scaled factors
            sqrt_S = torch.sqrt(S[:effective_rank])
            U_scaled = U[:, :effective_rank] * sqrt_S
            Vh_scaled = Vh[:effective_rank] * sqrt_S.unsqueeze(1)
            
            print(f"Factorized shapes - U_scaled: {U_scaled.shape}, Vh_scaled: {Vh_scaled.shape}")
            
            # Create direct A and B factors matching projection shapes
            # We'll initialize with zeros and then fill with as much data as possible
            A_weight = torch.zeros(target_A_shape[1], target_A_shape[0], 
                                  dtype=weight.dtype, device=weight.device)
            B_weight = torch.zeros(target_B_shape[1], target_B_shape[0], 
                                  dtype=weight.dtype, device=weight.device)
            
            # Reshape U_scaled and Vh_scaled to fill as much of A_weight and B_weight as possible
            if 'W_A_q' in A_proj.__class__.__name__ or 'W_A_q' in A_proj._get_name():
                # For Q projections, reshape according to q_rank
                q_rank = effective_rank
                A_entries = min(U_scaled.shape[0] * q_rank, A_weight.numel())
                B_entries = min(Vh_scaled.shape[0] * Vh_scaled.shape[1], B_weight.numel())
                
                # Flatten and copy data
                A_weight.view(-1)[:A_entries] = U_scaled.reshape(-1)[:A_entries]
                B_weight.view(-1)[:B_entries] = Vh_scaled.reshape(-1)[:B_entries]
                
            elif 'W_A_k' in A_proj.__class__.__name__ or 'W_A_k' in A_proj._get_name():
                # For K projections
                k_rank = effective_rank
                A_entries = min(U_scaled.shape[0] * k_rank, A_weight.numel())
                B_entries = min(Vh_scaled.shape[0] * Vh_scaled.shape[1], B_weight.numel())
                
                # Flatten and copy data
                A_weight.view(-1)[:A_entries] = U_scaled.reshape(-1)[:A_entries]
                B_weight.view(-1)[:B_entries] = Vh_scaled.reshape(-1)[:B_entries]
                
            elif 'W_A_v' in A_proj.__class__.__name__ or 'W_A_v' in A_proj._get_name():
                # For V projections
                v_rank = effective_rank
                A_entries = min(U_scaled.shape[0] * v_rank, A_weight.numel())
                B_entries = min(Vh_scaled.shape[0] * Vh_scaled.shape[1], B_weight.numel())
                
                # Flatten and copy data
                A_weight.view(-1)[:A_entries] = U_scaled.reshape(-1)[:A_entries]
                B_weight.view(-1)[:B_entries] = Vh_scaled.reshape(-1)[:B_entries]
                
            else:
                # Generic approach
                A_entries = min(U_scaled.numel(), A_weight.numel())
                B_entries = min(Vh_scaled.numel(), B_weight.numel())
                
                A_weight.view(-1)[:A_entries] = U_scaled.reshape(-1)[:A_entries]
                B_weight.view(-1)[:B_entries] = Vh_scaled.reshape(-1)[:B_entries]
            
            print(f"Final shapes - A_weight: {A_weight.shape}, B_weight: {B_weight.shape}")
            
            # Set weights to the projection modules
            with torch.no_grad():
                # Convert to the original dtype
                A_weight = A_weight.to(dtype=A_proj.weight.dtype)
                B_weight = B_weight.to(dtype=B_proj.weight.dtype)
                
                # Copy to the modules - we transpose here because Linear expects (out_features, in_features)
                A_proj.weight.copy_(A_weight.transpose(0, 1))
                B_proj.weight.copy_(B_weight.transpose(0, 1))
                
                # Set weight scalers if using quantization
                if hasattr(self.config, 'quant') and self.config.quant:
                    if hasattr(A_proj, 'weight_scaler'):
                        A_proj.weight_scaler.fill_(1.0)
                    if hasattr(B_proj, 'weight_scaler'):
                        B_proj.weight_scaler.fill_(1.0)
                        
        except Exception as e:
            print(f"SVD or weight setting failed: {e}")
            # Fill with small random values as a fallback
            with torch.no_grad():
                nn.init.normal_(A_proj.weight, mean=0.0, std=0.02)
                nn.init.normal_(B_proj.weight, mean=0.0, std=0.02)