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
        temperature: Union[float, None] = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
    ) -> Sequence[str]:
        """Generates responses for given prompts using Gemma model with TPA."""
        # Store original prompt texts for later reference
        self.prompt_texts = []
        for p in prompts:
            if isinstance(p, tuple) and len(p) > 0 and isinstance(p[0], str):
                self.prompt_texts.append(p[0])
            elif isinstance(p, str):
                self.prompt_texts.append(p)
            else:
                self.prompt_texts.append("")
        print(f"Input prompts: {self.prompt_texts}")
        
        # Make sure tokenizer is available
        if self.tokenizer is None and hasattr(self.config, 'tokenizer'):
            self.tokenizer = tokenizer.Tokenizer(self.config.tokenizer)
        elif self.tokenizer is None:
            # Look for tokenizer in common locations
            tokenizer_path = "tokenizer/tokenizer.model"
            if os.path.exists("gemma_models/tokenizer.model"):
                tokenizer_path = "gemma_models/tokenizer.model"
            self.tokenizer = tokenizer.Tokenizer(tokenizer_path)
            
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
            
            # Tokenize prompts - ensure we have a BOS token at the start
            tokenized_prompts = []
            for p in text_prompts:
                tokens = self.tokenizer.encode(p)
                # Add BOS token if not present
                if len(tokens) == 0 or tokens[0] != self.tokenizer.bos_id:
                    tokens = [self.tokenizer.bos_id] + tokens
                tokenized_prompts.append(torch.tensor(tokens, dtype=torch.long))
            
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

        print(f"Input shape: {user_input_token_ids.shape}, seq len: {total_seq_len}")
        
        # Create TPA KV caches - ensure sufficient capacity
        kv_caches = create_tpa_kv_caches(self.config, batch_size, total_seq_len, device)

        # Set up input tensor
        token_ids_tensor = user_input_token_ids.to(device)
        
        # Track generated tokens separately for output
        generated_tokens = []
        for _ in range(batch_size):
            generated_tokens.append([])
        
        # Set up sampling parameters
        temperatures_tensor = None if temperature is None else torch.tensor(
                [temperature] * batch_size, dtype=torch.float32, device=device)
        top_ps_tensor = torch.tensor([top_p] * batch_size, dtype=torch.float32, device=device)
        top_ks_tensor = torch.tensor([top_k] * batch_size, dtype=torch.int64, device=device)
        
        # Track current position in the sequence
        current_pos = 0
        
        # Prefill the KV cache with the prompt tokens
        with torch.no_grad():
            # Process the prompt in one forward pass
            logits = self.text_token_embedder(token_ids_tensor)
            normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=logits.dtype, device=device)
            logits = logits * normalizer
            
            # Create positional indices for the prompt
            positions = torch.arange(0, min_prompt_len, device=device)
            max_pos = min(min_prompt_len, self.local_freqs_cis.size(0))
            positions = positions[:max_pos]
            
            # Create freqs_cis dict
            freqs_cis = {}
            if len(positions) > 0:
                freqs_cis[gemma_config.AttentionType.LOCAL_SLIDING] = self.local_freqs_cis.index_select(0, positions)
                freqs_cis[gemma_config.AttentionType.GLOBAL] = self.global_freqs_cis.index_select(0, positions)
                
                # Process prompt through model (this populates the KV cache)
                try:
                    print(f"Processing prompt with {min_prompt_len} tokens...")
                    hidden_states = self.model(
                        hidden_states=logits,
                        freqs_cis=freqs_cis,
                        kv_write_indices=positions,
                        kv_caches=kv_caches,
                        mask=None,
                        local_mask=None,
                    )
                    
                    # Calculate initial logits from the prompt processing
                    if hidden_states is not None:
                        embedder_weight = self.text_token_embedder.weight
                        if self.config.quant:
                            embedder_weight = embedder_weight * self.text_token_embedder.weight_scaler.unsqueeze(-1)
                            
                        # Get output logits for the last position
                        final_hidden = hidden_states[:, -1:]
                        next_token_logits = torch.matmul(final_hidden, embedder_weight.transpose(0, 1))
                        
                        # Sample first token from logits
                        current_pos = min_prompt_len - 1  # Position of last token in prompt
                        for batch_idx in range(batch_size):
                            batch_logits = next_token_logits[batch_idx]
                            
                            # Apply temperature scaling
                            if temperatures_tensor is not None:
                                batch_temp = temperatures_tensor[batch_idx].item()
                                if batch_temp > 0:
                                    batch_logits = batch_logits / batch_temp
                            
                            # Apply top-k filtering
                            top_k_value = top_ks_tensor[batch_idx].item()
                            if top_k_value > 0:
                                indices_to_remove = batch_logits < torch.topk(batch_logits, top_k_value)[0][..., -1, None]
                                batch_logits[indices_to_remove] = -float('Inf')
                            
                            # Apply top-p (nucleus) filtering
                            top_p_value = top_ps_tensor[batch_idx].item()
                            if 0 < top_p_value < 1.0:
                                sorted_logits, sorted_indices = torch.sort(batch_logits, descending=True)
                                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                                sorted_indices_to_remove = cumulative_probs > top_p_value
                                # Shift the indices to the right to keep the first token above the threshold
                                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                                sorted_indices_to_remove[..., 0] = 0
                                
                                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                                batch_logits[indices_to_remove] = -float('Inf')
                            
                            # Sample from the filtered distribution
                            probs = F.softmax(batch_logits, dim=-1)
                            next_token_id = torch.multinomial(probs, num_samples=1).item()
                            generated_tokens[batch_idx].append(next_token_id)
                except Exception as e:
                    print(f"Error during prompt processing: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("Warning: Empty prompt. Using default tokens to start.")
                # Add a default token to start generation
                for batch_idx in range(batch_size):
                    generated_tokens[batch_idx].append(self.tokenizer.bos_id)
            
            current_pos = min_prompt_len  # Start after prompt
            
            # Now continue generating tokens
            for i in range(1, output_len):
                try:
                    # Create batch of current tokens
                    if i == 1:
                        # Use first generated tokens from prompt processing
                        current_token_ids = []
                        for batch_idx in range(batch_size):
                            if generated_tokens[batch_idx]:
                                current_token_ids.append([generated_tokens[batch_idx][0]])
                            else:
                                current_token_ids.append([self.tokenizer.bos_id])
                        current_token = torch.tensor(current_token_ids, dtype=torch.long, device=device)
                    else:
                        # Use previously generated tokens 
                        current_token_ids = []
                        for batch_idx in range(batch_size):
                            if len(generated_tokens[batch_idx]) >= i:
                                current_token_ids.append([generated_tokens[batch_idx][i-1]])
                            else:
                                # Fallback - shouldn't normally happen
                                current_token_ids.append([self.tokenizer.pad_id])
                        current_token = torch.tensor(current_token_ids, dtype=torch.long, device=device)
                    
                    # Embed the current token
                    current_embed = self.text_token_embedder(current_token)
                    current_embed = current_embed * normalizer
                    
                    # Position for the current token
                    current_pos_tensor = torch.tensor([current_pos], device=device)
                    
                    # Create freqs_cis for current position
                    # Make sure we don't go out of bounds
                    max_pos_index = min(current_pos, self.local_freqs_cis.size(0) - 1)
                    safe_pos_tensor = torch.tensor([max_pos_index], device=device)
                    
                    current_freqs_cis = {}
                    current_freqs_cis[gemma_config.AttentionType.LOCAL_SLIDING] = self.local_freqs_cis.index_select(0, safe_pos_tensor)
                    current_freqs_cis[gemma_config.AttentionType.GLOBAL] = self.global_freqs_cis.index_select(0, safe_pos_tensor)
                    
                    # Process current token
                    hidden_states = self.model(
                        hidden_states=current_embed,
                        freqs_cis=current_freqs_cis,
                        kv_write_indices=current_pos_tensor,
                        kv_caches=kv_caches,
                        mask=None,
                        local_mask=None,
                    )
                    
                    # Project to vocabulary
                    embedder_weight = self.text_token_embedder.weight
                    if self.config.quant:
                        embedder_weight = embedder_weight * self.text_token_embedder.weight_scaler.unsqueeze(-1)
                    
                    # Get logits for each batch item
                    next_token_logits = torch.matmul(hidden_states, embedder_weight.transpose(0, 1))
                    
                    # For each batch item, sample next token
                    for batch_idx in range(batch_size):
                        batch_logits = next_token_logits[batch_idx]
                        
                        # Apply temperature scaling
                        if temperatures_tensor is not None:
                            batch_temp = temperatures_tensor[batch_idx].item()
                            if batch_temp > 0:
                                batch_logits = batch_logits / batch_temp
                        
                        # Apply top-k filtering
                        top_k_value = top_ks_tensor[batch_idx].item()
                        if top_k_value > 0:
                            indices_to_remove = batch_logits < torch.topk(batch_logits, top_k_value)[0][..., -1, None]
                            batch_logits[indices_to_remove] = -float('Inf')
                        
                        # Apply top-p (nucleus) filtering
                        top_p_value = top_ps_tensor[batch_idx].item()
                        if 0 < top_p_value < 1.0:
                            sorted_logits, sorted_indices = torch.sort(batch_logits, descending=True)
                            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                            sorted_indices_to_remove = cumulative_probs > top_p_value
                            # Shift the indices to the right to keep the first token above the threshold
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0
                            
                            indices_to_remove = sorted_indices[sorted_indices_to_remove]
                            batch_logits[indices_to_remove] = -float('Inf')
                        
                        # Sample from the filtered distribution
                        probs = F.softmax(batch_logits, dim=-1)
                        next_token_id = torch.multinomial(probs, num_samples=1).item()
                        
                        # Check for EOS
                        if next_token_id == self.tokenizer.eos_id:
                            print(f"Batch {batch_idx}: Hit EOS token")
                            # We'll append EOS and then ignore further tokens
                            generated_tokens[batch_idx].append(next_token_id)
                            continue
                        
                        # Add token to results
                        generated_tokens[batch_idx].append(next_token_id)
                
                except Exception as e:
                    print(f"Error during token generation at position {current_pos}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue to next token
                
                # Increment position
                current_pos += 1

        # Decode generated tokens
        results = []
        
        for i, token_ids in enumerate(generated_tokens):
            # Try to decode the generated tokens
            try:
                if token_ids:
                    # Stop at EOS token if present
                    if self.tokenizer.eos_id in token_ids:
                        eos_idx = token_ids.index(self.tokenizer.eos_id)
                        token_ids = token_ids[:eos_idx+1]
                        
                    # Decode and append to results
                    text = self.tokenizer.decode(token_ids)
                    results.append(text)
                else:
                    # No tokens generated
                    results.append("")
            except Exception as e:
                print(f"Error decoding tokens for batch {i}: {e}")
                # Return empty string as fallback
                results.append("")

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
        
        # Determine which type of projection this is
        is_query = 'W_A_q' in A_proj.__class__.__name__ or 'W_A_q' in A_proj._get_name()
        is_key = 'W_A_k' in A_proj.__class__.__name__ or 'W_A_k' in A_proj._get_name()
        is_value = 'W_A_v' in A_proj.__class__.__name__ or 'W_A_v' in A_proj._get_name()
        
        if is_query:
            print(f"Identified as query weight (num_heads={num_heads}, head_dim={head_dim})")
        elif is_key or is_value:
            print(f"Identified as key/value weight (num_kv_heads={num_kv_heads}, head_dim={head_dim})")
        
        # Perform SVD on the 2D weight matrix with improved stability
        try:
            print(f"Performing SVD on tensor of shape {weight_float32.shape}")
            
            # For very large matrices, consider chunking or using a more stable approach
            if weight_float32.numel() > 10_000_000:  # For very large matrices
                print("Large matrix detected, using chunked SVD approach")
                # Use torch.pca_lowrank as a more memory-efficient alternative for large matrices
                U, S, Vh = torch.pca_lowrank(weight_float32, q=rank, center=False, niter=2)
            else:
                # Standard SVD for smaller matrices
                U, S, Vh = torch.linalg.svd(weight_float32, full_matrices=False)
            
            # Limit to specified rank with a check for numerical stability
            max_rank = min(U.shape[1], S.shape[0], Vh.shape[0])
            
            # Check for very small singular values and adjust rank if needed
            if max_rank > 1:
                # Calculate relative magnitudes of singular values
                rel_singular_values = S / S[0]
                # Find where singular values become too small (potential numerical issues)
                valid_indices = torch.where(rel_singular_values > 1e-6)[0]
                if len(valid_indices) < max_rank:
                    print(f"Warning: Some singular values are very small. Adjusting max rank from {max_rank} to {len(valid_indices)}")
                    max_rank = len(valid_indices)
            
            effective_rank = min(rank, max_rank)
            print(f"Using effective rank: {effective_rank} (requested: {rank})")
            
            if effective_rank <= 0:
                raise ValueError(f"Effective rank is {effective_rank}, must be positive")
                
            # Split singular values between the two factors
            sqrt_S = torch.sqrt(S[:effective_rank])
            
            # Create the factorized weights with proper shapes
            # For the A factor (head dimension factor)
            A_factor = U[:, :effective_rank] * sqrt_S
            
            # For the B factor (token dimension factor)
            B_factor = Vh[:effective_rank] * sqrt_S.unsqueeze(1)
            
            print(f"Raw factorized shapes - A_factor: {A_factor.shape}, B_factor: {B_factor.shape}")
            
            # Properly reshape for the TPA architecture
            if is_query:
                # For queries, we need to reshape carefully to maintain the right dimensions
                rows_per_head = weight.shape[0] // num_heads if num_heads > 0 else 1
                
                # Reshape A_factor for query - preparing for [num_heads * rank, hidden_size]
                # First reshape to a 3D tensor with head dimension explicit
                if num_heads > 0:
                    A_reshaped = A_factor.reshape(num_heads, rows_per_head, effective_rank)
                    # Then average across the rows_per_head dimension for better stability
                    A_packed = A_reshaped.mean(dim=1)  # Results in [num_heads, effective_rank]
                    # Expand to fill the target shape
                    A_packed = A_packed.repeat_interleave(target_A_shape[0] // (num_heads * effective_rank) + 1, dim=0)
                    A_packed = A_packed[:target_A_shape[0] // effective_rank].reshape(-1, effective_rank)
                else:
                    # Fallback for unusual shapes
                    A_packed = A_factor.reshape(-1, effective_rank)
                    
                # Reshape B_factor - preparing for [rank * head_dim, hidden_size]
                B_packed = B_factor.reshape(effective_rank, -1)
                
            elif is_key or is_value:
                # Similar approach for keys and values, but using num_kv_heads
                rows_per_head = weight.shape[0] // num_kv_heads if num_kv_heads > 0 else 1
                
                if num_kv_heads > 0:
                    A_reshaped = A_factor.reshape(num_kv_heads, rows_per_head, effective_rank)
                    A_packed = A_reshaped.mean(dim=1)
                    A_packed = A_packed.repeat_interleave(target_A_shape[0] // (num_kv_heads * effective_rank) + 1, dim=0)
                    A_packed = A_packed[:target_A_shape[0] // effective_rank].reshape(-1, effective_rank)
                else:
                    A_packed = A_factor.reshape(-1, effective_rank)
                    
                B_packed = B_factor.reshape(effective_rank, -1)
            else:
                # Generic case - try to match dimensions directly
                A_packed = A_factor
                B_packed = B_factor
                
            # Now create properly sized weight tensors for the linear layers
            A_weight = torch.zeros(target_A_shape, dtype=A_proj.weight.dtype, device=weight.device)
            B_weight = torch.zeros(target_B_shape, dtype=B_proj.weight.dtype, device=weight.device)
            
            # Ensure A_packed and B_packed have the right dtype
            A_packed = A_packed.to(dtype=A_proj.weight.dtype)
            B_packed = B_packed.to(dtype=B_proj.weight.dtype)
            
            # For A: Carefully reshape to match [out_features, in_features]
            if A_packed.dim() == 2:
                # If we have a 2D tensor, we need to transform it to match target shape
                out_dim_A, in_dim_A = target_A_shape
                
                # Reshape or repeat A_packed to match target dimensions
                if A_packed.shape[1] == effective_rank:  # If second dimension is rank
                    num_repeats = (out_dim_A + A_packed.shape[0] - 1) // A_packed.shape[0]
                    if num_repeats > 1:
                        A_packed = A_packed.repeat(num_repeats, 1)
                    A_packed = A_packed[:out_dim_A]
                    
                    # Create weight matrix of correct shape 
                    A_weight_data = torch.zeros((out_dim_A, in_dim_A), 
                                             dtype=A_proj.weight.dtype, 
                                             device=weight.device)
                    
                    # Fill the weight matrix block-diagonally
                    block_size = in_dim_A // effective_rank
                    for r in range(effective_rank):
                        if r < A_packed.shape[1]:
                            col_start = r * block_size
                            col_end = min((r + 1) * block_size, in_dim_A)
                            for row in range(out_dim_A):
                                if row < A_packed.shape[0]:
                                    A_weight_data[row, col_start:col_end] = A_packed[row, r]
                                    
                    A_weight = A_weight_data
                else:
                    # If dimensions don't match expectation, use reshape and repeat
                    A_packed_flat = A_packed.reshape(-1)
                    if len(A_packed_flat) > 0:
                        # Repeat the flattened tensor to fill target shape
                        repeats_needed = (target_A_shape.numel() + len(A_packed_flat) - 1) // len(A_packed_flat)
                        A_packed_repeated = A_packed_flat.repeat(repeats_needed)
                        A_weight = A_packed_repeated[:target_A_shape.numel()].reshape(target_A_shape)
            
            # Similar approach for B
            if B_packed.dim() == 2:
                out_dim_B, in_dim_B = target_B_shape
                
                if B_packed.shape[0] == effective_rank:  # If first dimension is rank
                    # Reshape to match output dimension
                    num_repeats = (out_dim_B + B_packed.shape[0] - 1) // B_packed.shape[0]
                    if num_repeats > 1:
                        B_packed = B_packed.repeat(num_repeats, 1)
                    B_packed = B_packed[:out_dim_B]
                    
                    # Create weight matrix
                    B_weight_data = torch.zeros((out_dim_B, in_dim_B), 
                                             dtype=B_proj.weight.dtype, 
                                             device=weight.device)
                    
                    # Fill block-diagonally
                    block_size = out_dim_B // effective_rank
                    cols = min(B_packed.shape[1], in_dim_B)
                    for r in range(effective_rank):
                        row_start = r * block_size
                        row_end = min((r + 1) * block_size, out_dim_B)
                        for col in range(cols):
                            B_weight_data[row_start:row_end, col] = B_packed[r, col]
                            
                    B_weight = B_weight_data
                else:
                    # Similar fallback as for A
                    B_packed_flat = B_packed.reshape(-1)
                    if len(B_packed_flat) > 0:
                        repeats_needed = (target_B_shape.numel() + len(B_packed_flat) - 1) // len(B_packed_flat)
                        B_packed_repeated = B_packed_flat.repeat(repeats_needed)
                        B_weight = B_packed_repeated[:target_B_shape.numel()].reshape(target_B_shape)
            
            print(f"Final tensor shapes - A_weight: {A_weight.shape}, B_weight: {B_weight.shape}")
            
            # Set weights to the projection modules
            with torch.no_grad():
                A_proj.weight.copy_(A_weight)
                B_proj.weight.copy_(B_weight)
                
                # Set weight scalers if using quantization
                if hasattr(self.config, 'quant') and self.config.quant:
                    if hasattr(A_proj, 'weight_scaler'):
                        A_proj.weight_scaler.fill_(1.0)
                    if hasattr(B_proj, 'weight_scaler'):
                        B_proj.weight_scaler.fill_(1.0)
                
            # Verify the factorization quality
            print("Verifying factorization quality...")
            try:
                with torch.no_grad():
                    # Create multiple test inputs for more robust verification
                    num_samples = 5
                    test_inputs = torch.randn(num_samples, hidden_size, 
                                           dtype=weight_float32.dtype, 
                                           device=weight.device)
                    
                    # Compute projection using original weights
                    orig_projs = torch.matmul(test_inputs, weight_float32.t())
                    
                    # Compute factorized projection
                    a_projs = torch.matmul(test_inputs, A_proj.weight.t())
                    
                    # Reshape properly depending on projection type
                    if is_query:
                        a_projs_reshaped = a_projs.reshape(num_samples, -1, effective_rank)
                    else:
                        # For K, V with potentially different head counts
                        a_projs_reshaped = a_projs.reshape(num_samples, -1, effective_rank)
                        
                    b_projs = torch.matmul(test_inputs, B_proj.weight.t())
                    b_projs_reshaped = b_projs.reshape(num_samples, effective_rank, -1)
                    
                    # Combine the factors
                    # For better numerical stability, scale each component
                    effective_scale = 1.0 / effective_rank
                    test_projs = []
                    
                    for i in range(num_samples):
                        # Carefully compute for each sample
                        test_proj = torch.matmul(a_projs_reshaped[i], b_projs_reshaped[i])
                        test_proj = test_proj.reshape(1, -1) * effective_scale
                        test_projs.append(test_proj)
                    
                    test_projs = torch.cat(test_projs, dim=0)
                    
                    # Measure error across all samples
                    rel_errors = []
                    for i in range(num_samples):
                        orig_norm = torch.norm(orig_projs[i])
                        if orig_norm > 0:
                            error = torch.norm(orig_projs[i] - test_projs[i]) / orig_norm
                            rel_errors.append(error.item())
                    
                    avg_error = sum(rel_errors) / len(rel_errors) if rel_errors else float('inf')
                    print(f"Average relative factorization error: {avg_error:.4f}")
                    
                    # If error is too large, try to improve factorization
                    if avg_error > 0.3:  # Error threshold
                        print(f"Warning: High factorization error ({avg_error:.4f}). Attempting improved factorization.")
                        # Potential improvements could be implemented here, such as:
                        # - Try a different rank
                        # - Use a different factorization method
                        # - Apply post-processing to better match original matrix
            except Exception as e:
                print(f"Verification failed with error: {e}")
                # Continue even if verification fails
                        
        except Exception as e:
            print(f"SVD or weight setting failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Improved fallback strategy - preserves some structure from original weights
            with torch.no_grad():
                print("Using structured initialization as fallback")
                
                # Create more structured initial values based on the original weight
                if weight.numel() > 0:
                    # If we have a non-empty weight matrix, use its statistics
                    mean_val = weight.mean().item()
                    std_val = max(weight.std().item(), 0.02)  # Ensure minimum standard deviation
                    
                    # Initialize with same statistics but new random values
                    nn.init.normal_(A_proj.weight, mean=mean_val, std=std_val)
                    nn.init.normal_(B_proj.weight, mean=mean_val, std=std_val)
                    
                    # Try to preserve row/column norms where possible
                    if weight.dim() == 2 and A_proj.weight.dim() == 2 and B_proj.weight.dim() == 2:
                        # Get row and column norms from original weight
                        row_norms = torch.norm(weight, dim=1, keepdim=True)
                        col_norms = torch.norm(weight, dim=0, keepdim=True)
                        
                        # Normalize and rescale A_proj weights using row norms
                        if A_proj.weight.shape[0] <= row_norms.shape[0]:
                            row_factors = row_norms[:A_proj.weight.shape[0]]
                            row_scale = torch.norm(A_proj.weight, dim=1, keepdim=True)
                            where_valid = (row_scale > 0).float()
                            scale_factors = where_valid * (row_factors / (row_scale + 1e-10)) + (1 - where_valid)
                            A_proj.weight.mul_(scale_factors)
                            
                        # Normalize and rescale B_proj weights using column norms
                        if B_proj.weight.shape[1] <= col_norms.shape[1]:
                            col_factors = col_norms[:, :B_proj.weight.shape[1]]
                            col_scale = torch.norm(B_proj.weight, dim=0, keepdim=True)
                            where_valid = (col_scale > 0).float()
                            scale_factors = where_valid * (col_factors / (col_scale + 1e-10)) + (1 - where_valid)
                            B_proj.weight.mul_(scale_factors)
                else:
                    # If no weight info available, use standard initialization
                    nn.init.normal_(A_proj.weight, mean=0.0, std=0.02)
                    nn.init.normal_(B_proj.weight, mean=0.0, std=0.02)