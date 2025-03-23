"""Inference-only Gemma 3 multimodal model implementation with TPA."""

import torch
import os
import json
import gc
from torch import nn
import torch.nn.functional as F
from PIL import Image
from typing import Any, List, Optional, Sequence, Tuple, Union, Mapping

from .. import model as gemma_model
from .. import config as gemma_config
from .. import gemma3_preprocessor
from .. import tokenizer
from ..siglip_vision import siglip_vision_model


class RMSNorm(nn.Module):
    """RMS Normalization module."""
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        add_unit_offset: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        if self.add_unit_offset:
            output = output * (1 + self.weight.float())
        else:
            output = output * self.weight.float()
        return output.type_as(x)


class TPAAttention(nn.Module):
    """Tensor Product Attention module for Gemma 3."""
    def __init__(self, config: gemma_config.GemmaConfig, attn_type: gemma_config.AttentionType):
        super().__init__()
        self.config = config
        self.attn_type = attn_type
        
        self.num_heads = config.num_attention_heads
        # Simplified: Only use num_heads - no GQA
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        
        # TPA specific parameters
        self.q_rank = getattr(config, 'q_rank', 6)
        self.k_rank = getattr(config, 'k_rank', 2)
        self.v_rank = getattr(config, 'v_rank', 2)
        
        # Simplified: Use consistent head dimensions
        self.head_dim = config.head_dim
        
        # Debug info about dimensions
        print(f"TPAAttention: hidden_size={self.hidden_size}")
        print(f"TPAAttention: num_heads={self.num_heads}")
        print(f"TPAAttention: head_dim={self.head_dim}")
        print(f"TPAAttention: ranks: q={self.q_rank}, k={self.k_rank}, v={self.v_rank}")
        
        # Scaling for attention
        if config.query_pre_attn_scalar is not None:
            self.scaling = config.query_pre_attn_scalar**-0.5
        else:
            self.scaling = self.head_dim**-0.5
        
        # Define TPA projection matrices with appropriate dimensions
        # W_A projections: hidden_size -> num_heads * rank
        self.W_A_q = nn.Linear(self.hidden_size, self.num_heads * self.q_rank, bias=False)
        self.W_A_k = nn.Linear(self.hidden_size, self.num_heads * self.k_rank, bias=False)
        self.W_A_v = nn.Linear(self.hidden_size, self.num_heads * self.v_rank, bias=False)
        
        # W_B projections: hidden_size -> rank * head_dim
        self.W_B_q = nn.Linear(self.hidden_size, self.q_rank * self.head_dim, bias=False)
        self.W_B_k = nn.Linear(self.hidden_size, self.k_rank * self.head_dim, bias=False)
        self.W_B_v = nn.Linear(self.hidden_size, self.v_rank * self.head_dim, bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # Optional normalization
        self.query_norm = (
            RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if getattr(config, 'use_qk_norm', False)
            else None
        )
        self.key_norm = (
            RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if getattr(config, 'use_qk_norm', False)
            else None
        )
        
        # Softcapping parameters
        self.attn_logit_softcapping = getattr(config, 'attn_logit_softcapping', None)
        self.sliding_window_size = getattr(config, 'sliding_window_size', None)
        
        # Flag for using factorized weights
        self.use_factorized_weights = False
        
        # Initialize KV cache parameters with batch_size, seq_len, num_heads, rank/head_dim
        # These will be created as needed during inference
        self.cache_kA = None
        self.cache_kB = None
        self.cache_vA = None
        self.cache_vB = None

    def _init_kv_cache(self, batch_size, max_seq_len):
        """Initialize KV cache for TPA attention."""
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        # For TPA, we need to cache both A and B components for K and V matrices
        # This is a key difference from standard attention where we cache final K and V
        
        # Ensure that batch_size and max_seq_len are proper integers
        batch_size = int(batch_size)
        max_seq_len = int(max_seq_len)
        
        # Shape for A components: [batch_size, max_seq_len, num_heads, rank]
        # A components are of shape [hidden_dim, num_heads*rank] projected to [batch, seq_len, num_heads, rank]
        try:
            self.cache_kA = torch.zeros(
                (batch_size, max_seq_len, self.num_heads, self.k_rank),
                device=device, dtype=dtype
            )
            self.cache_vA = torch.zeros(
                (batch_size, max_seq_len, self.num_heads, self.v_rank),
                device=device, dtype=dtype
            )
            
            # Shape for B components: [batch_size, max_seq_len, rank, head_dim]
            # B components are of shape [hidden_dim, rank*head_dim] projected to [batch, seq_len, rank, head_dim]
            self.cache_kB = torch.zeros(
                (batch_size, max_seq_len, self.k_rank, self.head_dim),
                device=device, dtype=dtype
            )
            self.cache_vB = torch.zeros(
                (batch_size, max_seq_len, self.v_rank, self.head_dim),
                device=device, dtype=dtype
            )
        except RuntimeError as e:
            # Handle out-of-memory scenarios by reducing dimensions if possible
            print(f"Warning: Error creating KV cache with dimensions [batch={batch_size}, seq={max_seq_len}]. Error: {e}")
            print("Trying to create smaller cache...")
            
            # Reduce batch size if needed
            adjusted_batch_size = max(1, batch_size // 2)
            # Reduce sequence length if needed
            adjusted_seq_len = max(64, max_seq_len // 2)
            
            print(f"Creating cache with adjusted dimensions: batch={adjusted_batch_size}, seq={adjusted_seq_len}")
            
            self.cache_kA = torch.zeros(
                (adjusted_batch_size, adjusted_seq_len, self.num_heads, self.k_rank),
                device=device, dtype=dtype
            )
            self.cache_vA = torch.zeros(
                (adjusted_batch_size, adjusted_seq_len, self.num_heads, self.v_rank),
                device=device, dtype=dtype
            )
            
            self.cache_kB = torch.zeros(
                (adjusted_batch_size, adjusted_seq_len, self.k_rank, self.head_dim),
                device=device, dtype=dtype
            )
            self.cache_vB = torch.zeros(
                (adjusted_batch_size, adjusted_seq_len, self.v_rank, self.head_dim),
                device=device, dtype=dtype
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
        local_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # A projections
        A_q = self.W_A_q(hidden_states).view(batch_size, seq_len, self.num_heads, self.q_rank)
        A_k = self.W_A_k(hidden_states).view(batch_size, seq_len, self.num_heads, self.k_rank)
        A_v = self.W_A_v(hidden_states).view(batch_size, seq_len, self.num_heads, self.v_rank)
        
        # B projections
        B_q = self.W_B_q(hidden_states).reshape(batch_size, seq_len, self.q_rank, self.head_dim)
        B_k = self.W_B_k(hidden_states).reshape(batch_size, seq_len, self.k_rank, self.head_dim)
        B_v = self.W_B_v(hidden_states).reshape(batch_size, seq_len, self.v_rank, self.head_dim)
        
        # Apply rotary positional embedding to B_q and B_k
        
        # Define a TPA-specific RoPE application that works with our tensor format
        def apply_rotary_emb_to_B(x, freqs_cis):
            # x shape: [batch, seq_len, rank, head_dim]
            batch_size, seq_len, rank, head_dim = x.shape
            
            # We need to ensure head_dim is even (required for complex representation)
            if head_dim % 2 != 0:
                raise ValueError(f"Head dimension {head_dim} must be even for RoPE")
            
            # Process each rank separately to avoid dimension mixing
            output = torch.zeros_like(x)
            
            for r in range(rank):
                # Extract just this rank: [batch, seq_len, head_dim]
                x_r = x[:, :, r, :]
                
                # Apply standard RoPE
                x_complex = torch.view_as_complex(
                    torch.stack(torch.chunk(x_r.float(), 2, dim=-1), dim=-1)
                )
                
                x_rotated = torch.view_as_real(x_complex * freqs_cis)
                
                # Reshape back to [batch, seq_len, head_dim]
                x_rotated = torch.cat(torch.chunk(x_rotated, 2, dim=-1), dim=-2)
                x_rotated = x_rotated.reshape(batch_size, seq_len, head_dim)
                
                # Store the result in the original rank's position
                output[:, :, r, :] = x_rotated
            
            return output.type_as(x)
        
        # Apply RoPE to B tensors
        B_q = apply_rotary_emb_to_B(B_q, freqs_cis)
        B_k = apply_rotary_emb_to_B(B_k, freqs_cis)
        
        # Handle KV cache - create if it doesn't exist
        if self.cache_kA is None or batch_size > self.cache_kA.shape[0]:
            self._init_kv_cache(batch_size, kv_cache[0].shape[1])
        
        # Write to cache
        self.cache_kA = self.cache_kA.to(A_k.device, A_k.dtype)
        self.cache_vA = self.cache_vA.to(A_v.device, A_v.dtype)
        self.cache_kB = self.cache_kB.to(B_k.device, B_k.dtype)
        self.cache_vB = self.cache_vB.to(B_v.device, B_v.dtype)
        
        self.cache_kA[:batch_size].index_copy_(1, kv_write_indices, A_k)
        self.cache_vA[:batch_size].index_copy_(1, kv_write_indices, A_v)
        self.cache_kB[:batch_size].index_copy_(1, kv_write_indices, B_k)
        self.cache_vB[:batch_size].index_copy_(1, kv_write_indices, B_v)
        
        # Get the KV cache sections up to the current position
        cache_len = kv_write_indices[-1] + 1 if kv_write_indices.numel() > 0 else 0
        
        A_k = self.cache_kA[:batch_size, :cache_len]
        A_v = self.cache_vA[:batch_size, :cache_len]
        B_k = self.cache_kB[:batch_size, :cache_len]
        B_v = self.cache_vB[:batch_size, :cache_len]
        
        # Define kv_seq_len for reshaping
        kv_seq_len = A_k.size(1)
        
        # Reshape for tensor product computation
        A_q_flat = A_q.reshape(batch_size * seq_len, self.num_heads, self.q_rank)
        A_k_flat = A_k.reshape(batch_size * kv_seq_len, self.num_heads, self.k_rank)
        A_v_flat = A_v.reshape(batch_size * kv_seq_len, self.num_heads, self.v_rank)
        
        B_q_flat = B_q.reshape(batch_size * seq_len, self.q_rank, self.head_dim)
        B_k_flat = B_k.reshape(batch_size * kv_seq_len, self.k_rank, self.head_dim)
        B_v_flat = B_v.reshape(batch_size * kv_seq_len, self.v_rank, self.head_dim)
        
        # Compute Q, K, V using tensor product with factorization
        q = torch.bmm(A_q_flat, B_q_flat).div(self.q_rank)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        k = torch.bmm(A_k_flat, B_k_flat).div(self.k_rank)
        k = k.view(batch_size, kv_seq_len, self.num_heads, self.head_dim)
        
        v = torch.bmm(A_v_flat, B_v_flat).div(self.v_rank)
        v = v.view(batch_size, kv_seq_len, self.num_heads, self.head_dim)
        
        # Apply query/key normalization if needed
        if self.query_norm is not None and self.key_norm is not None:
            q = self.query_norm(q)
            k = self.key_norm(k)
            
        # Transpose Q, K, V for attention computation
        # [batch, seq, num_heads, head_dim] -> [batch, num_heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply attention scaling
        q = q * self.scaling
        
        # Compute attention scores
        # [batch, num_heads, seq, head_dim] x [batch, num_heads, head_dim, kv_seq]
        # -> [batch, num_heads, seq, kv_seq]
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        
        # Apply softcapping if configured
        if self.attn_logit_softcapping is not None:
            attn_weights = attn_weights / self.attn_logit_softcapping
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * self.attn_logit_softcapping
        
        # Apply sliding window mask if needed
        if (
            self.attn_type == gemma_config.AttentionType.LOCAL_SLIDING
            and self.sliding_window_size is not None
            and local_mask is not None
        ):
            mask = local_mask
        
        # Apply attention mask
        attn_weights = attn_weights + mask
        
        # Apply softmax
        attn_probs = F.softmax(attn_weights.float(), dim=-1).type_as(q)
        
        # Apply attention to values
        # [batch, num_heads, seq, kv_seq] x [batch, num_heads, kv_seq, head_dim]
        # -> [batch, num_heads, seq, head_dim]
        attn_output = torch.matmul(attn_probs, v)
        
        # Reshape to [batch, seq, num_heads*head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # Apply output projection
        output = self.o_proj(attn_output)
        
        return output


class Gemma3TPADecoderLayer(nn.Module):
    """Gemma3 decoder layer using TPA attention."""
    
    def __init__(
        self,
        config: gemma_config.GemmaConfig,
        attn_type: gemma_config.AttentionType,
    ):
        super().__init__()
        self.attn_type = attn_type
        self.self_attn = TPAAttention(
            config=config,
            attn_type=self.attn_type,
        )
        self.mlp = gemma_model.GemmaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant=config.quant,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if getattr(config, 'use_pre_ffw_norm', False)
            else None
        )
        self.post_feedforward_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if getattr(config, 'use_post_ffw_norm', False)
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
        local_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
            mask=mask,
            local_mask=local_mask,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        if self.pre_feedforward_layernorm is not None:
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.post_feedforward_layernorm is not None:
            hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Gemma3TPAModel(nn.Module):
    """Gemma3 model with TPA attention."""
    
    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            if config.architecture in (
                gemma_config.Architecture.GEMMA_2,
                gemma_config.Architecture.GEMMA_3,
            ):
                attn_type = (
                    config.attn_types[i % len(config.attn_types)]
                    if config.attn_types is not None
                    else gemma_config.AttentionType.GLOBAL
                )
                self.layers.append(Gemma3TPADecoderLayer(config, attn_type))
            else:
                raise ValueError(f'Unsupported architecture: {config.architecture}')
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: Mapping[gemma_config.AttentionType, torch.Tensor],
        kv_write_indices: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
        local_mask: torch.Tensor,
    ) -> torch.Tensor:
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis.get(layer.attn_type),
                kv_write_indices=kv_write_indices,
                kv_cache=kv_caches[i],
                mask=mask,
                local_mask=local_mask,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Gemma3ForMultimodalLMwithTPA(nn.Module):
    """Gemma3 model for multimodal causal LM with TPA attention."""
    def __init__(
        self,
        config: gemma_config.GemmaConfig,
    ):
        super().__init__()
        self.dtype = config.get_dtype()
        assert config.architecture == gemma_config.Architecture.GEMMA_3
        self.config = config
        max_seq_len = config.max_position_embeddings
        head_dim = config.head_dim
        vocab_size = config.vocab_size
        self.tokenizer = tokenizer.Tokenizer(config.tokenizer)
        self.text_token_embedder = gemma_model.Embedding(vocab_size, config.hidden_size, config.quant)
        
        # Use our TPA model instead of standard GemmaModel
        self.model = Gemma3TPAModel(config)
        
        self.sampler = gemma_model.Sampler(vocab_size, config)

        if config.vision_config is None:
            print("Creating text-only TPA model - no vision embedding")
        else:
            # Vision embeddings setup for multimodal model
            self.siglip_vision_model = siglip_vision_model.SiglipVisionModel(config.vision_config)
            # transformer/embedder/mm_soft_embedding_norm
            self.mm_soft_embedding_norm = RMSNorm(config.vision_config.embedding_dim,
                                                  eps=config.rms_norm_eps)
            # transformer/embedder/mm_input_projection
            self.mm_input_projection = gemma_model.Linear(config.vision_config.embedding_dim, 
                                                        config.hidden_size, config.quant)

        if config.rope_wave_length is None:
            raise ValueError('rope_wave_length must be provided for Gemma3.')
        rope_lengths = config.rope_wave_length
        defaults = {
                gemma_config.AttentionType.LOCAL_SLIDING: 10_000,
                gemma_config.AttentionType.GLOBAL: 10_000,
            }
        self._register_freqs_cis('local_freqs_cis', head_dim, max_seq_len, theta=rope_lengths.get(
                  gemma_config.AttentionType.LOCAL_SLIDING, defaults[gemma_config.AttentionType.LOCAL_SLIDING]
              ))
        self._register_freqs_cis('global_freqs_cis', head_dim, max_seq_len, theta=rope_lengths.get(
                  gemma_config.AttentionType.GLOBAL, defaults[gemma_config.AttentionType.GLOBAL]
              ), rope_scaling_factor=config.rope_scaling_factor)

    def _register_freqs_cis(
        self, name: str, head_dim: int, max_seq_len: int, theta: int = 10_000, rope_scaling_factor: int = 1
    ):
        self.register_buffer(
            name, gemma_model.precompute_freqs_cis(head_dim, max_seq_len * 2, theta=theta, rope_scaling_factor=rope_scaling_factor)
        )

    @torch.no_grad()
    def forward(self,
              input_token_ids: torch.Tensor, # B x L
              image_patches: torch.Tensor = None, # B x N x C x H x W (3x896x896)
              image_presence_mask: torch.Tensor = None, # B x N
              input_positions: torch.Tensor = None,
              kv_caches: List[Tuple[torch.Tensor, torch.Tensor]] = None,
              mask: torch.Tensor = None,
              output_positions: torch.Tensor = None,
              temperatures: Union[torch.Tensor, None] = None,
              top_ps: torch.Tensor = None,
              top_ks: torch.Tensor = None,
              local_mask: torch.Tensor | None = None,
              **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
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
        
        if image_patches is not None and hasattr(self, 'siglip_vision_model'):
            # If we have vision capabilities and image input, process it
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
        # Step 1 of 2: Fetch valid image embeddings
        # flatten indices of valid image embeddings
        valid_image_embeddings_indices = torch.nonzero(image_presence_mask.flatten(), as_tuple=False).squeeze()
        # num_valid_images x model_dim
        valid_image_embeddings = image_embeddings.index_select(0, valid_image_embeddings_indices)

        # Step 2 of 2: Replace image embeddings at right places.
        image_placeholder_mask = input_token_ids == self.tokenizer.image_token_placeholder_id
        image_placeholder_indices = image_placeholder_mask.flatten().nonzero(as_tuple=False).squeeze()

        hidden_states = hidden_states.reshape(-1, self.config.hidden_size)
        hidden_states[image_placeholder_indices] = valid_image_embeddings.reshape(-1, self.config.hidden_size)
        return hidden_states.reshape(batch_size, seq_len, model_dim).contiguous()

    def create_attention_mask(self, input_ids: torch.Tensor, sequence_length: int):
        batch_size = input_ids.shape[0]
        causal_mask = torch.tril(torch.ones((batch_size, 1, sequence_length, sequence_length), dtype=torch.bool, device=input_ids.device))
        image_token_mask = input_ids == self.tokenizer.image_token_placeholder_id
        # Pad the mask to the left with 0. This is to make sure the boundary
        # detection works correctly. Boundary (starting index of image patch) is
        # detected when the value changes from 0 to 1.
        padded_mask = nn.functional.pad(image_token_mask, (1, 0), value=0)
        # Find the boundary (starting index) of the image tokens patch.
        boundary = padded_mask[:, 1:] > padded_mask[:, :-1]
        # Number the boundary.
        # boundary:
        # [[False, False,  True, False, False],
        #  [False,  True, False,  True, False]]
        # numbered_boundary:
        # [[0, 0, 1, 1, 1],
        #  [0, 1, 1, 2, 2]]
        numbered_boundary = torch.cumsum(boundary, dim=-1)

        # image_token_mask:
        # [[False, False,  True,  True, False],
        #  [True,  True, False,  True, True]]
        # numbered_boundary:
        # [[0, 0, 1, 1, 1],
        #  [1, 1, 1, 2, 2]]
        # q_block_indices:
        # [[0, 0, 1, 1, 0],
        #  [1, 1, 0, 2, 2]]
        q_block_indices = image_token_mask * numbered_boundary
        kv_block_indices = q_block_indices
        # Test the equality of vertical and horizontal numbered patches
        # to create the bidirectional mask.
        bidirectional_mask = torch.logical_and(
            kv_block_indices[:, None, :] == q_block_indices.unsqueeze(-1),
            q_block_indices.unsqueeze(-1) > 0,
        )
        attention_mask = torch.logical_or(causal_mask, bidirectional_mask.unsqueeze(1))
        # The upper triangular matrix's diagonal is shifted by sliding window size
        # before doing logical 'and' with attention mask. This is to make sure the
        # local attention is within the sliding window.
        local_mask = torch.logical_and(
                attention_mask,
                torch.triu(torch.ones((1, 1, sequence_length, sequence_length), dtype=torch.bool, device=input_ids.device), diagonal=-(self.config.sliding_window_size-1))
            )
        return attention_mask, local_mask

    def generate(
        self,
        prompts: Sequence[Union[str, Sequence[Union[str, Image.Image]]]],
        device: Any = None,
        max_tokens: int = 100,
        temperature: Union[float, None] = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
    ) -> Sequence[str]:
        """Generates responses for given prompts using Gemma model."""
        # For backward compatibility, support both output_len and max_tokens
        output_len = max_tokens
        
        # Determine if device is provided, otherwise use model device
        if device is None:
            device = next(self.parameters()).device
        
        # Handle different prompt formats
        if isinstance(prompts, str):
            prompts = [prompts]
        elif isinstance(prompts, list) and all(isinstance(p, str) for p in prompts):
            pass
        else:
            # Assume we have a sequence of sequences (for multimodal input)
            prompts = [[p] if isinstance(p, (str, Image.Image)) else p for p in prompts]
        
        # For text-only models, convert string prompts to expected format
        if not hasattr(self, 'siglip_vision_model'):
            prompts = [[p] if isinstance(p, str) else p for p in prompts]
        
        # Process input using gemma3_preprocessor
        processing_result = gemma3_preprocessor.tokenize_raw_input(
            self.tokenizer, prompts, self.config, output_len, device
        )
        
        batch_size = processing_result["batch_size"]
        user_input_token_ids = processing_result["user_input_token_ids"]
        image_batch = processing_result.get("image_batch")
        image_presence_mask = processing_result.get("image_presence_mask")
        min_prompt_len = processing_result["min_prompt_len"]
        max_prompt_len = processing_result["max_prompt_len"]
        total_seq_len = processing_result["max_seq_len"]

        # Create attention mask
        min_dtype = torch.finfo(self.dtype).min
        if self.config.sliding_window_size is None:
            raise ValueError('gemma 3 model requires sliding_window size')
        boolean_mask, local_boolean_mask = self.create_attention_mask(user_input_token_ids, total_seq_len)
        mask_tensor = torch.where(boolean_mask, 0, torch.tensor(min_dtype, dtype=torch.float32, device=device)).contiguous()
        local_mask_tensor = torch.where(local_boolean_mask, 0, torch.tensor(min_dtype, dtype=torch.float32, device=device)).contiguous()

        # Build KV caches
        kv_caches = []
        for _ in range(self.config.num_hidden_layers):
            size = (batch_size, total_seq_len, self.config.num_attention_heads,
                    self.config.head_dim)
            dtype = self.config.get_dtype()
            k_cache = torch.zeros(size=size, dtype=dtype, device=device)
            v_cache = torch.zeros(size=size, dtype=dtype, device=device)
            kv_caches.append((k_cache, v_cache))

        # Prepare input tensors
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
        curr_local_mask_tensor = local_mask_tensor.index_select(2, input_positions_tensor)
        output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(device)
        temperatures_tensor = None if not temperature else torch.FloatTensor(
                [temperature] * batch_size).to(device)
        top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
        top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)
        output_index = torch.tensor(min_prompt_len, dtype=torch.int64, device=device)

        # Generate tokens
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
            curr_local_mask_tensor = local_mask_tensor.index_select(
                2, input_positions_tensor
            ) if local_mask_tensor is not None else None
            output_positions_tensor = torch.tensor(0, dtype=torch.int64, device=device)
            output_index = output_index + 1
            # We only need image data for the first token generation
            image_batch = None
            image_presence_mask = None

        # Detokenize output
        token_ids = token_ids_tensor.tolist()
        results = []
        for i, tokens in enumerate(token_ids):
            output = tokens
            if self.tokenizer.eos_id in output:
                eos_index = output.index(self.tokenizer.eos_id)
                output = output[:eos_index]
            results.append(self.tokenizer.decode(output))

        # Return single string if single prompt was provided
        return results[0] if isinstance(prompts, str) else results

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