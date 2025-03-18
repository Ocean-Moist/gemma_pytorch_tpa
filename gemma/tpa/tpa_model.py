"""TPA-based Gemma model implementations."""

import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Mapping, Union

from .. import config as gemma_config
from .. import model as gemma_model
from .tpa_attention import GemmaTensorProductAttention

class GemmaTPADecoderLayer(nn.Module):
    """Gemma decoder layer using Tensor Product Attention."""

    def __init__(
        self,
        config: gemma_config.GemmaConfig,
        attn_type: gemma_config.AttentionType,
    ):
        super().__init__()
        self.attn_type = attn_type
        self.self_attn = GemmaTensorProductAttention(
            config=config,
            attn_type=self.attn_type,
        )
        self.mlp = gemma_model.GemmaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant=config.quant,
        )
        self.input_layernorm = gemma_model.RMSNorm(config.hidden_size,
                                                  eps=config.rms_norm_eps)
        self.post_attention_layernorm = gemma_model.RMSNorm(config.hidden_size,
                                                           eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = (
            gemma_model.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_pre_ffw_norm
            else None
        )
        self.post_feedforward_layernorm = (
            gemma_model.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_post_ffw_norm
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
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

class GemmaTPAModel(nn.Module):
    """Gemma model using Tensor Product Attention."""

    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            attn_type = (
                config.attn_types[i % len(config.attn_types)]
                if config.attn_types is not None
                else gemma_config.AttentionType.GLOBAL
            )
            self.layers.append(GemmaTPADecoderLayer(config, attn_type))
            
        self.norm = gemma_model.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: Mapping[gemma_config.AttentionType, torch.Tensor],
        kv_write_indices: torch.Tensor,
        kv_caches: List[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]],
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

def create_tpa_kv_caches(config: gemma_config.GemmaConfig, batch_size: int, max_seq_len: int, device: torch.device) -> List[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """
    Create TPA-based KV caches for Gemma models.
    
    Args:
        config: Gemma configuration
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        device: Device to create tensors on
        
    Returns:
        List of tuples containing factorized KV caches for each layer
    """
    k_rank = getattr(config, "k_rank", 2)  # Default to 2 as in the TPA paper
    v_rank = getattr(config, "v_rank", 2)  # Default to 2 as in the TPA paper
    
    # Ensure we have valid dimensions
    if batch_size <= 0:
        batch_size = 1
        print(f"Warning: Invalid batch_size {batch_size}, using 1 instead")
    
    if max_seq_len <= 0:
        max_seq_len = 1
        print(f"Warning: Invalid max_seq_len {max_seq_len}, using 1 instead")
        
    # Ensure we have valid ranks
    if k_rank <= 0:
        k_rank = 1
        print(f"Warning: Invalid k_rank {k_rank}, using 1 instead")
        
    if v_rank <= 0:
        v_rank = 1
        print(f"Warning: Invalid v_rank {v_rank}, using 1 instead")
    
    # Ensure num_key_value_heads is set
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    
    print(f"Creating TPA KV caches with dimensions: batch_size={batch_size}, max_seq_len={max_seq_len}, "
          f"num_kv_heads={num_kv_heads}, k_rank={k_rank}, v_rank={v_rank}, head_dim={head_dim}")
    
    kv_caches = []
    for i in range(config.num_hidden_layers):
        # Create separate caches for A and B factors
        try:
            k_cache_A = torch.zeros(
                size=(batch_size, max_seq_len, num_kv_heads, k_rank),
                dtype=config.get_dtype(),
                device=device
            )
            k_cache_B = torch.zeros(
                size=(batch_size, max_seq_len, k_rank, head_dim),
                dtype=config.get_dtype(),
                device=device
            )
            v_cache_A = torch.zeros(
                size=(batch_size, max_seq_len, num_kv_heads, v_rank),
                dtype=config.get_dtype(),
                device=device
            )
            v_cache_B = torch.zeros(
                size=(batch_size, max_seq_len, v_rank, head_dim),
                dtype=config.get_dtype(),
                device=device
            )
            
            kv_caches.append((k_cache_A, k_cache_B, v_cache_A, v_cache_B))
        except Exception as e:
            print(f"Error creating KV cache for layer {i}: {e}")
            # Create minimal cache as fallback
            k_cache_A = torch.zeros((1, 1, 1, 1), dtype=torch.float32, device=device)
            k_cache_B = torch.zeros((1, 1, 1, 1), dtype=torch.float32, device=device)
            v_cache_A = torch.zeros((1, 1, 1, 1), dtype=torch.float32, device=device)
            v_cache_B = torch.zeros((1, 1, 1, 1), dtype=torch.float32, device=device)
            kv_caches.append((k_cache_A, k_cache_B, v_cache_A, v_cache_B))
    
    return kv_caches