"""
Utility functions for tensor product operations and model handling.

This module contains functions that support Tensor Product Attention (TPA)
implementations, including attention mask creation, position embedding generation,
and other helper functions.
"""

import torch
import math
from torch import nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any, Mapping


def register_freqs_cis(model, name: str, head_dim: int, max_seq_len: int, theta: int = 10_000, rope_scaling_factor: int = 1):
    """
    Create and register frequency tensors for rotary position embeddings.
    
    Args:
        model: The model to register the buffer on
        name: Name for the buffer
        head_dim: Dimension of attention heads
        max_seq_len: Maximum sequence length
        theta: Base frequency parameter
        rope_scaling_factor: Scaling factor for RoPE frequencies
    """
    # Handle None value for rope_scaling_factor
    if rope_scaling_factor is None:
        rope_scaling_factor = 1
        
    # Create freqs
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[:(head_dim // 2)].float() / head_dim))
    
    # Create position indices
    t = torch.arange(max_seq_len, device=freqs.device)
    t = t * rope_scaling_factor  # Apply scaling if needed
    
    # Create embedding
    freqs = torch.outer(t, freqs)
    
    # Create complex rotation matrix
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    
    # Register as buffer
    model.register_buffer(name, freqs_cis)


def create_attention_mask(
    input_tokens: torch.Tensor,
    max_seq_len: int, 
    sliding_window: Optional[int] = None,
    is_causal: bool = True,
    image_tokens_indices: Optional[torch.Tensor] = None,
    attention_scale: float = 1.0
) -> torch.Tensor:
    """
    Create an attention mask for transformer models including support for 
    sliding window attention and multimodal (image token) handling.
    
    Args:
        input_tokens: Input token IDs tensor [batch_size, seq_len]
        max_seq_len: Maximum sequence length
        sliding_window: Size of the sliding window for attention (optional)
        is_causal: Whether to create a causal mask
        image_tokens_indices: Indices of image tokens (optional)
        attention_scale: Scale factor for attention scores
        
    Returns:
        torch.Tensor: Attention mask
    """
    batch_size, seq_len = input_tokens.shape
    device = input_tokens.device
    
    # Create base causal mask if needed
    if is_causal:
        # Create mask filled with negative infinity to suppress invalid attention
        mask = torch.full((max_seq_len, max_seq_len), 
                          float("-inf"), 
                          device=device)
        
        # Fill the lower triangular part with zeros (allow attention)
        mask = torch.triu(mask, diagonal=1)
        
        # Reshape for broadcasting across batch and heads
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, max_seq_len, max_seq_len]
    else:
        # Non-causal mask - all positions can attend to each other
        mask = torch.zeros((max_seq_len, max_seq_len), device=device)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, max_seq_len, max_seq_len]
    
    # Apply sliding window if specified
    if sliding_window is not None and sliding_window > 0:
        # Create the sliding window limitation mask
        window_mask = torch.ones((max_seq_len, max_seq_len), 
                                device=device) * float("-inf")
        
        # For each position i, allow attention to positions [i-sliding_window, i]
        for i in range(max_seq_len):
            start = max(0, i - sliding_window + 1)
            window_mask[i, start:i+1] = 0
            
        # Combine with existing mask
        window_mask = window_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, max_seq_len, max_seq_len]
        mask = mask + window_mask
    
    # Special handling for image tokens if provided
    if image_tokens_indices is not None:
        # Create image token mask
        for batch_idx in range(batch_size):
            # Get image token indices for this batch
            img_indices = image_tokens_indices[batch_idx]
            
            # Handle bidirectional attention for image tokens
            for img_pos in img_indices:
                if img_pos >= 0:  # Skip padding positions
                    # Image tokens can see all tokens and be seen by all tokens
                    # Adjust existing mask to allow this
                    mask[0, 0, :img_pos+1, img_pos] = 0  # All tokens can attend to image token
                    mask[0, 0, img_pos, :img_pos+1] = 0  # Image token can attend to all previous tokens
    
    # Scale attention scores if needed
    if attention_scale != 1.0:
        # Only scale the non-masked values
        mask = torch.where(
            mask == 0, 
            torch.tensor(0.0, device=device), 
            mask * attention_scale
        )
    
    return mask


def populate_image_embeddings(
    hidden_states: torch.Tensor,
    image_embeddings: torch.Tensor,
    image_token_indices: torch.Tensor
) -> torch.Tensor:
    """
    Insert image embeddings into the hidden states at specified positions.
    
    Args:
        hidden_states: Model hidden states [batch_size, seq_len, hidden_dim]
        image_embeddings: Embeddings for images [batch_size, num_images, hidden_dim]
        image_token_indices: Indices where to insert image tokens [batch_size, num_images]
        
    Returns:
        torch.Tensor: Hidden states with image embeddings populated
    """
    batch_size, seq_len, hidden_dim = hidden_states.shape
    _, num_images, _ = image_embeddings.shape
    
    # Create a copy of hidden states to modify
    result = hidden_states.clone()
    
    # For each batch
    for b in range(batch_size):
        # For each image
        for i in range(num_images):
            # Get position where to insert this image embedding
            pos = image_token_indices[b, i].item()
            
            # Skip if the position is invalid (e.g., -1 for padding)
            if pos < 0 or pos >= seq_len:
                continue
                
            # Insert the image embedding at the specified position
            result[b, pos] = image_embeddings[b, i]
    
    return result


def reshape_for_broadcast(tensor: torch.Tensor, target_shape: List[int]) -> torch.Tensor:
    """
    Reshape a tensor for broadcasting against another tensor.
    
    Args:
        tensor: Input tensor to reshape
        target_shape: Target shape to broadcast against
        
    Returns:
        torch.Tensor: Reshaped tensor suitable for broadcasting
    """
    # Get the current shape
    tensor_shape = tensor.shape
    
    # Add dimension(s) to match target tensor shape
    while len(tensor_shape) < len(target_shape):
        tensor = tensor.unsqueeze(0)
        tensor_shape = tensor.shape
    
    # Expand dimensions to enable broadcasting
    for dim, (src, tgt) in enumerate(zip(tensor_shape, target_shape)):
        if src == 1 and tgt > 1:
            tensor = tensor.expand(target_shape)
            break
            
    return tensor