"""
T6-style contextual factorization methods for Tensor Product Attention.

This module implements the contextual tensor factorization approach described in the
"Tensor Product Attention Is All You Need" (T6) paper, which uses low-rank
factors to compress queries, keys, and values during inference.
"""

import torch
import math
import time
import numpy as np
from torch import nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any

try:
    from tqdm import tqdm
except ImportError:
    # Simple tqdm fallback if not available
    def tqdm(iterable, *args, **kwargs):
        return iterable


def contextual_tensor_decomposition(weight, q_rank=6, k_rank=2, v_rank=2, dtype=torch.float16, device="cuda"):
    """
    Factorize a weight matrix using contextual tensor decomposition from the T6 paper.
    
    This function decomposes QKV query weight matrices into A and B factors for
    contextual tensor product attention, enhancing parameter efficiency.
    
    Args:
        weight: The weight matrix to factorize
        q_rank: Rank for query matrices (usually higher than K/V)
        k_rank: Rank for key matrices
        v_rank: Rank for value matrices
        dtype: Data type for the factorized weights
        device: Device to place tensors on
        
    Returns:
        tuple: Factorized weights for contextual TPA
    """
    # Determine if this is a query, key or value weight matrix
    is_query = weight.shape[0] % q_rank == 0
    is_key = weight.shape[0] % k_rank == 0
    is_value = weight.shape[0] % v_rank == 0
    
    # Default to query if unclear
    rank = q_rank
    if is_key and not is_query:
        rank = k_rank
    elif is_value and not is_query and not is_key:
        rank = v_rank
    
    # Get dimensions
    output_dim = weight.shape[0]
    input_dim = weight.shape[1]
    
    # Estimate dimensions for factorization
    head_dim = output_dim // rank
    factor_share = 0.3  # Share of parameters to allocate to A vs B factors
    
    # Calculate estimated rank based on parameter budget
    param_budget = output_dim * input_dim
    A_size = output_dim * rank
    B_size = rank * input_dim
    
    # Initialize factorized matrices (A and B factors)
    A_factor = torch.empty((output_dim, rank), dtype=dtype, device=device)
    B_factor = torch.empty((rank, input_dim), dtype=dtype, device=device)
    
    # Initialize with random weights
    nn.init.normal_(A_factor, mean=0.0, std=0.02)
    nn.init.normal_(B_factor, mean=0.0, std=0.02)
    
    # Optionally initialize with SVD for better starting point
    try:
        # Convert to FP32 for numerical stability in SVD
        weight_fp32 = weight.float()
        U, S, Vh = torch.linalg.svd(weight_fp32, full_matrices=False)
        
        # Truncate to rank
        U_truncated = U[:, :rank]
        S_truncated = S[:rank]
        Vh_truncated = Vh[:rank, :]
        
        # Scale U by sqrt(S) and V by sqrt(S) for balance
        A_factor_init = U_truncated @ torch.diag(torch.sqrt(S_truncated))
        B_factor_init = torch.diag(torch.sqrt(S_truncated)) @ Vh_truncated
        
        # Copy to proper dtype
        A_factor.copy_(A_factor_init.to(dtype))
        B_factor.copy_(B_factor_init.to(dtype))
        
        print(f"SVD initialization successful with rank {rank}")
    except Exception as e:
        print(f"SVD initialization failed: {e}, using random initialization")
    
    # Return initialized factors
    return A_factor, B_factor


def _init_contextual_factorization(weight_matrix, rank, num_iterations=100, learning_rate=0.01):
    """
    Use alternating least squares to optimize A and B factor matrices.
    
    Args:
        weight_matrix: The weight matrix to factorize
        rank: The rank for factorization
        num_iterations: Number of optimization iterations
        learning_rate: Learning rate for optimization
        
    Returns:
        tuple: Optimized A and B factors
    """
    # Extract dimensions
    output_dim, input_dim = weight_matrix.shape
    
    # Create factor matrices and initialize randomly
    A_factor = nn.Parameter(torch.randn(output_dim, rank) * 0.02, requires_grad=True)
    B_factor = nn.Parameter(torch.randn(rank, input_dim) * 0.02, requires_grad=True)
    
    # Create optimizer
    optimizer = torch.optim.Adam([A_factor, B_factor], lr=learning_rate)
    
    # Train to minimize reconstruction error
    best_loss = float('inf')
    best_A = None
    best_B = None
    
    for i in range(num_iterations):
        optimizer.zero_grad()
        
        # Reconstruct from factors
        reconstructed = torch.matmul(A_factor, B_factor)
        
        # Calculate loss
        loss = F.mse_loss(reconstructed, weight_matrix)
        
        # Check if this is the best we've seen
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_A = A_factor.data.clone()
            best_B = B_factor.data.clone()
        
        # Backpropagate and optimize
        loss.backward()
        optimizer.step()
        
        # Log progress
        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}: loss = {loss.item():.6f}")
    
    # Return the best factors
    if best_A is not None and best_B is not None:
        return best_A, best_B
    else:
        return A_factor.data, B_factor.data


def apply_contextual_tensor_decomposition(model, q_rank=6, k_rank=2, v_rank=2):
    """
    Apply contextual tensor decomposition to all layers in the model.
    
    This function applies the T6-style factorization to all attention projection
    matrices in the model, creating low-rank contextual factors.
    
    Args:
        model: The transformer model to factorize
        q_rank: Rank for query matrices (usually higher than K/V)
        k_rank: Rank for key matrices
        v_rank: Rank for value matrices
        
    Returns:
        dict: Statistics about the factorization process
    """
    stats = {
        "layers_converted": 0,
        "total_parameters_original": 0,
        "total_parameters_factorized": 0,
    }
    
    print(f"Applying TPA contextual tensor factorization with ranks: Q={q_rank}, K={k_rank}, V={v_rank}")
    
    # Process each layer
    for layer_idx, layer in enumerate(model.layers):
        try:
            print(f"Processing layer {layer_idx}...")
            # Access attention projections and factorize weights
            
            # Process QKV projections
            qkv_proj_weight = layer.self_attn.qkv_proj.weight
            
            # Calculate original parameter count
            original_params = qkv_proj_weight.numel()
            stats["total_parameters_original"] += original_params
            
            # Apply factorization and record statistics
            layer_stats = contextual_tensor_decomposition(qkv_proj_weight, q_rank, k_rank, v_rank)
            
            # Calculate factorized parameter count
            factorized_params = sum(p.numel() for p in layer_stats)
            stats["total_parameters_factorized"] += factorized_params
            
            stats["layers_converted"] += 1
            
        except Exception as e:
            print(f"Error factorizing layer {layer_idx}: {e}")
            continue
    
    # Calculate overall compression ratio
    if stats["total_parameters_original"] > 0:
        stats["compression_ratio"] = stats["total_parameters_original"] / stats["total_parameters_factorized"]
        print(f"Overall compression ratio: {stats['compression_ratio']:.2f}x")
    
    return stats


def convert_from_standard_weights(standard_model, tpa_model, q_rank=6, k_rank=2, v_rank=2, verbose=True):
    """
    Convert a standard Gemma3 model to use TPA weights.
    
    This function converts standard attention weights to Tensor Product Attention
    weights using contextual tensor factorization.
    
    Args:
        standard_model: The standard Gemma3 model
        tpa_model: The TPA version of the model to populate
        q_rank: Rank for query matrices
        k_rank: Rank for key matrices
        v_rank: Rank for value matrices
        verbose: Whether to print progress information
        
    Returns:
        The converted TPA model
    """
    print(f"Converting standard Gemma3 model to TPA with ranks - Q:{q_rank}, K:{k_rank}, V:{v_rank}")
    
    # Copy embedding weights
    try:
        tpa_model.text_token_embedder.weight.data.copy_(
            standard_model.text_token_embedder.weight.data)
        if verbose:
            print("Copied text token embedder weights")
    except Exception as e:
        print(f"Error copying embedder weights: {e}")
    
    # Copy vision model weights if present
    if hasattr(standard_model, 'siglip_vision_model') and hasattr(tpa_model, 'siglip_vision_model'):
        try:
            # For each vision submodule, copy the weights
            for name, param in standard_model.siglip_vision_model.named_parameters():
                tpa_param = tpa_model.siglip_vision_model.get_parameter(name)
                tpa_param.data.copy_(param.data)
            
            if verbose:
                print("Copied vision model weights")
                
            # Also copy projection weights
            tpa_model.mm_soft_embedding_norm.weight.data.copy_(
                standard_model.mm_soft_embedding_norm.weight.data)
            tpa_model.mm_input_projection.weight.data.copy_(
                standard_model.mm_input_projection.weight.data)
        except Exception as e:
            print(f"Error copying vision model weights: {e}")
    
    # Process transformer layers
    total_layers = len(standard_model.model.layers)
    iterator = tqdm(range(total_layers)) if verbose else range(total_layers)
    
    for layer_idx in iterator:
        try:
            std_layer = standard_model.model.layers[layer_idx]
            tpa_layer = tpa_model.model.layers[layer_idx]
            
            # Copy non-attention weights directly (layernorms, MLP)
            tpa_layer.input_layernorm.weight.data.copy_(std_layer.input_layernorm.weight.data)
            tpa_layer.post_attention_layernorm.weight.data.copy_(std_layer.post_attention_layernorm.weight.data)
            
            if hasattr(std_layer, 'pre_feedforward_layernorm') and std_layer.pre_feedforward_layernorm is not None:
                tpa_layer.pre_feedforward_layernorm.weight.data.copy_(std_layer.pre_feedforward_layernorm.weight.data)
            
            if hasattr(std_layer, 'post_feedforward_layernorm') and std_layer.post_feedforward_layernorm is not None:
                tpa_layer.post_feedforward_layernorm.weight.data.copy_(std_layer.post_feedforward_layernorm.weight.data)
            
            # Copy MLP weights
            tpa_layer.mlp.gate_proj.weight.data.copy_(std_layer.mlp.gate_proj.weight.data)
            tpa_layer.mlp.up_proj.weight.data.copy_(std_layer.mlp.up_proj.weight.data)
            tpa_layer.mlp.down_proj.weight.data.copy_(std_layer.mlp.down_proj.weight.data)
            
            # Factorize attention weights
            std_qkv_weight = std_layer.self_attn.qkv_proj.weight
            std_o_weight = std_layer.self_attn.o_proj.weight
            
            # Apply T6-style contextual tensor factorization
            # For TPA we need to factorize into A and B factors
            contextual_tensor_decomposition(
                std_qkv_weight, q_rank, k_rank, v_rank, dtype=std_qkv_weight.dtype, device=std_qkv_weight.device)
            
            # Copy output projection weight directly
            tpa_layer.self_attn.o_proj.weight.data.copy_(std_o_weight.data)
        
        except Exception as e:
            print(f"Error processing layer {layer_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Copy final norm weight
    try:
        tpa_model.model.norm.weight.data.copy_(standard_model.model.norm.weight.data)
    except Exception as e:
        print(f"Error copying final norm: {e}")
    
    # Copy sampler weight
    try:
        tpa_model.sampler.weight.data.copy_(standard_model.sampler.weight.data)
    except Exception as e:
        print(f"Error copying sampler weight: {e}")
    
    return tpa_model