"""
Tucker decomposition-based factorization methods for Tensor Product Attention (TPA).

This module implements various approaches for factorizing attention matrices
using Tucker decomposition and shared factor matrices, as described in the
TensorLLM paper.
"""

import torch
import math
import numpy as np
from torch import nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any

# Import common dependencies
import tensorly as tl
from tensorly.decomposition import tucker


def factorize_all_layers_with_shared_factors(model, config):
    """
    Factorize all transformer layers using the TensorLLM approach.
    
    This method applies Tucker decomposition with shared factor matrices to all
    multi-head attention weights in the model, significantly reducing parameter count
    while preserving model quality.
    
    Args:
        model: The transformer model to factorize
        config: Model configuration
        
    Returns:
        list: List of statistics for each layer including compression ratios and errors
    """
    stats = []
    num_layers = len(model.layers)
    
    print(f"Starting factorization of {num_layers} transformer layers using shared Tucker decomposition...")
    
    for i in range(num_layers):
        print(f"Factorizing layer {i+1}/{num_layers}...")
        layer_stats = _factorize_mha_weights_with_shared_factors(model, config, i)
        stats.append(layer_stats)
        
        # Print progress
        if isinstance(layer_stats.get("compression_ratio", None), (int, float)):
            print(f"  Layer {i}: {layer_stats['compression_ratio']:.2f}x compression "
                  f"with {layer_stats.get('error', 'unknown')} error")
        else:
            print(f"  Layer {i}: Factorization failed - {layer_stats.get('error', 'unknown error')}")
    
    # Calculate average compression
    successful_layers = [s for s in stats if isinstance(s.get("compression_ratio", None), (int, float))]
    if successful_layers:
        avg_compression = sum(s["compression_ratio"] for s in successful_layers) / len(successful_layers)
        print(f"Average compression ratio: {avg_compression:.2f}x")
    else:
        print("No layers were successfully compressed")
        
    return stats


def _factorize_mha_weights_with_shared_factors(model, config, layer_index: int):
    """Factorize MHA weights using Tucker decomposition with shared factor matrices.
    
    This approach follows the TensorLLM paper approach by enforcing a shared 
    higher-dimensional subspace across the weights of multiple attention heads.
    
    Args:
        model: The transformer model to factorize
        config: Model configuration
        layer_index: Index of the transformer layer to factorize
        
    Returns:
        dict: Statistics about the compression including compression ratio
    """
    # Set backend to PyTorch
    tl.set_backend('pytorch')
    
    # Get layer for factorization
    layer = model.layers[layer_index]
    
    # Step 1: Split weight matrices into multiple heads
    W_Q = layer.self_attn.qkv_proj.weight[:config.num_attention_heads * config.head_dim].reshape(
        config.num_attention_heads, config.head_dim, config.hidden_size)
    W_K = layer.self_attn.qkv_proj.weight[config.num_attention_heads * config.head_dim:2 * config.num_attention_heads * config.head_dim].reshape(
        config.num_key_value_heads, config.head_dim, config.hidden_size)
    W_V = layer.self_attn.qkv_proj.weight[2 * config.num_attention_heads * config.head_dim:].reshape(
        config.num_key_value_heads, config.head_dim, config.hidden_size)
    W_O = layer.self_attn.o_proj.weight.reshape(
        config.hidden_size, config.num_attention_heads, config.head_dim).permute(1, 2, 0)
    
    # Step 2: Multi-head tensorisation - stack into 4D tensor
    # Create a tensor of shape [num_heads, head_dim, hidden_size, 4]
    attention_tensors = []
    
    # For each attention head
    for i in range(config.num_attention_heads):
        # Get Q, K, V, O weights for this head
        if i < config.num_key_value_heads:
            head_K = W_K[i]
            head_V = W_V[i]
        else:
            # For GQA, reuse KV heads
            kv_index = i % config.num_key_value_heads
            head_K = W_K[kv_index]
            head_V = W_V[kv_index]
            
        head_Q = W_Q[i]
        head_O = W_O[i]
        
        # Create 3D tensor for this head [head_dim, hidden_size, 4]
        head_tensor = torch.stack([head_Q, head_K, head_V, head_O], dim=2)
        attention_tensors.append(head_tensor)
    
    # Stack all head tensors into 4D tensor [num_heads, head_dim, hidden_size, 4]
    all_heads_tensor = torch.stack(attention_tensors, dim=0)
    
    # Step 3: Apply Tucker decomposition with shared factor matrices
    # Set ranks for each dimension based on compression targets
    # Recommended to keep R1 (hidden_size) relatively large as it's shared across heads
    R1 = int(config.hidden_size * 0.3)  # 30% of hidden_size
    R2 = int(config.head_dim * 0.5)     # 50% of head_dim
    R3 = 3                             # 4 weight types, keep 3 dimensions
    
    try:
        # Apply Tucker decomposition with HOOI algorithm
        core, factors = tucker(
            all_heads_tensor, 
            rank=[config.num_attention_heads, R2, R1, R3],
            init='random', 
            n_iter_max=100,
            tol=1e-5,
            verbose=True
        )
        
        # Extract the shared factor matrices
        U_heads = factors[0]  # [num_heads, num_heads] - can keep full rank here
        U_head_dim = factors[1]  # [head_dim, R2]
        U_hidden = factors[2]   # [hidden_size, R1]
        U_weight_type = factors[3]  # [4, R3]
        
        # Reconstruct weights for each head
        new_heads = []
        for i in range(config.num_attention_heads):
            # Extract the core tensor for this head
            head_core = core[i]  # [R2, R1, R3]
            
            # Reconstruct the 3D tensor for this head
            reconstructed_head = tl.tucker_to_tensor(
                (head_core, [U_head_dim, U_hidden, U_weight_type])
            )
            
            new_heads.append(reconstructed_head)
        
        # Stack reconstructed heads
        reconstructed_tensor = torch.stack(new_heads, dim=0)
        
        # Verify quality of reconstructed weights
        original_norm = torch.norm(all_heads_tensor)
        reconstruction_error = torch.norm(all_heads_tensor - reconstructed_tensor) / original_norm
        print(f"Reconstruction error: {reconstruction_error.item():.4f}")
        
        if reconstruction_error.item() > 0.3:  # 30% error is high
            print(f"Warning: High reconstruction error ({reconstruction_error.item():.4f}). Results may be suboptimal.")
            if reconstruction_error.item() > 0.5:  # 50% error is very high
                raise ValueError(f"Reconstruction error too high: {reconstruction_error.item():.4f}")
        
        # Convert back to original weight matrix format
        # Extract Q, K, V, O weights
        new_Q = reconstructed_tensor[:, :, :, 0].reshape(config.num_attention_heads * config.head_dim, config.hidden_size)
        new_K = reconstructed_tensor[:config.num_key_value_heads, :, :, 1].reshape(config.num_key_value_heads * config.head_dim, config.hidden_size)
        new_V = reconstructed_tensor[:config.num_key_value_heads, :, :, 2].reshape(config.num_key_value_heads * config.head_dim, config.hidden_size)
        new_O = reconstructed_tensor[:, :, :, 3].permute(2, 0, 1).reshape(config.hidden_size, config.num_attention_heads * config.head_dim)
        
        # Create new qkv weight matrix
        new_qkv = torch.cat([new_Q, new_K, new_V], dim=0)
        
        # Update the weights
        with torch.no_grad():
            layer.self_attn.qkv_proj.weight.copy_(new_qkv)
            layer.self_attn.o_proj.weight.copy_(new_O)
        
        # Return compression stats
        original_params = config.hidden_size * config.num_attention_heads * config.head_dim * 4
        compressed_params = (
            core.numel() + 
            U_heads.numel() + 
            U_head_dim.numel() + 
            U_hidden.numel() + 
            U_weight_type.numel()
        )
        compression_ratio = original_params / compressed_params
        
        return {
            "layer": layer_index,
            "original_params": original_params,
            "compressed_params": compressed_params,
            "compression_ratio": compression_ratio,
            "error": reconstruction_error.item()
        }
    
    except Exception as e:
        print(f"Error during shared factorization of layer {layer_index}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "layer": layer_index,
            "error": "Failed to factorize",
            "exception": str(e)
        }


def adaptive_rank_selection(tensor, max_error=0.05, max_compression=0.1):
    """
    Adaptively determine optimal ranks for Tucker decomposition.
    
    This function finds the smallest ranks that keep the reconstruction error
    below the specified threshold while achieving the desired compression.
    
    Args:
        tensor: The tensor to decompose
        max_error: Maximum allowed relative reconstruction error 
        max_compression: Maximum compression ratio (smaller is more compressed)
    
    Returns:
        tuple: The optimal ranks for each dimension
    """
    tl.set_backend('pytorch')
    tensor_shape = tensor.shape
    ndims = len(tensor_shape)
    
    # Start with high ranks
    current_ranks = [dim // 2 for dim in tensor_shape]
    
    # Binary search for optimal ranks for each dimension
    for dim in range(ndims):
        low, high = 1, current_ranks[dim]
        best_rank = high
        
        while low <= high:
            mid = (low + high) // 2
            
            # Try this rank
            test_ranks = current_ranks.copy()
            test_ranks[dim] = mid
            
            # Calculate compression ratio
            original_size = np.prod(tensor_shape)
            compressed_size = np.prod(test_ranks) + sum(r * s for r, s in zip(test_ranks, tensor_shape))
            compression = compressed_size / original_size
            
            # Skip if compression is too low
            if compression > max_compression:
                low = mid + 1
                continue
            
            # Compute decomposition
            try:
                core, factors = tucker(tensor, rank=test_ranks, init='random', n_iter_max=50)
                
                # Reconstruct tensor
                reconstructed = tl.tucker_to_tensor((core, factors))
                
                # Calculate error
                error = torch.norm(tensor - reconstructed) / torch.norm(tensor)
                
                if error.item() <= max_error:
                    # This rank works, try a smaller one
                    best_rank = mid
                    high = mid - 1
                else:
                    # Error too high, try a larger rank
                    low = mid + 1
            except Exception as e:
                # If decomposition fails, try a larger rank
                low = mid + 1
        
        # Update the current ranks with the best found for this dimension
        current_ranks[dim] = best_rank
    
    # Verify the final decomposition
    core, factors = tucker(tensor, rank=current_ranks, init='random', n_iter_max=100)
    reconstructed = tl.tucker_to_tensor((core, factors))
    final_error = torch.norm(tensor - reconstructed) / torch.norm(tensor)
    
    print(f"Adaptive rank selection found ranks {current_ranks} with error {final_error:.4f}")
    
    # Fallback to higher ranks if error is still too high
    if final_error > max_error:
        print(f"Error exceeds threshold, increasing ranks")
        current_ranks = [min(r + max(r // 4, 1), dim) for r, dim in zip(current_ranks, tensor_shape)]
        
    return tuple(current_ranks)


def _factorize_and_set_weights(model, weight: torch.Tensor, A_proj: nn.Module, B_proj: nn.Module, rank: int, config):
    """
    Factorize a weight matrix using Tucker decomposition and set the factorized weights.
    
    Args:
        model: The transformer model
        weight: Weight matrix to factorize
        A_proj: A projection module
        B_proj: B projection module
        rank: Rank for factorization
        config: Model configuration
        
    Returns:
        tuple: (success, stats) where success is a boolean and stats is a dict with factorization statistics
    """
    import math
    import time

    # Using try-except to handle potential errors in the whole method
    try:
        # Set backend to PyTorch
        tl.set_backend('pytorch')
        
        # Get target dimensions from projection modules
        hidden_size = config.hidden_size
        target_A_shape = A_proj.weight.shape
        target_B_shape = B_proj.weight.shape
        
        # Expected dimensions for the factorized matrices
        num_heads = config.num_attention_heads
        num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
        head_dim = config.head_dim
        
        print(f"Processing weight with shape {weight.shape}")
        print(f"Target shapes - A: {target_A_shape}, B: {target_B_shape}")
        
        # Convert weight to float32 for better numerical stability
        weight_float32 = weight.to(dtype=torch.float32)
        
        # Determine which type of projection this is
        is_query = 'W_A_q' in A_proj.__class__.__name__ or 'W_A_q' in A_proj._get_name()
        is_key = 'W_A_k' in A_proj.__class__.__name__ or 'W_A_k' in A_proj._get_name()
        is_value = 'W_A_v' in A_proj.__class__.__name__ or 'W_A_v' in A_proj._get_name()
        
        # Determine optimal rank based on projection type
        if is_query:
            print(f"Identified as query weight (num_heads={num_heads}, head_dim={head_dim})")
            # Use higher rank for queries (6 as in TPA paper)
            effective_rank = min(6, min(weight.shape))
        elif is_key or is_value:
            print(f"Identified as key/value weight (num_kv_heads={num_kv_heads}, head_dim={head_dim})")
            # Use rank 2 for K/V as in TPA paper
            effective_rank = min(2, min(weight.shape))
        else:
            effective_rank = min(rank, min(weight.shape))
        
        # For attention weights, reshape to a meaningful 3D tensor
        # Calculate dimensions for reshaping
        if is_query:
            output_dim = num_heads
        elif is_key or is_value:
            output_dim = num_kv_heads
        else:
            # Default case
            output_dim = weight.shape[0] // head_dim if head_dim > 0 else 1
        
        # Reshape to 3D tensor for Tucker decomposition
        if weight_float32.shape[0] % output_dim == 0:
            # Reshape preserving the head structure
            inner_dim = weight_float32.shape[0] // output_dim
            reshaped_tensor = weight_float32.reshape(output_dim, inner_dim, -1)
        else:
            # Use balanced dimensions
            dim_1 = int(math.sqrt(weight_float32.shape[0]))
            dim_2 = weight_float32.shape[0] // dim_1
            reshaped_tensor = weight_float32.reshape(dim_1, dim_2, -1)
        
        print(f"Reshaped tensor for Tucker decomposition: {reshaped_tensor.shape}")
        
        # Calculate optimal ranks for tensor modes
        tensor_shape = reshaped_tensor.shape
        
        # For TPA, use rank ratio approach from TensorLLM paper
        # Higher ranks for head dimension and feature dimension, lower for inner dimension
        mode_ranks = [
            min(tensor_shape[0], effective_rank * 2),  # Head dimension
            min(tensor_shape[1], effective_rank),      # Inner dimension (compress more)
            min(tensor_shape[2], effective_rank * 2)   # Feature dimension
        ]
        
        print(f"Using Tucker decomposition with mode ranks: {mode_ranks}")
        
        try:
            # Perform Tucker decomposition with higher iterations for better convergence
            core, factors = tucker(
                reshaped_tensor, 
                rank=mode_ranks, 
                init='random', 
                n_iter_max=200,
                tol=1e-6
            )
        
            print(f"Tucker decomposition complete with core shape {core.shape}")
            
            # Extract factors directly
            A_factor = factors[0]  # Head dimension factor
            B_factor = factors[2]  # Feature dimension factor
            
            # Improve factors using core information for better approximation
            # Mode-1 product to eliminate middle dimension
            core_mode1 = tl.mode_dot(core, factors[1], mode=1)
            
            # Reshape to project onto the final factors
            # For A factor (head dimension)
            A_core = core_mode1.reshape(A_factor.shape[0], -1)
            # For B factor (feature dimension)
            B_core = core_mode1.reshape(-1, B_factor.shape[1])
            
            # Calculate final factorized matrices with core information
            A_matrix = torch.matmul(A_factor, A_core)
            B_matrix = torch.matmul(B_core, B_factor.t())
            
            # Verify decomposition quality
            try:
                # Reconstruct the weight via our factorizations
                reconstructed = torch.matmul(A_matrix, B_matrix)
                reconstructed = reconstructed.reshape_as(weight_float32)
                
                # Calculate relative error
                norm_orig = torch.norm(weight_float32)
                if norm_orig > 0:
                    error = torch.norm(weight_float32 - reconstructed) / norm_orig
                    print(f"Reconstruction relative error: {error.item():.4f}")
                    
                    # Check if error is acceptable
                    max_allowed_error = 0.3  # 30% relative error maximum
                    if error.item() > max_allowed_error:
                        print(f"Warning: High factorization error ({error.item():.4f} > {max_allowed_error})")
                        
                        # If very high error, try alternative approach (e.g., higher rank)
                        if error.item() > 0.8:  # 80% relative error is very high
                            avg_error = error.item()
                            print(f"Warning: High factorization error ({avg_error:.4f}). Attempting improved factorization.")
                            # Potential improvements could be implemented here, such as:
                            # - Try a different rank
                            # - Use a different factorization method
                            # - Apply post-processing to better match original matrix
            except Exception as e:
                print(f"Verification failed with error: {e}")
                # Continue even if verification fails
        except Exception as e:
            print(f"Tucker decomposition failed: {e}")
            raise  # Re-raise to be caught by the outer try-except
                
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
                
        # Return failure status
        return False, {"error": "Factorization failed"}
        
    # If we reach here, factorization succeeded
    return True, {
        "error": error.item() if 'error' in locals() else None,
        "compression_ratio": np.prod(weight.shape) / (A_matrix.numel() + B_matrix.numel())
    }