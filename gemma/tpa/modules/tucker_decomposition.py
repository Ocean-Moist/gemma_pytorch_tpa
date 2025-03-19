"""
Tucker decomposition implementations for Tensor Product Attention.

This module provides memory-efficient implementations of Tucker decomposition
for large tensors, including tile-based approaches for minimal memory footprint.
"""

import torch
import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union

from .svd_utils import patched_svd

def memory_efficient_tucker(tensor, ranks):
    """
    Memory-efficient implementation of Tucker decomposition.
    
    This implementation avoids materializing large intermediate tensors
    and uses numerically stable decomposition approaches.
    
    Args:
        tensor: Input tensor for decomposition
        ranks: Target ranks for each mode
        
    Returns:
        core: Core tensor
        factors: List of factor matrices
    """
    if tensor.dim() != 3:
        raise ValueError(f"Expected 3D tensor, got {tensor.dim()}D tensor")
    
    # Work in float32 for numerical stability
    original_dtype = tensor.dtype
    tensor = tensor.to(torch.float32)
    
    # Save original shape and device
    shape = tensor.shape
    device = tensor.device
    
    # Ensure tensor is on GPU if available
    if torch.cuda.is_available() and device.type != 'cuda':
        tensor = tensor.cuda()
        device = tensor.device
        print(f"Moving tensor to {device} for Tucker decomposition")
    
    # Check if tensor contains any NaN or Inf values
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print("Warning: Tensor contains NaN or Inf values. Replacing with zeros...")
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize tensor to improve numerical stability
    tensor_norm = torch.norm(tensor)
    if tensor_norm > 0:
        tensor = tensor / tensor_norm
    
    # Adjust ranks if needed
    ranks = [min(r, s) for r, s in zip(ranks, shape)]
    
    # Initialize factors
    factors = []
    
    # Process each mode
    modes_processed = 0
    for mode in range(tensor.dim()):
        try:
            # Unfold tensor along current mode
            tensor_unfolded = unfold_tensor(tensor, mode)
            
            # Center the data for improved SVD stability
            mean_val = torch.mean(tensor_unfolded, dim=1, keepdim=True)
            tensor_unfolded = tensor_unfolded - mean_val
            
            # Compute SVD with error handling
            try:
                U, S, Vh = torch.linalg.svd(tensor_unfolded, full_matrices=False)
            except RuntimeError as e:
                print(f"SVD failed, trying with lower precision: {e}")
                # Try with numpy's SVD
                tensor_unfolded_np = tensor_unfolded.cpu().numpy()
                U_np, S_np, Vh_np = patched_svd(tensor_unfolded_np, full_matrices=False)
                U = torch.from_numpy(U_np).to(device)
                S = torch.from_numpy(S_np).to(device)
                Vh = torch.from_numpy(Vh_np).to(device)
            
            # Check for NaN/Inf values in decomposition results
            if torch.isnan(U).any() or torch.isinf(U).any() or \
               torch.isnan(S).any() or torch.isinf(S).any():
                print(f"Warning: NaN/Inf in mode {mode} decomposition, trying with noise regularization")
                
                # Add small noise to tensor for regularization
                noise = torch.randn_like(tensor_unfolded) * 1e-5
                tensor_unfolded = tensor_unfolded + noise
                
                # Retry SVD
                try:
                    U, S, Vh = torch.linalg.svd(tensor_unfolded, full_matrices=False)
                except RuntimeError:
                    tensor_unfolded_np = tensor_unfolded.cpu().numpy()
                    U_np, S_np, Vh_np = patched_svd(tensor_unfolded_np, full_matrices=False)
                    U = torch.from_numpy(U_np).to(device)
                    S = torch.from_numpy(S_np).to(device)
                    Vh = torch.from_numpy(Vh_np).to(device)
            
            # Extract factor matrix (truncate to rank)
            rank = ranks[mode]
            factor = U[:, :rank]
            
            # Orthogonalization via QR factorization for improved stability
            q, r = torch.linalg.qr(factor)
            factor = q
            
            factors.append(factor)
            
            # Project tensor onto factor
            tensor = mode_dot(tensor, factor.T, mode)
            modes_processed += 1
            
        except Exception as e:
            print(f"Error processing mode {mode}: {e}")
            # Fallback: use random orthogonal matrix
            rank = ranks[mode]
            random_factor = torch.randn((shape[mode], rank), device=device)
            q, r = torch.linalg.qr(random_factor)
            factors.append(q)
    
    # The core tensor is the result after all projections
    core = tensor
    
    # Restore original scale
    core = core * tensor_norm
    
    # Ensure all factors are properly created
    if len(factors) < tensor.dim():
        print("Warning: Some modes failed to process, using fallback factors")
        for mode in range(len(factors), tensor.dim()):
            rank = ranks[mode]
            random_factor = torch.randn((shape[mode], rank), device=device)
            q, r = torch.linalg.qr(random_factor)
            factors.append(q)
    
    # Return core and factors in original dtype
    return core.to(original_dtype), [f.to(original_dtype) for f in factors]

def tile_based_tucker(tensor, ranks, tile_size=1000):
    """
    Tile-based Tucker decomposition for extremely large tensors.
    
    This method decomposes tensors using smaller tiles to minimize
    memory usage, particularly useful for very large tensors.
    
    Args:
        tensor: Input tensor
        ranks: Target ranks for each mode
        tile_size: Maximum size of tiles to process
        
    Returns:
        core: Core tensor
        factors: List of factor matrices
    """
    if tensor.dim() != 3:
        raise ValueError(f"Expected 3D tensor, got {tensor.dim()}D tensor")
    
    # Work in float32 for numerical stability
    original_dtype = tensor.dtype
    tensor = tensor.to(torch.float32)
    
    # Save original shape and device
    shape = tensor.shape
    device = tensor.device
    
    # Ensure tensor is on GPU if available
    if torch.cuda.is_available() and device.type != 'cuda':
        tensor = tensor.cuda()
        device = tensor.device
        print(f"Moving tensor to {device} for tile-based Tucker decomposition")
    device = tensor.device
    
    # Replace any NaN/Inf values
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Initialize factors
    factors = []
    
    # Adjust tile size based on tensor dimensions
    tile_size = min(tile_size, max(shape))
    
    # Process each mode
    for mode in range(tensor.dim()):
        try:
            # Determine the dimension size for current mode
            mode_size = shape[mode]
            
            # Calculate number of tiles needed
            num_tiles = math.ceil(mode_size / tile_size)
            
            # Initialize covariance matrix for this mode
            cov_matrix = torch.zeros((mode_size, mode_size), dtype=torch.float32, device=device)
            
            # Process tiles
            for i in range(num_tiles):
                start_idx = i * tile_size
                end_idx = min((i + 1) * tile_size, mode_size)
                
                # Create slice for current tile
                slice_indices = [slice(None)] * tensor.dim()
                slice_indices[mode] = slice(start_idx, end_idx)
                
                # Extract tile
                tile = tensor[tuple(slice_indices)]
                
                # Unfold tile along current mode
                tile_unfolded = unfold_tensor(tile, mode)
                
                # Update covariance matrix
                for j in range(num_tiles):
                    j_start = j * tile_size
                    j_end = min((j + 1) * tile_size, mode_size)
                    
                    # Create slice for covariance update
                    j_slice_indices = [slice(None)] * tensor.dim()
                    j_slice_indices[mode] = slice(j_start, j_end)
                    
                    # Extract corresponding tile
                    j_tile = tensor[tuple(j_slice_indices)]
                    j_tile_unfolded = unfold_tensor(j_tile, mode)
                    
                    # Update covariance submatrix
                    cov_matrix[start_idx:end_idx, j_start:j_end] = tile_unfolded @ j_tile_unfolded.T
            
            # Compute eigendecomposition of covariance matrix
            try:
                eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
                
                # Sort eigenvalues and eigenvectors in descending order
                idx = torch.argsort(eigenvalues, descending=True)
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
                
            except RuntimeError as e:
                print(f"Eigendecomposition failed: {e}, trying alternative approach")
                
                # Use power iteration as fallback
                rank = ranks[mode]
                eigenvectors = torch.randn((mode_size, rank), device=device)
                
                # Power iteration
                for _ in range(10):
                    # Orthogonalize
                    q, r = torch.linalg.qr(eigenvectors)
                    eigenvectors = q
                    
                    # Power iteration step
                    eigenvectors = cov_matrix @ eigenvectors
                    
                    # Re-orthogonalize
                    q, r = torch.linalg.qr(eigenvectors)
                    eigenvectors = q
            
            # Extract factor matrix (truncate to rank)
            rank = min(ranks[mode], eigenvalues.size(0))
            factor = eigenvectors[:, :rank]
            
            # Ensure orthogonality
            q, r = torch.linalg.qr(factor)
            factor = q
            
            factors.append(factor)
            
            # Project tensor onto factor
            tensor = mode_dot(tensor, factor.T, mode)
            
        except Exception as e:
            print(f"Error in tile-based processing for mode {mode}: {e}")
            # Fallback: use random orthogonal matrix
            rank = ranks[mode]
            random_factor = torch.randn((shape[mode], rank), device=device)
            q, r = torch.linalg.qr(random_factor)
            factors.append(q)
    
    # The core tensor is the result after all projections
    core = tensor
    
    # Return core and factors in original dtype
    return core.to(original_dtype), [f.to(original_dtype) for f in factors]

def unfold_tensor(tensor, mode):
    """
    Unfold a tensor along a specific mode.
    
    Args:
        tensor: Input tensor
        mode: Mode along which to unfold
        
    Returns:
        Unfolded tensor as a matrix
    """
    shape = tensor.shape
    fibers = []
    
    if mode == 0:
        return tensor.reshape(shape[0], -1)
    elif mode == 1:
        return tensor.permute(1, 0, 2).reshape(shape[1], -1)
    elif mode == 2:
        return tensor.permute(2, 0, 1).reshape(shape[2], -1)
    else:
        raise ValueError(f"Invalid mode {mode} for 3D tensor")

def mode_dot(tensor, matrix, mode):
    """
    Mode-n product of a tensor with a matrix.
    
    Args:
        tensor: Input tensor
        matrix: Matrix to multiply with
        mode: Mode along which to perform the multiplication
        
    Returns:
        Result of the mode-n product
    """
    # Handle tensor with dimension <= 3
    if tensor.dim() > 3:
        raise ValueError(f"Only tensors with dim <= 3 supported, got {tensor.dim()}")
    
    # Compute mode-n product
    if mode == 0:
        return torch.matmul(matrix, tensor.reshape(tensor.shape[0], -1)).reshape(matrix.shape[0], tensor.shape[1], tensor.shape[2])
    elif mode == 1:
        return torch.matmul(matrix, tensor.permute(1, 0, 2).reshape(tensor.shape[1], -1)).reshape(matrix.shape[0], tensor.shape[0], tensor.shape[2]).permute(1, 0, 2)
    elif mode == 2:
        return torch.matmul(matrix, tensor.permute(2, 0, 1).reshape(tensor.shape[2], -1)).reshape(matrix.shape[0], tensor.shape[0], tensor.shape[1]).permute(1, 2, 0)
    else:
        raise ValueError(f"Invalid mode {mode} for tensor with dim {tensor.dim()}")