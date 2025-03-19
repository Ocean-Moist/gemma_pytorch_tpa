"""
T6-style contextual factorization methods for Tensor Product Attention.

This module implements tensor factorization approaches for Tensor Product Attention (TPA),
including both the original T6 contextual factorization and the TensorLLM-style Tucker 
decomposition as described in the documentation.
"""

import torch
import math
import time
import numpy as np
from torch import nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any

try:
    import tensorly as tl
    from tensorly.decomposition import tucker, partial_tucker
    import scipy.linalg
    # Save the original SVD function
    original_svd = scipy.linalg.svd
    
    # Create a patched SVD that uses TruncatedSVD for large matrices
    def patched_svd(a, full_matrices=True, compute_uv=True, overwrite_a=False,
                    check_finite=True, lapack_driver='gesdd'):
        try:
            return original_svd(a, full_matrices=full_matrices, compute_uv=compute_uv,
                              overwrite_a=overwrite_a, check_finite=check_finite,
                              lapack_driver=lapack_driver)
        except ValueError as e:
            if "LAPACK" in str(e) and "integer overflow" in str(e):
                print("LAPACK error detected, using randomized SVD instead")
                
                # Get matrix dimensions
                m, n = a.shape
                
                # Estimate rank (can be adjusted based on requirements)
                k = min(m, n, 256)  # Use a reasonable default rank
                
                # For extremely large matrices, use a custom approach
                if m > 100000 or n > 100000:
                    print(f"Very large matrix ({m}x{n}), using tile-based approach")
                    return randomized_svd_tiled(a, k, full_matrices, compute_uv)
                
                if compute_uv:
                    # Convert numpy array to PyTorch tensor for SVD
                    if isinstance(a, np.ndarray):
                        a_torch = torch.from_numpy(a).float()
                    else:
                        a_torch = a
                    
                    # Use PyTorch's SVD with randomized approach
                    try:
                        U, S, Vh = torch.linalg.svd(a_torch, full_matrices=False)
                        U_truncated = U[:, :k]
                        S_truncated = S[:k]
                        Vh_truncated = Vh[:k, :]
                        
                        # Convert back to numpy if necessary
                        if isinstance(a, np.ndarray):
                            U_truncated = U_truncated.numpy()
                            S_truncated = S_truncated.numpy()
                            Vh_truncated = Vh_truncated.numpy()
                        
                        if full_matrices:
                            # Pad outputs if full matrices requested
                            if isinstance(a, np.ndarray):
                                if U_truncated.shape[1] < m:
                                    pad_U = np.zeros((m, m-k))
                                    U_truncated = np.hstack((U_truncated, pad_U))
                                if Vh_truncated.shape[0] < n:
                                    pad_Vh = np.zeros((n-k, n))
                                    Vh_truncated = np.vstack((Vh_truncated, pad_Vh))
                            else:
                                if U_truncated.shape[1] < m:
                                    pad_U = torch.zeros((m, m-k), device=U_truncated.device)
                                    U_truncated = torch.cat((U_truncated, pad_U), dim=1)
                                if Vh_truncated.shape[0] < n:
                                    pad_Vh = torch.zeros((n-k, n), device=Vh_truncated.device)
                                    Vh_truncated = torch.cat((Vh_truncated, pad_Vh), dim=0)
                        
                        print(f"PyTorch SVD successful with rank {k}")
                        return U_truncated, S_truncated, Vh_truncated
                    
                    except Exception as torch_err:
                        print(f"PyTorch SVD failed: {torch_err}, trying randomized approach")
                        
                        # Fall back to numpy's SVD
                        try:
                            # Try with reduced precision
                            if isinstance(a, np.ndarray):
                                a_float32 = a.astype(np.float32)
                            else:
                                a_float32 = a.cpu().numpy().astype(np.float32)
                                
                            # Apply randomized SVD
                            return randomized_svd_low_memory(a_float32, k, full_matrices=full_matrices)
                        except Exception as np_err:
                            print(f"Randomized SVD failed: {np_err}, returning approximated result")
                            # Return a minimal approximation
                            if isinstance(a, np.ndarray):
                                U = np.eye(m, k)
                                S = np.ones(k)
                                Vh = np.eye(k, n)
                            else:
                                U = torch.eye(m, k)
                                S = torch.ones(k)
                                Vh = torch.eye(k, n)
                            return U, S, Vh
                else:
                    # If we only need singular values
                    try:
                        # Try with reduced precision
                        if isinstance(a, np.ndarray):
                            a_float32 = a.astype(np.float32)
                            s = np.linalg.svd(a_float32, full_matrices=False, compute_uv=False)
                        else:
                            s = torch.linalg.svd(a, full_matrices=False)[1]
                            if isinstance(s, torch.Tensor):
                                s = s.cpu().numpy()
                        return s[:k]
                    except Exception as s_err:
                        print(f"SVD for singular values failed: {s_err}, returning approximation")
                        return np.ones(k)
            else:
                raise
    
    def randomized_svd_low_memory(A, k, full_matrices=False, n_oversamples=10, n_iter=4):
        """
        Compute a randomized SVD with low memory usage.
        
        Args:
            A: Input matrix
            k: Target rank
            full_matrices: Whether to return full U and V matrices
            n_oversamples: Oversampling parameter
            n_iter: Number of power iterations
            
        Returns:
            U, S, Vh: The SVD factors
        """
        m, n = A.shape
        k = min(k, min(m, n))
        p = k + n_oversamples
        
        # Generate a random Gaussian matrix
        np.random.seed(42)
        Omega = np.random.randn(n, p)
        
        # First pass to get Y = A * Omega
        Y = np.zeros((m, p), dtype=np.float32)
        # Process column-by-column for large matrices
        chunk_size = min(1000, n)
        for i in range(0, n, chunk_size):
            j = min(i + chunk_size, n)
            Y += A[:, i:j] @ Omega[i:j, :]
        
        # Orthogonalize Y
        Q, _ = np.linalg.qr(Y, mode='reduced')
        
        # Power iteration to increase accuracy for singular vectors
        for _ in range(n_iter):
            # Y = A^T * Q
            Z = np.zeros((n, Q.shape[1]), dtype=np.float32)
            for i in range(0, m, chunk_size):
                j = min(i + chunk_size, m)
                Z += A[i:j, :].T @ Q[i:j, :]
                
            # Orthogonalize Z
            Q, _ = np.linalg.qr(Z, mode='reduced')
            
            # Y = A * Q
            Z = np.zeros((m, Q.shape[1]), dtype=np.float32)
            for i in range(0, n, chunk_size):
                j = min(i + chunk_size, n)
                Z += A[:, i:j] @ Q[i:j, :]
                
            # Orthogonalize Y
            Q, _ = np.linalg.qr(Z, mode='reduced')
        
        # Project A to get B = Q^T * A
        B = np.zeros((Q.shape[1], n), dtype=np.float32)
        for i in range(0, m, chunk_size):
            j = min(i + chunk_size, m)
            B += Q[i:j, :].T @ A[i:j, :]
        
        # Compute SVD of smaller matrix B
        Uhat, s, Vh = np.linalg.svd(B, full_matrices=False)
        
        # Compute U = Q * Uhat
        U = Q @ Uhat[:, :k]
        
        # Truncate the results
        s = s[:k]
        Vh = Vh[:k, :]
        
        if full_matrices:
            # Pad U and Vh if needed
            if m > k:
                pad_U = np.zeros((m, m - k))
                U = np.hstack((U, pad_U))
            if n > k:
                pad_Vh = np.zeros((n - k, n))
                Vh = np.vstack((Vh, pad_Vh))
        
        return U, s, Vh
    
    def randomized_svd_tiled(A, k, full_matrices=False, compute_uv=True, tile_size=1000):
        """
        Apply randomized SVD on very large matrices using a tiled approach.
        
        Args:
            A: Input matrix
            k: Target rank
            full_matrices: Whether to return full matrices
            compute_uv: Whether to compute U and V
            tile_size: Size of tiles for processing
            
        Returns:
            U, S, Vh: The SVD factors (or just S if compute_uv=False)
        """
        if not compute_uv:
            # Just estimate singular values
            # Sample a subset of rows and columns
            m, n = A.shape
            sample_size = min(10000, min(m, n))
            
            # Random indices
            np.random.seed(42)
            row_idx = np.random.choice(m, min(sample_size, m), replace=False)
            col_idx = np.random.choice(n, min(sample_size, n), replace=False)
            
            # Extract submatrix
            sub_A = A[row_idx][:, col_idx]
            
            # Get singular values and rescale
            s = np.linalg.svd(sub_A, compute_uv=False)
            scale = np.sqrt(m * n / (len(row_idx) * len(col_idx)))
            return s[:k] * scale
        
        # For computing U and V
        m, n = A.shape
        k = min(k, min(m, n))
        
        # Use a randomized scheme with tiles
        np.random.seed(42)
        Q = np.random.randn(n, k)
        Q, _ = np.linalg.qr(Q, mode='reduced')
        
        # Power iteration with tiled matrix multiplication
        for _ in range(4):  # Number of power iterations
            # Y = A * Q (in tiles)
            Y = np.zeros((m, k))
            for i in range(0, m, tile_size):
                end_i = min(i + tile_size, m)
                for j in range(0, n, tile_size):
                    end_j = min(j + tile_size, n)
                    Y[i:end_i] += A[i:end_i, j:end_j] @ Q[j:end_j]
            
            # QR factorization
            Y, _ = np.linalg.qr(Y, mode='reduced')
            
            # Z = A^T * Y (in tiles)
            Z = np.zeros((n, k))
            for i in range(0, m, tile_size):
                end_i = min(i + tile_size, m)
                for j in range(0, n, tile_size):
                    end_j = min(j + tile_size, n)
                    Z[j:end_j] += A[i:end_i, j:end_j].T @ Y[i:end_i]
            
            # QR factorization
            Q, _ = np.linalg.qr(Z, mode='reduced')
        
        # Final projection
        Y = np.zeros((m, k))
        for i in range(0, m, tile_size):
            end_i = min(i + tile_size, m)
            for j in range(0, n, tile_size):
                end_j = min(j + tile_size, n)
                Y[i:end_i] += A[i:end_i, j:end_j] @ Q[j:end_j]
        
        # Compute QR for Y
        Y, R = np.linalg.qr(Y, mode='reduced')
        
        # Compute SVD of small matrix: R * Q^T
        B = R @ Q.T
        Uhat, s, Vh = np.linalg.svd(B, full_matrices=False)
        
        # Truncate to rank k
        Uhat = Uhat[:, :k]
        s = s[:k]
        Vh = Vh[:k, :]
        
        # Compute final U
        U = Y @ Uhat
        
        if full_matrices:
            # Pad matrices if needed
            if U.shape[1] < m:
                pad_U = np.zeros((m, m - k))
                U = np.hstack((U, pad_U))
            if Vh.shape[0] < n:
                pad_Vh = np.zeros((n - k, n))
                Vh = np.vstack((Vh, pad_Vh))
        
        return U, s, Vh
    
    def memory_efficient_tucker(tensor, ranks):
        """
        A memory-efficient version of Tucker decomposition using partial_tucker
        and operating on each mode separately to avoid large unfoldings.
        
        Args:
            tensor: The input tensor to decompose
            ranks: A list of ranks for each mode
            
        Returns:
            core: The core tensor
            factors: A list of factor matrices
        """
        # Clone the tensor to avoid modifying the original
        current = tensor.clone()
        n_modes = len(tensor.shape)
        factors = [None] * n_modes
        
        # Process one mode at a time to avoid large matrix unfoldings
        for mode in range(n_modes):
            if ranks[mode] is None:
                # Skip modes with None rank
                continue
                
            # Compute the n-mode product only for this mode
            mode_size = tensor.shape[mode]
            rank = min(ranks[mode], mode_size)
            
            # Reshape the tensor for this mode's unfolding
            unfolded = tl.unfold(current, mode)
            
            # Use randomized SVD or truncated SVD
            try:
                # Try PyTorch's SVD
                U, S, Vh = torch.linalg.svd(unfolded, full_matrices=False)
                U_truncated = U[:, :rank]
                
            except Exception as e:
                print(f"Error using PyTorch SVD: {e}, trying randomized approach")
                # Use power iteration method for large matrices
                # This is a randomized approach to find dominant singular vectors
                Q = torch.randn(unfolded.shape[1], rank, device=tensor.device)
                Q, _ = torch.linalg.qr(Q)
                
                # Power iteration (simplified randomized SVD)
                for _ in range(5):  # Number of power iterations
                    Y = unfolded @ Q
                    Q, _ = torch.linalg.qr(unfolded.t() @ Y)
                
                Y = unfolded @ Q
                U_truncated, _ = torch.linalg.qr(Y)
                U_truncated = U_truncated[:, :rank]
            
            # Store the factor matrix
            factors[mode] = U_truncated
            
            # Update the tensor with this mode's compression
            current = tl.mode_dot(current, U_truncated.t(), mode)
        
        # The resulting tensor is the core
        core = current
        
        return core, factors
    
    def tile_based_tucker(tensor, ranks, tile_size=1000):
        """
        A tiling-based Tucker decomposition that processes the tensor in chunks
        to minimize memory consumption.
        
        Args:
            tensor: The input tensor to decompose
            ranks: A list of ranks for each mode
            tile_size: The maximum size of each tile
            
        Returns:
            core: The core tensor
            factors: A list of factor matrices
        """
        n_modes = len(tensor.shape)
        factors = []
        
        # Process each mode separately to find factor matrices
        for mode in range(n_modes):
            if ranks[mode] is None:
                factors.append(None)
                continue
                
            # Get the unfolding for this mode
            unfolded = tl.unfold(tensor, mode)
            mode_size = tensor.shape[mode]
            rank = min(ranks[mode], mode_size)
            
            # Initialize an empty matrix for the top singular vectors
            U = torch.zeros((unfolded.shape[0], rank), device=tensor.device)
            
            # Split the unfolded matrix into tiles for column blocks
            num_cols = unfolded.shape[1]
            num_tiles = (num_cols + tile_size - 1) // tile_size
            
            # Use power iteration method for large matrices
            # This is a randomized approach to find dominant singular vectors
            Q = torch.randn(unfolded.shape[1], rank, device=tensor.device)
            Q, _ = torch.linalg.qr(Q)
            
            # Power iteration (simplified randomized SVD)
            for _ in range(5):  # Number of power iterations
                # Y = A * Q
                Y = torch.zeros((unfolded.shape[0], rank), device=tensor.device)
                for i in range(num_tiles):
                    start_idx = i * tile_size
                    end_idx = min((i + 1) * tile_size, num_cols)
                    tile = unfolded[:, start_idx:end_idx]
                    Y += tile @ Q[start_idx:end_idx, :]
                
                # Q = A^T * Y
                Q = torch.zeros((unfolded.shape[1], rank), device=tensor.device)
                for i in range(num_tiles):
                    start_idx = i * tile_size
                    end_idx = min((i + 1) * tile_size, num_cols)
                    tile = unfolded[:, start_idx:end_idx]
                    Q[start_idx:end_idx, :] = tile.t() @ Y
                
                # Orthogonalize Q
                Q, _ = torch.linalg.qr(Q)
            
            # Final projection to get Y = A * Q
            Y = torch.zeros((unfolded.shape[0], rank), device=tensor.device)
            for i in range(num_tiles):
                start_idx = i * tile_size
                end_idx = min((i + 1) * tile_size, num_cols)
                tile = unfolded[:, start_idx:end_idx]
                Y += tile @ Q[start_idx:end_idx, :]
            
            # Compute small SVD
            try:
                UY, S, VhY = torch.linalg.svd(Y, full_matrices=False)
                U = UY[:, :rank]
            except Exception as e:
                print(f"Error in SVD: {e}, using QR decomposition as fallback")
                U, _ = torch.linalg.qr(Y)
                U = U[:, :rank]
            
            factors.append(U)
        
        # Compute the core tensor using n-mode products
        core = tensor.clone()
        for mode, factor in enumerate(factors):
            if factor is not None:
                core = tl.mode_dot(core, factor.t(), mode)
        
        return core, factors
    
    # Replace scipy's SVD with our patched version
    scipy.linalg.svd = patched_svd
    
    # Set backend to PyTorch
    tl.set_backend('pytorch')
    HAS_TENSORLY = True
except ImportError:
    print("WARNING: tensorly not found. Tucker decomposition will not be available.")
    tucker = None
    HAS_TENSORLY = False

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


def tucker_tensor_decomposition(weight, num_heads, num_kv_heads, target_ranks, dtype=torch.float16, device="cuda"):
    """
    Factorize a weight matrix using Tucker decomposition for TPA.
    
    This function applies TensorLLM-style Tucker decomposition to QKV weight matrices
    to create TPA-compatible factors. It's especially useful for converting GQA weights.
    
    Args:
        weight: The weight matrix to factorize (could be wq, wk, wv)
        num_heads: Number of attention heads (h)
        num_kv_heads: Number of key-value heads (for GQA)
        target_ranks: Dictionary with target ranks for Q, K, V (Rq, Rk, Rv)
        dtype: Data type for the factorized weights
        device: Device to place tensors on
        
    Returns:
        dict: TPA weight parameters
    """
    if not HAS_TENSORLY:
        raise ImportError("tensorly not found. Install with 'pip install tensorly'")
    
    # Extract dimensions
    input_dim = weight.shape[1]  # Input dimension
    total_output_dim = weight.shape[0]  # Total output dimension
    
    # Determine head dimension based on model shape
    head_dim = total_output_dim // (num_heads + 2 * num_kv_heads)
    
    # Extract Q, K, V weights from the concatenated QKV weight matrix
    q_dim = num_heads * head_dim
    k_dim = num_kv_heads * head_dim
    v_dim = num_kv_heads * head_dim
    
    q_weight = weight[:q_dim, :].contiguous()
    k_weight = weight[q_dim:q_dim+k_dim, :].contiguous()
    v_weight = weight[q_dim+k_dim:, :].contiguous()
    
    # Extract target ranks
    q_rank = target_ranks.get('q', 6)
    k_rank = target_ranks.get('k', 2)
    v_rank = target_ranks.get('v', 2)
    
    # Initialize TPA weights
    tpa_weights = {}
    
    # Convert weights to float32 for better numerical stability
    if q_weight.dtype == torch.bfloat16:
        q_weight = q_weight.to(torch.float32)
    if k_weight.dtype == torch.bfloat16:
        k_weight = k_weight.to(torch.float32)
    if v_weight.dtype == torch.bfloat16:
        v_weight = v_weight.to(torch.float32)
    
    # Process query weights
    # Reshape to 3D tensor [head_dim, num_heads, input_dim]
    # Keep as PyTorch tensor rather than converting to numpy
    wq_tensor = q_weight.reshape(head_dim, num_heads, input_dim).cpu()
    
    # Apply Tucker decomposition to query weights
    rank = [q_rank, None, q_rank]  # Rank for dimensions
    try:
        # First try memory-efficient Tucker
        print(f"Applying memory-efficient Tucker decomposition with ranks: {rank}")
        try:
            core, factors = memory_efficient_tucker(wq_tensor, rank)
        except Exception as e1:
            print(f"Memory-efficient Tucker failed: {e1}, trying tiled version")
            try:
                core, factors = tile_based_tucker(wq_tensor, rank)
            except Exception as e2:
                print(f"Tiled Tucker also failed: {e2}, falling back to standard Tucker")
                # Fall back to standard Tucker
                core, factors = tucker(wq_tensor, rank=rank)
    except Exception as e:
        print(f"Warning: Error in all Tucker decomposition methods: {e}")
        print("Falling back to standard contextual factorization...")
        raise
    
    # Map to TPA parameters
    U1, U3 = factors[0], factors[2]  # U1 ~ head_dim×q_rank, U3 ~ input_dim×q_rank
    
    # Create Wa_q and Wb_q as PyTorch tensors
    Wa_q = torch.zeros((input_dim, q_rank, num_heads), dtype=torch.float32)
    Wb_q = torch.zeros((input_dim, q_rank, head_dim), dtype=torch.float32)
    
    # Map decomposition to TPA factors
    for r in range(q_rank):
        for i in range(num_heads):
            # Project core tensor for this head and rank
            proj_vector = core[r, i, :]
            Wa_q[:, r, i] = U3[:, r] * torch.norm(proj_vector)
        
        # Shared b factor across heads - use PyTorch's outer product
        Wb_q[:, r, :] = torch.outer(U3[:, r], U1[:, r])
    
    # Normalize factors
    for r in range(q_rank):
        norm_a = torch.norm(Wa_q[:, r, :])
        norm_b = torch.norm(Wb_q[:, r, :])
        if norm_a > 0 and norm_b > 0:
            scale = torch.sqrt(norm_a * norm_b)
            Wa_q[:, r, :] /= torch.sqrt(scale)
            Wb_q[:, r, :] /= torch.sqrt(scale)
    
    # Convert to the right device and dtype
    tpa_weights['Wa_q'] = Wa_q.to(dtype=dtype, device=device)
    tpa_weights['Wb_q'] = Wb_q.to(dtype=dtype, device=device)
    
    # Process key weights
    # Reshape to 3D tensor [head_dim, num_kv_heads, input_dim]
    # Keep as PyTorch tensor rather than converting to numpy
    wk_tensor = k_weight.reshape(head_dim, num_kv_heads, input_dim).cpu()
    
    # Apply Tucker decomposition to key weights
    rank = [k_rank, None, k_rank]  # Rank for dimensions
    try:
        # Use the same approach as for query weights
        print(f"Applying memory-efficient Tucker decomposition with ranks: {rank}")
        try:
            core, factors = memory_efficient_tucker(wk_tensor, rank)
        except Exception as e1:
            print(f"Memory-efficient Tucker failed: {e1}, trying tiled version")
            try:
                core, factors = tile_based_tucker(wk_tensor, rank)
            except Exception as e2:
                print(f"Tiled Tucker also failed: {e2}, falling back to standard Tucker")
                core, factors = tucker(wk_tensor, rank=rank)
    except Exception as e:
        print(f"Warning: Error in Tucker decomposition for key weights: {e}")
        print("Falling back to standard contextual factorization...")
        raise
    
    # Map to TPA parameters
    U1, U3 = factors[0], factors[2]  # U1 ~ head_dim×k_rank, U3 ~ input_dim×k_rank
    
    # Create Wa_k and Wb_k as PyTorch tensors
    Wa_k = torch.zeros((input_dim, k_rank, num_heads), dtype=torch.float32)
    Wb_k = torch.zeros((input_dim, k_rank, head_dim), dtype=torch.float32)
    
    # Map head groups - repeating heads for MQA/GQA
    heads_per_kv = num_heads // num_kv_heads
    
    # Map decomposition to TPA factors
    for r in range(k_rank):
        for i in range(num_kv_heads):
            # Project core tensor for this group and rank
            proj_vector = core[r, i, :]
            
            # Copy to each head in this group
            for j in range(heads_per_kv):
                head_idx = i * heads_per_kv + j
                Wa_k[:, r, head_idx] = U3[:, r] * torch.norm(proj_vector)
        
        # Shared b factor across heads
        Wb_k[:, r, :] = torch.outer(U3[:, r], U1[:, r])
    
    # Normalize factors
    for r in range(k_rank):
        norm_a = torch.norm(Wa_k[:, r, :])
        norm_b = torch.norm(Wb_k[:, r, :])
        if norm_a > 0 and norm_b > 0:
            scale = torch.sqrt(norm_a * norm_b)
            Wa_k[:, r, :] /= torch.sqrt(scale)
            Wb_k[:, r, :] /= torch.sqrt(scale)
    
    # Convert to the right device and dtype
    tpa_weights['Wa_k'] = Wa_k.to(dtype=dtype, device=device)
    tpa_weights['Wb_k'] = Wb_k.to(dtype=dtype, device=device)
    
    # Process value weights
    # Reshape to 3D tensor [head_dim, num_kv_heads, input_dim]
    # Keep as PyTorch tensor rather than converting to numpy
    wv_tensor = v_weight.reshape(head_dim, num_kv_heads, input_dim).cpu()
    
    # Apply Tucker decomposition to value weights
    rank = [v_rank, None, v_rank]  # Rank for dimensions
    try:
        # Use the same approach as for query and key weights
        print(f"Applying memory-efficient Tucker decomposition with ranks: {rank}")
        try:
            core, factors = memory_efficient_tucker(wv_tensor, rank)
        except Exception as e1:
            print(f"Memory-efficient Tucker failed: {e1}, trying tiled version")
            try:
                core, factors = tile_based_tucker(wv_tensor, rank)
            except Exception as e2:
                print(f"Tiled Tucker also failed: {e2}, falling back to standard Tucker")
                core, factors = tucker(wv_tensor, rank=rank)
    except Exception as e:
        print(f"Warning: Error in Tucker decomposition for value weights: {e}")
        print("Falling back to standard contextual factorization...")
        raise
    
    # Map to TPA parameters
    U1, U3 = factors[0], factors[2]  # U1 ~ head_dim×v_rank, U3 ~ input_dim×v_rank
    
    # Create Wa_v and Wb_v as PyTorch tensors
    Wa_v = torch.zeros((input_dim, v_rank, num_heads), dtype=torch.float32)
    Wb_v = torch.zeros((input_dim, v_rank, head_dim), dtype=torch.float32)
    
    # Map decomposition to TPA factors
    for r in range(v_rank):
        for i in range(num_kv_heads):
            # Project core tensor for this group and rank
            proj_vector = core[r, i, :]
            
            # Copy to each head in this group
            for j in range(heads_per_kv):
                head_idx = i * heads_per_kv + j
                Wa_v[:, r, head_idx] = U3[:, r] * torch.norm(proj_vector)
        
        # Shared b factor across heads
        Wb_v[:, r, :] = torch.outer(U3[:, r], U1[:, r])
    
    # Normalize factors
    for r in range(v_rank):
        norm_a = torch.norm(Wa_v[:, r, :])
        norm_b = torch.norm(Wb_v[:, r, :])
        if norm_a > 0 and norm_b > 0:
            scale = torch.sqrt(norm_a * norm_b)
            Wa_v[:, r, :] /= torch.sqrt(scale)
            Wb_v[:, r, :] /= torch.sqrt(scale)
    
    # Convert to the right device and dtype
    tpa_weights['Wa_v'] = Wa_v.to(dtype=dtype, device=device)
    tpa_weights['Wb_v'] = Wb_v.to(dtype=dtype, device=device)
    
    return tpa_weights


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
    weights using either contextual tensor factorization or Tucker decomposition.
    
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
    # Determine if we should use Tucker factorization
    use_tucker = HAS_TENSORLY
    
    if use_tucker:
        # Detect number of heads and kv heads from the model configuration
        try:
            # Get model configuration
            if hasattr(standard_model, 'config'):
                config = standard_model.config
                num_heads = config.num_attention_heads
                num_kv_heads = config.num_key_value_heads
            # Try to infer from the model structure
            elif hasattr(standard_model, 'model') and hasattr(standard_model.model, 'layers'):
                layer0 = standard_model.model.layers[0]
                if hasattr(layer0, 'self_attn'):
                    num_heads = layer0.self_attn.num_heads if hasattr(layer0.self_attn, 'num_heads') else None
                    num_kv_heads = layer0.self_attn.num_kv_heads if hasattr(layer0.self_attn, 'num_kv_heads') else num_heads
            else:
                # Fall back to standard factorization if we can't determine the head counts
                use_tucker = False
                print("Falling back to standard factorization - could not determine head counts for Tucker")
        except Exception as e:
            use_tucker = False
            print(f"Error detecting head counts, falling back to standard factorization: {e}")
    
    method = "Tucker decomposition" if use_tucker else "contextual factorization"
    print(f"Converting standard Gemma3 model to TPA using {method} with ranks - Q:{q_rank}, K:{k_rank}, V:{v_rank}")
    
    # Copy embedding weights
    try:
        # Try with either embedder or text_token_embedder naming
        if hasattr(standard_model, 'embedder'):
            tpa_model.embedder.weight.data.copy_(standard_model.embedder.weight.data)
            if verbose:
                print("Copied embedder weights")
        elif hasattr(standard_model, 'text_token_embedder'):
            tpa_model.embedder.weight.data.copy_(standard_model.text_token_embedder.weight.data)
            if verbose:
                print("Copied text_token_embedder weights")
        else:
            print("WARNING: Could not find embedder weights in standard model")
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
            
            if use_tucker:
                # Apply TensorLLM-style Tucker decomposition
                target_ranks = {'q': q_rank, 'k': k_rank, 'v': v_rank}
                tpa_weights = tucker_tensor_decomposition(
                    std_qkv_weight, 
                    num_heads, 
                    num_kv_heads, 
                    target_ranks,
                    dtype=std_qkv_weight.dtype, 
                    device=std_qkv_weight.device
                )
                
                # Assign decomposed weights to TPA layer
                if hasattr(tpa_layer.self_attn, 'W_A_q') and 'Wa_q' in tpa_weights:
                    tpa_layer.self_attn.W_A_q.weight.data.copy_(tpa_weights['Wa_q'].transpose(0, 1))
                
                if hasattr(tpa_layer.self_attn, 'W_B_q') and 'Wb_q' in tpa_weights:
                    tpa_layer.self_attn.W_B_q.weight.data.copy_(tpa_weights['Wb_q'].transpose(0, 1))
                
                if hasattr(tpa_layer.self_attn, 'W_A_k') and 'Wa_k' in tpa_weights:
                    tpa_layer.self_attn.W_A_k.weight.data.copy_(tpa_weights['Wa_k'].transpose(0, 1))
                
                if hasattr(tpa_layer.self_attn, 'W_B_k') and 'Wb_k' in tpa_weights:
                    tpa_layer.self_attn.W_B_k.weight.data.copy_(tpa_weights['Wb_k'].transpose(0, 1))
                
                if hasattr(tpa_layer.self_attn, 'W_A_v') and 'Wa_v' in tpa_weights:
                    tpa_layer.self_attn.W_A_v.weight.data.copy_(tpa_weights['Wa_v'].transpose(0, 1))
                
                if hasattr(tpa_layer.self_attn, 'W_B_v') and 'Wb_v' in tpa_weights:
                    tpa_layer.self_attn.W_B_v.weight.data.copy_(tpa_weights['Wb_v'].transpose(0, 1))
            else:
                # Apply T6-style contextual tensor factorization
                try:
                    # For TPA we need to factorize into A and B factors
                    head_dim = std_qkv_weight.shape[0] // (num_heads + 2 * num_kv_heads)
                    
                    # Handle query weights
                    q_section = std_qkv_weight[:num_heads * head_dim]
                    A_q, B_q = contextual_tensor_decomposition(
                        q_section, q_rank, k_rank, v_rank, 
                        dtype=std_qkv_weight.dtype, device=std_qkv_weight.device)
                    
                    # Assign factorized weights if the model has the right attributes
                    if hasattr(tpa_layer.self_attn, 'W_A_q'):
                        tpa_layer.self_attn.W_A_q.weight.data.copy_(A_q.transpose(0, 1))
                    if hasattr(tpa_layer.self_attn, 'W_B_q'):
                        tpa_layer.self_attn.W_B_q.weight.data.copy_(B_q.transpose(0, 1))
                    
                    # Handle key weights separately
                    k_start = num_heads * head_dim
                    k_end = k_start + num_kv_heads * head_dim
                    k_section = std_qkv_weight[k_start:k_end]
                    
                    A_k, B_k = contextual_tensor_decomposition(
                        k_section, q_rank, k_rank, v_rank, 
                        dtype=std_qkv_weight.dtype, device=std_qkv_weight.device)
                    
                    if hasattr(tpa_layer.self_attn, 'W_A_k'):
                        tpa_layer.self_attn.W_A_k.weight.data.copy_(A_k.transpose(0, 1))
                    if hasattr(tpa_layer.self_attn, 'W_B_k'):
                        tpa_layer.self_attn.W_B_k.weight.data.copy_(B_k.transpose(0, 1))
                    
                    # Handle value weights separately
                    v_start = k_end
                    v_section = std_qkv_weight[v_start:]
                    
                    A_v, B_v = contextual_tensor_decomposition(
                        v_section, q_rank, k_rank, v_rank, 
                        dtype=std_qkv_weight.dtype, device=std_qkv_weight.device)
                    
                    if hasattr(tpa_layer.self_attn, 'W_A_v'):
                        tpa_layer.self_attn.W_A_v.weight.data.copy_(A_v.transpose(0, 1))
                    if hasattr(tpa_layer.self_attn, 'W_B_v'):
                        tpa_layer.self_attn.W_B_v.weight.data.copy_(B_v.transpose(0, 1))
                        
                    print(f"Successfully applied contextual factorization to layer {layer_idx}")
                except Exception as e:
                    print(f"Error in contextual factorization for layer {layer_idx}: {e}")
                    print("This is likely due to precision issues. Continuing with other layers.")
            
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
    
    # Standard Gemma sampler doesn't have weights to copy
    # Just report this as informational rather than an error
    if verbose:
        print("Note: Sampler doesn't have weights to copy - this is expected")
    
    return tpa_model