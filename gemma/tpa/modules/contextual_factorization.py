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
    from tensorly.decomposition import tucker
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
                print("LAPACK error detected, using TruncatedSVD instead")
                
                # Get matrix dimensions
                m, n = a.shape
                
                # Estimate rank (can be adjusted based on requirements)
                k = min(m, n, 256)  # Use a reasonable default rank
                
                if compute_uv:
                    try:
                        # Use sklearn's TruncatedSVD 
                        from sklearn.decomposition import TruncatedSVD
                        
                        svd = TruncatedSVD(n_components=k, random_state=42)
                        Vt = svd.fit_transform(a.T).T
                        s = svd.singular_values_
                        U = svd.components_.T
                        
                        if full_matrices:
                            # Pad U and Vt to match full matrices if needed
                            if m > k and U.shape[1] < m:
                                pad_U = np.zeros((m, m-k))
                                U = np.hstack((U, pad_U))
                            if n > k and Vt.shape[0] < n:
                                pad_Vt = np.zeros((n-k, n))
                                Vt = np.vstack((Vt, pad_Vt))
                        
                        print(f"TruncatedSVD successful with rank {k}")
                        return U, s, Vt
                    except (ImportError, Exception) as svd_error:
                        print(f"TruncatedSVD failed: {svd_error}, falling back to NumPy's SVD")
                        return np.linalg.svd(a, full_matrices=full_matrices, compute_uv=compute_uv)
                else:
                    # If we only need singular values, use standard numpy SVD
                    # and truncate the result
                    s = np.linalg.svd(a, full_matrices=False, compute_uv=False)
                    return s[:k]
            else:
                raise
    
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
    
    # Convert weights to float32 for numpy compatibility
    # Numpy doesn't support bfloat16 directly
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
        print(f"Applying Tucker decomposition with ranks: {rank}")
        core, factors = tucker(wq_tensor, rank=rank)
    except Exception as e:
        print(f"Warning: Error in Tucker decomposition: {e}")
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
    # Make sure we have float32 for numpy compatibility
    if k_weight.dtype == torch.bfloat16:
        k_weight = k_weight.to(torch.float32)
    
    # Reshape to 3D tensor [head_dim, num_kv_heads, input_dim]
    # Keep as PyTorch tensor rather than converting to numpy
    wk_tensor = k_weight.reshape(head_dim, num_kv_heads, input_dim).cpu()
    
    # Apply Tucker decomposition to key weights
    rank = [k_rank, None, k_rank]  # Rank for dimensions
    try:
        print(f"Applying Tucker decomposition with ranks: {rank}")
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
    # Make sure we have float32 for numpy compatibility
    if v_weight.dtype == torch.bfloat16:
        v_weight = v_weight.to(torch.float32)
    
    # Reshape to 3D tensor [head_dim, num_kv_heads, input_dim]
    # Keep as PyTorch tensor rather than converting to numpy
    wv_tensor = v_weight.reshape(head_dim, num_kv_heads, input_dim).cpu()
    
    # Apply Tucker decomposition to value weights
    rank = [v_rank, None, v_rank]  # Rank for dimensions
    try:
        print(f"Applying Tucker decomposition with ranks: {rank}")
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