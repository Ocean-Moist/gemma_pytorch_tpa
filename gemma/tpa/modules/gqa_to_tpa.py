"""
GQA to TPA conversion using Tucker decomposition.

This module provides functionality to convert Grouped Query Attention (GQA) weights
to Tensor Product Attention (TPA) format using TensorLLM-style Tucker decomposition.
"""

import torch
import torch.nn.functional as F
import math
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

from .svd_utils import HAS_TENSORLY

# Import TensorLy if available
if HAS_TENSORLY:
    import tensorly as tl
    from tensorly.decomposition import tucker
    tl.set_backend('pytorch')

def gqa_to_tpa_conversion(
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    o_weight: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    q_rank: int = 6,
    k_rank: int = 2,
    v_rank: int = 2,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    Convert GQA attention weights to TPA format using Tucker decomposition.
    
    This function implements the conversion approach described in the document,
    using TensorLLM-style Tucker decomposition while preserving the grouped structure.
    
    Args:
        q_weight: Query projection weight matrix [hidden_dim, num_heads * head_dim]
        k_weight: Key projection weight matrix [hidden_dim, num_kv_heads * head_dim]
        v_weight: Value projection weight matrix [hidden_dim, num_kv_heads * head_dim]
        o_weight: Output projection weight matrix [num_heads * head_dim, hidden_dim]
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads (groups)
        q_rank: Rank for query factorization
        k_rank: Rank for key factorization
        v_rank: Rank for value factorization
        dtype: Data type for the output tensors
        device: Device for computation
        
    Returns:
        Dictionary of factorized weights for TPA implementation
    """
    if not HAS_TENSORLY:
        raise ImportError("TensorLy is required for gqa_to_tpa_conversion")
    
    # Start timing
    tic = time.time()
    print("Starting GQA to TPA conversion using Tucker decomposition...")
    
    # Make sure tensors are on the right device and dtype for processing
    q_weight = q_weight.to(torch.float32)
    k_weight = k_weight.to(torch.float32)
    v_weight = v_weight.to(torch.float32)
    o_weight = o_weight.to(torch.float32).t()  # Transpose to match expected format
    
    # Get dimensions
    hidden_dim = q_weight.shape[0]
    head_dim = q_weight.shape[1] // num_heads
    
    # Check that dimensions are consistent with GQA
    assert head_dim * num_heads == q_weight.shape[1], "Query weight dimensions don't match num_heads"
    assert head_dim * num_kv_heads == k_weight.shape[1], "Key weight dimensions don't match num_kv_heads"
    assert head_dim * num_kv_heads == v_weight.shape[1], "Value weight dimensions don't match num_kv_heads"
    assert hidden_dim == o_weight.shape[0], "Output weight dimensions don't match hidden_dim"
    
    # STEP 1: Multi-head Tensorisation
    # Reshape the weights to better represent the heads
    q_weights_reshaped = q_weight.reshape(hidden_dim, num_heads, head_dim)
    k_weights_reshaped = k_weight.reshape(hidden_dim, num_kv_heads, head_dim)
    v_weights_reshaped = v_weight.reshape(hidden_dim, num_kv_heads, head_dim)
    o_weights_reshaped = o_weight.reshape(hidden_dim, num_heads, head_dim)
    
    # Create mapping from query heads to kv groups
    # For each query head, identify which kv head it should use
    heads_per_group = num_heads // num_kv_heads
    q_to_kv_mapping = torch.tensor([i // heads_per_group for i in range(num_heads)],
                                  device=q_weight.device)
    
    # Create the 4D tensor representation (hidden_dim, head_dim, 4, num_heads)
    # We'll store the query, key, value, and output weights for each head
    all_weights = []
    
    for h in range(num_heads):
        # For each query head, get the corresponding kv head
        kv_head = q_to_kv_mapping[h]
        
        # Stack the weights for this head
        head_tensor = torch.stack([
            q_weights_reshaped[:, h, :],                  # Query weights for this head
            k_weights_reshaped[:, kv_head, :],            # Key weights for the group
            v_weights_reshaped[:, kv_head, :],            # Value weights for the group
            o_weights_reshaped[:, h, :]                   # Output weights for this head
        ], dim=2)  # Stack along dimension 2
        
        all_weights.append(head_tensor)
    
    # Combine all heads into a single 4D tensor
    W_all = torch.stack(all_weights, dim=3)  # Shape: [hidden_dim, head_dim, 4, num_heads]
    
    # Ensure proper dimensions
    print(f"Tensorized weights shape: {W_all.shape}")
    
    # STEP 2: Tucker Decomposition with Shared Factor Matrices
    # Set target ranks
    R1 = q_rank  # Rank for the model dimension
    R2 = q_rank  # Rank for the head dimension
    R3 = q_rank  # Rank for the QKV distinction
    
    # Adjust ranks if needed to prevent issues
    R1 = min(R1, W_all.shape[0])
    R2 = min(R2, W_all.shape[1])
    R3 = min(R3, W_all.shape[2])
    
    print(f"Applying Tucker decomposition with ranks: ({R1}, {R2}, {R3})")
    
    try:
        # Apply Tucker decomposition to the full tensor
        # This enforces shared factor matrices across all heads
        print(f"Tensor shape before Tucker decomposition: {W_all.shape}")
        print(f"Target ranks: {[R1, R2, R3, num_heads]}")
        
        # Set up timing
        decomp_start = time.time()
        print("Starting Tucker decomposition...")
        
        # Add verbose parameter to tucker
        core, factors = tucker(W_all, rank=[R1, R2, R3, num_heads], tol=1e-4, verbose=True)
        
        decomp_end = time.time()
        print(f"Tucker decomposition took {decomp_end - decomp_start:.2f} seconds")
        
        # Extract the factor matrices
        U1 = factors[0]  # For hidden dimension
        U2 = factors[1]  # For head dimension
        U3 = factors[2]  # For QKV distinction
        # No need for factor 3 since we'll keep each head separate
        
        print(f"Tucker decomposition complete. Core shape: {core.shape}")
        print(f"Factor shapes: {[f.shape for f in factors]}")
        
        # Check for NaN or infinite values
        if torch.isnan(core).any() or torch.isinf(core).any():
            print("WARNING: Core tensor contains NaN or Inf values!")
        
        for i, factor in enumerate(factors):
            if torch.isnan(factor).any() or torch.isinf(factor).any():
                print(f"WARNING: Factor {i} contains NaN or Inf values!")
    
    except Exception as e:
        print(f"Tucker decomposition failed: {e}")
        print("Falling back to separate decompositions for Q, K, V")
        import traceback
        traceback.print_exc()
        
        # Fallback: Decompose Q, K, V separately
        q_tensors = W_all[:, :, 0, :]  # Shape: [hidden_dim, head_dim, num_heads]
        k_tensors = W_all[:, :, 1, :]
        v_tensors = W_all[:, :, 2, :]
        o_tensors = W_all[:, :, 3, :]
        
        # Apply tucker to each separately
        q_core, q_factors = tucker(q_tensors, rank=[R1, R2, num_heads])
        k_core, k_factors = tucker(k_tensors, rank=[k_rank, k_rank, num_kv_heads])
        v_core, v_factors = tucker(v_tensors, rank=[v_rank, v_rank, num_kv_heads])
        o_core, o_factors = tucker(o_tensors, rank=[R1, R2, num_heads])
        
        # Create result using the separate decompositions
        result = {}
        result["Q_core"] = q_core.to(dtype=dtype, device=device)
        result["Q_hidden_factor"] = q_factors[0].to(dtype=dtype, device=device)
        result["Q_dim_factor"] = q_factors[1].to(dtype=dtype, device=device)
        result["Q_head_factor"] = torch.eye(num_heads, device=device).to(dtype=dtype)
        
        result["K_core"] = k_core.to(dtype=dtype, device=device)
        result["K_hidden_factor"] = k_factors[0].to(dtype=dtype, device=device)
        result["K_dim_factor"] = k_factors[1].to(dtype=dtype, device=device)
        result["K_head_factor"] = torch.eye(num_kv_heads, device=device).to(dtype=dtype)
        
        result["V_core"] = v_core.to(dtype=dtype, device=device)
        result["V_hidden_factor"] = v_factors[0].to(dtype=dtype, device=device)
        result["V_dim_factor"] = v_factors[1].to(dtype=dtype, device=device)
        result["V_head_factor"] = torch.eye(num_kv_heads, device=device).to(dtype=dtype)
        
        toc = time.time()
        print(f"Fallback conversion complete in {toc - tic:.2f} seconds")
        
        return result
    
    # STEP 3: Map Tucker Factors to TPA Parameters
    result = {}
    
    # Extract factor matrices for query, key, value parts
    # U3 contains factors for Q, K, V, O: shape [4, R3]
    q_factor = U3[0].reshape(-1, 1)  # [R3, 1]
    k_factor = U3[1].reshape(-1, 1)  # [R3, 1]
    v_factor = U3[2].reshape(-1, 1)  # [R3, 1]
    
    # For query:
    # Create the core tensor with QKV factors
    q_core_shape = (R1, R2, num_heads)
    q_core = torch.zeros(q_core_shape, dtype=core.dtype, device=core.device)
    
    # Project the core using q_factor
    for h in range(num_heads):
        q_core[:, :, h] = torch.tensordot(core[:, :, :, h], q_factor, dims=[[2], [0]]).squeeze(-1)
    
    # For keys and values, we need to handle the group structure
    k_core_shape = (R1, R2, num_kv_heads)
    v_core_shape = (R1, R2, num_kv_heads)
    
    k_core = torch.zeros(k_core_shape, dtype=core.dtype, device=core.device)
    v_core = torch.zeros(v_core_shape, dtype=core.dtype, device=core.device)
    
    # Project the core using k_factor and v_factor, preserving the group structure
    for g in range(num_kv_heads):
        # Find all query heads that map to this KV group
        group_q_heads = [h for h in range(num_heads) if q_to_kv_mapping[h] == g]
        
        # Average the cores from all heads in this group
        group_core = torch.zeros_like(core[:, :, :, 0])
        for h in group_q_heads:
            group_core += core[:, :, :, h]
        group_core /= len(group_q_heads)
        
        # Project using k_factor and v_factor
        k_core[:, :, g] = torch.tensordot(group_core, k_factor, dims=[[2], [0]]).squeeze(-1)
        v_core[:, :, g] = torch.tensordot(group_core, v_factor, dims=[[2], [0]]).squeeze(-1)
    
    # Store the results
    # We use tensordot instead of matmul to project from the higher-dimensional space
    
    # For queries:
    result["Q_core"] = q_core.to(dtype=dtype, device=device)
    result["Q_hidden_factor"] = U1.to(dtype=dtype, device=device)
    result["Q_dim_factor"] = U2.to(dtype=dtype, device=device)
    result["Q_head_factor"] = torch.eye(num_heads, device=device).to(dtype=dtype)
    
    # For keys:
    result["K_core"] = k_core.to(dtype=dtype, device=device)
    result["K_hidden_factor"] = U1.to(dtype=dtype, device=device)
    result["K_dim_factor"] = U2.to(dtype=dtype, device=device)
    result["K_head_factor"] = torch.eye(num_kv_heads, device=device).to(dtype=dtype)
    
    # For values:
    result["V_core"] = v_core.to(dtype=dtype, device=device)
    result["V_hidden_factor"] = U1.to(dtype=dtype, device=device)
    result["V_dim_factor"] = U2.to(dtype=dtype, device=device)
    result["V_head_factor"] = torch.eye(num_kv_heads, device=device).to(dtype=dtype)
    
    # Calculate the weight matrices for Wa and Wb in TPA format to verify correctness
    
    # Calculate the weight matrices for Wa and Wb in TPA format
    # For TPA implementation:
    # A_q = W_A_q(x) - shape [batch, seq, heads, q_rank]
    # B_q = W_B_q(x) - shape [batch, seq, q_rank, head_dim]
    
    # Expand dimensions for TPA format
    W_A_q_expanded = torch.zeros((hidden_dim, num_heads * q_rank), device=q_weight.device, dtype=torch.float32)
    W_A_k_expanded = torch.zeros((hidden_dim, num_kv_heads * k_rank), device=k_weight.device, dtype=torch.float32)
    W_A_v_expanded = torch.zeros((hidden_dim, num_kv_heads * v_rank), device=v_weight.device, dtype=torch.float32)
    
    # The proper way to create the expanded Wa is to repeat for each head
    for h in range(num_heads):
        W_A_q = U1 @ q_core[:, :, h]
        W_A_q_expanded[:, h*q_rank:(h+1)*q_rank] = W_A_q
        
    for g in range(num_kv_heads):
        W_A_k = U1 @ k_core[:, :, g]
        W_A_v = U1 @ v_core[:, :, g]
        W_A_k_expanded[:, g*k_rank:(g+1)*k_rank] = W_A_k
        W_A_v_expanded[:, g*v_rank:(g+1)*v_rank] = W_A_v
    
    # Create the B matrices which need reshaping for TPA format
    W_B_q = U2.t()  # [R2, head_dim]
    W_B_k = U2.t()  # [R2, head_dim]
    W_B_v = U2.t()  # [R2, head_dim]
    
    # Reshape B matrices for TPA format (head-interleaved)
    W_B_q_reshaped = W_B_q.repeat(num_heads, 1)  # [num_heads*R2, head_dim]
    W_B_k_reshaped = W_B_k.repeat(num_kv_heads, 1)  # [num_kv_heads*R2, head_dim]
    W_B_v_reshaped = W_B_v.repeat(num_kv_heads, 1)  # [num_kv_heads*R2, head_dim]
    
    # Add expanded matrices to result
    result["W_A_q"] = W_A_q_expanded.to(dtype=dtype, device=device)
    result["W_A_k"] = W_A_k_expanded.to(dtype=dtype, device=device)
    result["W_A_v"] = W_A_v_expanded.to(dtype=dtype, device=device)
    
    result["W_B_q"] = W_B_q_reshaped.to(dtype=dtype, device=device)
    result["W_B_k"] = W_B_k_reshaped.to(dtype=dtype, device=device)
    result["W_B_v"] = W_B_v_reshaped.to(dtype=dtype, device=device)
    
    # Verify reconstruction error
    # This is important to ensure the factorization is accurate
    
    # Reconstruct query weights
    q_recon = torch.zeros_like(q_weights_reshaped)
    for h in range(num_heads):
        q_head_A = W_A_q_expanded[:, h*q_rank:(h+1)*q_rank]
        q_head_B = W_B_q_reshaped[h*q_rank:(h+1)*q_rank, :]
        q_recon[:, h, :] = (q_head_A @ q_head_B).reshape(hidden_dim, head_dim)
    
    # Reconstruct key and value weights with the group structure
    k_recon = torch.zeros_like(k_weights_reshaped)
    v_recon = torch.zeros_like(v_weights_reshaped)
    
    for g in range(num_kv_heads):
        k_group_A = W_A_k_expanded[:, g*k_rank:(g+1)*k_rank]
        k_group_B = W_B_k_reshaped[g*k_rank:(g+1)*k_rank, :]
        k_recon[:, g, :] = (k_group_A @ k_group_B).reshape(hidden_dim, head_dim)
        
        v_group_A = W_A_v_expanded[:, g*v_rank:(g+1)*v_rank]
        v_group_B = W_B_v_reshaped[g*v_rank:(g+1)*v_rank, :]
        v_recon[:, g, :] = (v_group_A @ v_group_B).reshape(hidden_dim, head_dim)
    
    # Reshape for error calculation
    q_recon = q_recon.reshape(hidden_dim, -1)
    k_recon = k_recon.reshape(hidden_dim, -1)
    v_recon = v_recon.reshape(hidden_dim, -1)
    
    # Calculate relative errors
    q_err = torch.norm(q_recon - q_weight) / torch.norm(q_weight)
    k_err = torch.norm(k_recon - k_weight) / torch.norm(k_weight)
    v_err = torch.norm(v_recon - v_weight) / torch.norm(v_weight)
    
    print(f"Reconstruction relative errors - Q: {q_err:.4f}, K: {k_err:.4f}, V: {v_err:.4f}")
    
    toc = time.time()
    print(f"GQA to TPA conversion complete in {toc - tic:.2f} seconds")
    
    return result


def convert_gqa_model_to_tpa(model, q_rank=6, k_rank=2, v_rank=2, dtype=torch.float16, device="cuda"):
    """
    Convert a GQA model to TPA format by applying the conversion to each attention layer.
    
    Args:
        model: The input model with GQA
        q_rank: Rank for query factorization
        k_rank: Rank for key factorization
        v_rank: Rank for value factorization
        dtype: Data type for output tensors
        device: Device for computation
        
    Returns:
        Model with converted weights
    """
    if not HAS_TENSORLY:
        raise ImportError("TensorLy is required for convert_gqa_model_to_tpa")
    
    print("Converting GQA model to TPA format...")
    
    # Add timing and layer counting
    import time
    start_time = time.time()
    layers_converted = 0
    attention_modules_found = 0
    
    # Debug model structure
    print(f"Model type: {type(model).__name__}")
    print("Searching for attention modules...")
    
    # GemmaAttention modules are found in different structures based on the model architecture
    # We'll handle both GemmaForCausalLM and Gemma3ForMultimodalLM structures
    attention_modules = []
    
    # Check if using standard GemmaForCausalLM structure
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        print("Detected standard GemmaForCausalLM structure")
        for i, layer in enumerate(model.model.layers):
            if hasattr(layer, "self_attn"):
                attention_module = layer.self_attn
                if hasattr(attention_module, "qkv_proj"):
                    # Standard Gemma attention with combined QKV projection
                    print(f"  Found QKV-combined attention module in layer {i}")
                    # For standard Gemma attention, we need to split the QKV projection
                    # This requires additional handling for the weights
                    attention_modules.append((f"model.layers.{i}.self_attn", attention_module, "combined_qkv"))
                    attention_modules_found += 1
                elif hasattr(attention_module, "q_proj") and hasattr(attention_module, "k_proj") and hasattr(attention_module, "v_proj"):
                    # Already split Q,K,V projections
                    print(f"  Found split-QKV attention module in layer {i}")
                    attention_modules.append((f"model.layers.{i}.self_attn", attention_module, "split_qkv"))
                    attention_modules_found += 1
    
    # Check if using Gemma TPA model structure
    elif hasattr(model, "model") and hasattr(model.model, "layers") and hasattr(model.model.layers[0], "self_attn"):
        print("Detected Gemma TPA model structure")
        for i, layer in enumerate(model.model.layers):
            attention_module = layer.self_attn
            if hasattr(attention_module, "W_A_q") and hasattr(attention_module, "W_B_q"):
                # Already a TPA module
                print(f"  Layer {i} already using TPA attention - skipping")
            elif hasattr(attention_module, "q_proj") and hasattr(attention_module, "k_proj") and hasattr(attention_module, "v_proj"):
                print(f"  Found split-QKV attention module in layer {i}")
                attention_modules.append((f"model.model.layers.{i}.self_attn", attention_module, "split_qkv"))
                attention_modules_found += 1
    
    # If no attention modules found yet, try to find them in named_modules
    if attention_modules_found == 0:
        print("No attention modules found in standard structure, searching all modules...")
        for name, module in model.named_modules():
            if hasattr(module, "qkv_proj") and hasattr(module, "o_proj"):
                # Standard Gemma attention with combined QKV projection
                print(f"  Found combined-QKV attention module: {name}")
                attention_modules.append((name, module, "combined_qkv"))
                attention_modules_found += 1
            elif hasattr(module, "q_proj") and hasattr(module, "k_proj") and hasattr(module, "v_proj"):
                # Split Q,K,V projections
                print(f"  Found split-QKV attention module: {name}")
                attention_modules.append((name, module, "split_qkv"))
                attention_modules_found += 1
    
    if attention_modules_found == 0:
        print("ERROR: No attention modules found in the model!")
        print("Model structure may not be compatible with this conversion method")
        print("Expected structure: model.model.layers[i].self_attn")
        print("Showing all top-level attributes of model:")
        for attr_name in dir(model):
            if not attr_name.startswith('_'):
                print(f"  {attr_name}")
        
        if hasattr(model, "model"):
            print("Showing all attributes of model.model:")
            for attr_name in dir(model.model):
                if not attr_name.startswith('_'):
                    print(f"  {attr_name}")
        
        return model
        
    print(f"Found {attention_modules_found} attention modules to convert")
    
    # Process each module
    for name, module, module_type in attention_modules:
        print(f"Converting attention layer: {name} (type: {module_type})")
        layer_start = time.time()
        
        # Get or derive the number of heads and kv heads
        if hasattr(module, "num_heads") and hasattr(module, "num_key_value_heads"):
            num_heads = module.num_heads
            num_kv_heads = module.num_key_value_heads
            print(f"  GQA structure verified: {num_heads} heads, {num_kv_heads} KV heads")
        else:
            # Try to infer from model config
            if hasattr(model, "config"):
                num_heads = getattr(model.config, "num_attention_heads", None)
                num_kv_heads = getattr(model.config, "num_key_value_heads", num_heads)
                if num_heads is not None:
                    print(f"  Inferred from config: {num_heads} heads, {num_kv_heads} KV heads")
                else:
                    print(f"  ERROR: Could not determine num_heads from config")
                    continue
            else:
                print(f"  ERROR: Module {name} does not have GQA structure and no config found")
                continue
        
        # Check if apply_factorized_weights exists
        if hasattr(module, "apply_factorized_weights"):
            print(f"  Module supports factorized weights")
        else:
            print(f"  WARNING: Module does not support applying factorized weights")
            print(f"  Adding apply_factorized_weights method dynamically")
            
            # Dynamically add method to apply_factorized_weights if it doesn't exist
            def apply_factorized_weights_fn(self, weights_dict):
                """Dynamically added method to apply factorized weights to attention module"""
                # Set factorized flag
                self.use_factorized_weights = True
                
                # Store the factorized weights
                for key, weight in weights_dict.items():
                    setattr(self, key, nn.Parameter(weight))
                
                # Remember original weights were factorized
                self.original_weights_replaced = True
            
            # Bind the method to the module
            import types
            module.apply_factorized_weights = types.MethodType(apply_factorized_weights_fn, module)
            module.use_factorized_weights = False
            
        try:
            # Handle different module types
            if module_type == "combined_qkv":
                # For combined QKV projection, we need to extract the separate weights
                qkv_weight = module.qkv_proj.weight
                q_size = num_heads * module.head_dim
                kv_size = num_kv_heads * module.head_dim
                
                # Split the combined weights
                q_weight, k_weight, v_weight = qkv_weight.split(
                    [q_size, kv_size, kv_size], dim=0
                )
                
                o_weight = module.o_proj.weight
                
                print(f"  Split combined QKV projection: Q: {q_weight.shape}, K: {k_weight.shape}, V: {v_weight.shape}")
            
            elif module_type == "split_qkv":
                # Already has separate Q, K, V projections
                q_weight = module.q_proj.weight
                k_weight = module.k_proj.weight
                v_weight = module.v_proj.weight
                o_weight = module.o_proj.weight
                
                print(f"  Weight dimensions - Q: {q_weight.shape}, K: {k_weight.shape}, V: {v_weight.shape}, O: {o_weight.shape}")
            
            else:
                print(f"  ERROR: Unknown module type: {module_type}")
                continue
            
            # Apply GQA to TPA conversion
            print(f"  Starting tensor decomposition for layer {name}...")
            decomp_start = time.time()
            
            factorized_weights = gqa_to_tpa_conversion(
                q_weight, k_weight, v_weight, o_weight,
                num_heads, num_kv_heads,
                q_rank, k_rank, v_rank,
                dtype, device
            )
            
            decomp_end = time.time()
            print(f"  Decomposition completed in {decomp_end - decomp_start:.2f} seconds")
            
            # Check if factorized weights were created
            if not factorized_weights:
                print(f"  ERROR: Factorization returned empty result")
                continue
                
            print(f"  Factorized weights keys: {list(factorized_weights.keys())}")
            
            # Apply factorized weights to the model
            if hasattr(module, "apply_factorized_weights"):
                print(f"  Applying factorized weights to module...")
                module.apply_factorized_weights(factorized_weights)
                layers_converted += 1
            else:
                print(f"  ERROR: Module {name} does not support applying factorized weights")
        
        except Exception as e:
            print(f"  ERROR in layer {name}: {e}")
            import traceback
            traceback.print_exc()
        
        layer_end = time.time()
        print(f"  Layer conversion took {layer_end - layer_start:.2f} seconds")
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"GQA to TPA conversion complete: {layers_converted}/{attention_modules_found} layers converted in {total_time:.2f} seconds")
    return model