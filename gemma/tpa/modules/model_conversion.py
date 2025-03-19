"""
Model conversion utilities for Tensor Product Attention.

This module provides functions for converting standard transformer models
to TPA (Tensor Product Attention) models.
"""

import torch
from typing import Dict, Optional

from .tensor_factorization import contextual_tensor_decomposition, tucker_tensor_decomposition

def apply_contextual_tensor_decomposition(model, q_rank=6, k_rank=2, v_rank=2):
    """
    Apply contextual tensor decomposition to all attention layers in a model.
    
    Args:
        model: Input model
        q_rank: Rank for query projection
        k_rank: Rank for key projection
        v_rank: Rank for value projection
        
    Returns:
        Factorized model
    """
    # Factorize all attention layers in the model
    for name, module in model.named_modules():
        if hasattr(module, "q_proj") and hasattr(module, "k_proj") and hasattr(module, "v_proj"):
            print(f"Factorizing attention layer: {name}")
            
            # Extract weight matrices
            q_weight = module.q_proj.weight
            k_weight = module.k_proj.weight
            v_weight = module.v_proj.weight
            
            # Apply factorization
            factorized_weights = contextual_tensor_decomposition(
                [q_weight, k_weight, v_weight],
                q_rank=q_rank,
                k_rank=k_rank,
                v_rank=v_rank,
                dtype=q_weight.dtype,
                device=q_weight.device
            )
            
            # Apply factorized weights to model
            # (Implementation depends on the specific model architecture)
            if hasattr(module, "apply_factorized_weights"):
                module.apply_factorized_weights(factorized_weights)
            else:
                print(f"Warning: Module {name} does not support applying factorized weights")
    
    return model

def convert_from_standard_weights(standard_model, tpa_model, q_rank=6, k_rank=2, v_rank=2, verbose=True):
    """
    Convert a standard transformer model to a TPA model.
    
    Args:
        standard_model: Source model with standard attention
        tpa_model: Target TPA model
        q_rank: Rank for query projection
        k_rank: Rank for key projection
        v_rank: Rank for value projection
        verbose: Whether to print verbose information
        
    Returns:
        Converted TPA model
    """
    if verbose:
        print("Converting standard model to TPA model...")
    
    # Copy non-attention weights directly
    for tpa_name, tpa_param in tpa_model.named_parameters():
        if "attention" not in tpa_name or any(x in tpa_name for x in ["spatial", "context", "core"]):
            continue
            
        # Find corresponding parameter in standard model
        std_name = tpa_name
        if std_name in standard_model.state_dict():
            if verbose:
                print(f"Copying {std_name} -> {tpa_name}")
            tpa_param.data.copy_(standard_model.state_dict()[std_name])
    
    # Apply factorization to attention layers
    for (std_name, std_module), (tpa_name, tpa_module) in zip(
        standard_model.named_modules(), tpa_model.named_modules()
    ):
        if hasattr(std_module, "q_proj") and hasattr(std_module, "k_proj") and hasattr(std_module, "v_proj"):
            if verbose:
                print(f"Factorizing attention layer: {std_name} -> {tpa_name}")
                
            # Extract weight matrices
            q_weight = std_module.q_proj.weight
            k_weight = std_module.k_proj.weight
            v_weight = std_module.v_proj.weight
            
            # Get head dimensions
            if hasattr(std_module, "num_heads") and hasattr(std_module, "num_key_value_heads"):
                num_heads = std_module.num_heads
                num_kv_heads = std_module.num_key_value_heads
                
                # Determine factorization approach based on model type
                if hasattr(tpa_module, "use_tucker") and tpa_module.use_tucker:
                    # Use Tucker decomposition
                    combined_weight = torch.cat([q_weight, k_weight, v_weight], dim=1)
                    
                    target_ranks = {
                        "q_rank": q_rank,
                        "k_rank": k_rank,
                        "v_rank": v_rank
                    }
                    
                    factorized_weights = tucker_tensor_decomposition(
                        combined_weight,
                        num_heads=num_heads,
                        num_kv_heads=num_kv_heads,
                        target_ranks=target_ranks,
                        dtype=q_weight.dtype,
                        device=q_weight.device
                    )
                else:
                    # Use contextual factorization
                    factorized_weights = contextual_tensor_decomposition(
                        [q_weight, k_weight, v_weight],
                        q_rank=q_rank,
                        k_rank=k_rank,
                        v_rank=v_rank,
                        dtype=q_weight.dtype,
                        device=q_weight.device
                    )
                
                # Apply factorized weights
                if hasattr(tpa_module, "apply_factorized_weights"):
                    tpa_module.apply_factorized_weights(factorized_weights)
                else:
                    print(f"Warning: TPA module {tpa_name} does not support applying factorized weights")
    
    if verbose:
        print("Model conversion complete")
    
    return tpa_model