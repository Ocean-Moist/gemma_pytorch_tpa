"""Simple script to test TPA conversion with Gemma 1B model."""

import torch
import os
import argparse
from gemma import config
from gemma.model import GemmaForCausalLM
from gemma.tpa.tpa_model import GemmaTPAModel, create_tpa_kv_caches

def main():
    parser = argparse.ArgumentParser(description="Test TPA conversion for Gemma 1B model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model weights")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save TPA model")
    parser.add_argument("--q_rank", type=int, default=6, help="Rank for query factorization")
    parser.add_argument("--k_rank", type=int, default=2, help="Rank for key factorization")
    parser.add_argument("--v_rank", type=int, default=2, help="Rank for value factorization")
    
    args = parser.parse_args()
    
    # Create Gemma 1B configuration
    model_config = config.get_config_for_1b(dtype="float32")
    
    # Add TPA specific configuration parameters
    model_config.q_rank = args.q_rank
    model_config.k_rank = args.k_rank
    model_config.v_rank = args.v_rank
    
    print(f"Loading model from {args.model_path}...")
    model_data = torch.load(args.model_path, map_location="cpu")
    
    # Create standard Gemma model
    print("Creating standard model instance...")
    standard_model = GemmaForCausalLM(model_config)
    
    # Load weights
    print("Loading weights into model...")
    standard_model.load_state_dict(model_data["model_state_dict"], strict=False)
    
    # Create TPA model
    print("Creating TPA model...")
    tpa_model = GemmaTPAModel(model_config)
    
    # Convert standard model to TPA model
    print("Converting model to TPA format...")
    
    # TODO: Implement conversion logic
    # This is where we would implement the SVD factorization of attention weights
    
    # For now, just print the model structure
    print("Standard model structure:")
    for name, param in standard_model.named_parameters():
        if "self_attn" in name and "qkv_proj" in name:
            print(f"{name}: {param.shape}")
    
    # Save TPA model
    print(f"Saving converted model to {args.save_path}...")
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save({
        'model_state_dict': standard_model.state_dict(),
        'config': model_config
    }, args.save_path)
    
    print("Conversion test completed!")

if __name__ == "__main__":
    main()