#!/usr/bin/env python3
"""
Inference script for TPA with Gemma models.
This script loads an already converted TPA model and runs inference.
"""

import argparse
import os
import time
import torch
import traceback
from pathlib import Path

from gemma import tokenizer, config
from gemma.tpa.gemma3_tpa_model import Gemma3ForMultimodalLMwithTPA
import torch.serialization

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with TPA Gemma model")
    parser.add_argument("--model", type=str, required=True, help="Path to TPA model file")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/tokenizer.model", help="Path to tokenizer")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--prompt", type=str, default="What are large language models?", help="Test prompt")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling parameter")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}")
    gemma_tok = tokenizer.Tokenizer(args.tokenizer)
    
    try:
        # Load TPA model
        print(f"Loading TPA model from {args.model}")
        
        # Handle PyTorch 2.6+ security restrictions
        print("Adding GemmaConfig to safe globals for PyTorch loading...")
        # Add GemmaConfig to the safe globals list
        torch.serialization.add_safe_globals([config.GemmaConfig])
        
        try:
            # First try with weights_only=True (safe mode)
            checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
            print("Model loaded with weights_only=True")
        except Exception as e:
            print(f"Warning: Failed to load with weights_only=True: {e}")
            print("Trying with weights_only=False (less secure, but needed for older models)")
            checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
            print("Model loaded with weights_only=False")
        
        # Get model config
        if "config" in checkpoint:
            model_config = checkpoint["config"]
            print("Using config from checkpoint")
        else:
            print("ERROR: Model file doesn't contain config information")
            return
        
        # Create TPA model
        tpa_model = Gemma3ForMultimodalLMwithTPA(model_config)
        tpa_model.tokenizer = gemma_tok
        
        # Load state dict
        print("Loading model weights...")
        tpa_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        
        # Move to device
        tpa_model = tpa_model.to(device).eval()
        print(f"Model loaded successfully to {device}")
        
        # Print model configuration info
        print(f"Model configuration:")
        print(f"  - Q rank: {model_config.q_rank}")
        print(f"  - K rank: {model_config.k_rank}")
        print(f"  - V rank: {model_config.v_rank}")
        print(f"  - Sliding window: {model_config.sliding_window_size}")
        
        # Run inference
        print(f"Running inference with prompt: '{args.prompt}'")
        
        with torch.no_grad():
            # Format prompt for model
            prompt = [(args.prompt,)]
            
            # Generate text
            gen_start = time.time()
            outputs = tpa_model.generate(
                prompts=prompt,
                device=device,
                output_len=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k
            )
            gen_time = time.time() - gen_start
            
            # Print results
            print("\n" + "="*50)
            print(f"PROMPT: {args.prompt}")
            print(f"RESULT: {outputs[0]}")
            print("="*50)
            print(f"Generation time: {gen_time:.2f} seconds")
            print(f"Generation speed: {args.max_tokens/gen_time:.2f} tokens/second")
            
            # Calculate memory usage
            if device.type == "cuda":
                mem_allocated = torch.cuda.memory_allocated(device) / (1024**3)
                mem_reserved = torch.cuda.memory_reserved(device) / (1024**3)
                print(f"Memory allocated: {mem_allocated:.2f} GB")
                print(f"Memory reserved: {mem_reserved:.2f} GB")
                
                # Calculate theoretical KV cache sizes
                seq_len = model_config.max_position_embeddings
                num_heads = model_config.num_attention_heads
                head_dim = model_config.head_dim
                bytes_per_element = 2  # bfloat16
                
                std_kv_size = 2 * seq_len * num_heads * head_dim * bytes_per_element
                tpa_kv_size = (model_config.k_rank + model_config.v_rank) * (num_heads + head_dim) * seq_len * bytes_per_element
                
                reduction = std_kv_size / tpa_kv_size
                print(f"\nMemory efficiency:")
                print(f"Standard KV cache: {std_kv_size/(1024**3):.2f} GB")
                print(f"TPA KV cache: {tpa_kv_size/(1024**3):.2f} GB")
                print(f"Reduction ratio: {reduction:.2f}x")
        
        print("\nTPA inference completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()