#!/usr/bin/env python3
"""
Test script for validating TPA implementation with Gemma 1B model.
"""

import argparse
import os
import time
import torch
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm

from gemma import config, tokenizer
from gemma.model import GemmaForCausalLM
from gemma.tpa.gemma3_tpa_model import Gemma3ForMultimodalLMwithTPA

def parse_args():
    parser = argparse.ArgumentParser(description="Test TPA implementation with Gemma 1B model")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to Gemma 1B checkpoint")
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer/tokenizer.model", help="Path to tokenizer model")
    parser.add_argument("--output-dir", type=str, default="tpa_models", help="Directory to save TPA model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--q-rank", type=int, default=6, help="Rank for query factorization")
    parser.add_argument("--k-rank", type=int, default=2, help="Rank for key factorization")
    parser.add_argument("--v-rank", type=int, default=2, help="Rank for value factorization")
    parser.add_argument("--test-prompt", type=str, default="What are large language models?", help="Test prompt for inference")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}")
    gemma_tok = tokenizer.Tokenizer(args.tokenizer_path)
    
    # Create Gemma 1B config
    print("Creating Gemma 1B configuration")
    model_config = config.get_config_for_1b(dtype="float32" if device.type == "cpu" else "bfloat16")
    model_config.vision_config = None
    model_config.tokenizer = args.tokenizer_path
    
    # Add TPA specific configuration
    model_config.q_rank = args.q_rank
    model_config.k_rank = args.k_rank
    model_config.v_rank = args.v_rank
    model_config.quant = False
    
    # Ensure sliding window is set (needed for attention masking)
    if not hasattr(model_config, 'sliding_window_size'):
        model_config.sliding_window_size = 4096
        print(f"Setting sliding_window_size to {model_config.sliding_window_size}")
    
    # Load standard model
    print(f"Loading standard Gemma 1B model from {args.ckpt_path}")
    start_time = time.time()
    
    try:
        checkpoint = torch.load(args.ckpt_path, map_location="cpu")
        
        # Create standard model
        standard_model = GemmaForCausalLM(model_config)
        standard_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        standard_model.eval()
        
        load_time = time.time() - start_time
        print(f"Standard model loaded in {load_time:.2f} seconds")
        
        # Create TPA model
        print("Creating TPA model...")
        tpa_model = Gemma3ForMultimodalLMwithTPA(model_config)
        tpa_model.tokenizer = gemma_tok
        
        # Convert standard model to TPA
        print(f"Converting standard model to TPA with q_rank={args.q_rank}, k_rank={args.k_rank}, v_rank={args.v_rank}...")
        convert_start = time.time()
        tpa_model.convert_from_standard_weights(standard_model)
        convert_time = time.time() - convert_start
        print(f"Model converted to TPA in {convert_time:.2f} seconds")
        
        # Save TPA model
        output_path = Path(args.output_dir) / f"gemma_1b_tpa_q{args.q_rank}_k{args.k_rank}_v{args.v_rank}.pt"
        print(f"Saving TPA model to {output_path}")
        torch.save({
            'model_state_dict': tpa_model.state_dict(),
            'config': model_config
        }, output_path)
        
        # Clear standard model from memory
        del standard_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Move TPA model to device for inference
        tpa_model = tpa_model.to(device).eval()
        
        # Run inference with TPA model
        print(f"Running inference with prompt: '{args.test_prompt}'")
        
        # Prepare prompt
        prompt = [(args.test_prompt,)]
        
        # Generate text
        generate_start = time.time()
        
        outputs = tpa_model.generate(
            prompts=prompt,
            device=device,
            output_len=args.max_tokens,
            temperature=0.7,
            top_p=0.95,
            top_k=50
        )
        
        generate_time = time.time() - generate_start
        
        # Display results
        print("\n" + "="*50)
        print(f"PROMPT: {args.test_prompt}")
        print(f"RESULT: {outputs[0]}")
        print("="*50)
        
        # Print performance metrics
        tokens_generated = len(outputs[0].split())
        print(f"\nPerformance metrics:")
        print(f"Total generation time: {generate_time:.2f} seconds")
        print(f"Tokens generated: {tokens_generated}")
        print(f"Tokens per second: {tokens_generated / generate_time:.2f}")
        
        # Print memory usage if on CUDA
        if device.type == "cuda" and torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
            memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
            print(f"Memory allocated: {memory_allocated:.2f} GB")
            print(f"Memory reserved:  {memory_reserved:.2f} GB")
            
            # Calculate memory savings
            batch_size = 1
            seq_len = model_config.max_position_embeddings
            num_heads = model_config.num_attention_heads
            head_dim = model_config.head_dim
            bytes_per_element = 2  # 2 bytes for bfloat16/float16
            
            standard_kv_size = 2 * batch_size * seq_len * num_heads * head_dim * bytes_per_element
            tpa_kv_size = (args.k_rank + args.v_rank) * (num_heads + head_dim) * batch_size * seq_len * bytes_per_element
            
            standard_kv_gb = standard_kv_size / (1024 ** 3)
            tpa_kv_gb = tpa_kv_size / (1024 ** 3)
            reduction_ratio = standard_kv_size / tpa_kv_size
            
            print(f"\nMemory efficiency:")
            print(f"Standard KV cache size: {standard_kv_gb:.2f} GB")
            print(f"TPA KV cache size: {tpa_kv_gb:.2f} GB")
            print(f"Reduction ratio: {reduction_ratio:.2f}x")
        
        print("\nTPA test completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()