#!/usr/bin/env python3
"""
Simple test script for TPA with Gemma 1B model.
"""

import argparse
import os
import time
import torch
import traceback
from pathlib import Path

from gemma import config, tokenizer
from gemma.model import GemmaForCausalLM
from gemma.tpa.gemma3_tpa_model import Gemma3ForMultimodalLMwithTPA

def parse_args():
    parser = argparse.ArgumentParser(description="Test TPA with Gemma 1B model")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to standard Gemma 1B checkpoint")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/tokenizer.model", help="Path to tokenizer")
    parser.add_argument("--output", type=str, default="tpa_model.pt", help="Path to save TPA model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--prompt", type=str, default="What are large language models?", help="Test prompt")
    parser.add_argument("--q-rank", type=int, default=6, help="Rank for Q factorization")
    parser.add_argument("--k-rank", type=int, default=2, help="Rank for K factorization")
    parser.add_argument("--v-rank", type=int, default=2, help="Rank for V factorization")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}")
    gemma_tok = tokenizer.Tokenizer(args.tokenizer)
    
    # Create config
    print("Creating Gemma 1B config")
    model_config = config.get_config_for_1b(dtype="float32" if device.type == "cpu" else "bfloat16")
    model_config.vision_config = None  # Explicitly set to None for text-only model
    model_config.tokenizer = args.tokenizer
    
    # Set TPA parameters
    model_config.q_rank = args.q_rank
    model_config.k_rank = args.k_rank
    model_config.v_rank = args.v_rank
    model_config.quant = False  # Disable quantization
    
    # Set sliding window size
    model_config.sliding_window_size = 4096
    print(f"Set sliding_window_size to {model_config.sliding_window_size}")
    
    try:
        # Load standard model
        print(f"Loading standard model from {args.ckpt}")
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        standard_model = GemmaForCausalLM(model_config)
        standard_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        standard_model.eval()
        print("Standard model loaded successfully")
        
        # Create TPA model
        print(f"Creating TPA model with q_rank={args.q_rank}, k_rank={args.k_rank}, v_rank={args.v_rank}")
        tpa_model = Gemma3ForMultimodalLMwithTPA(model_config)
        tpa_model.tokenizer = gemma_tok
        
        # Convert weights
        print("Converting standard weights to TPA format...")
        start_time = time.time()
        tpa_model.convert_from_standard_weights(standard_model)
        convert_time = time.time() - start_time
        print(f"Conversion completed in {convert_time:.2f} seconds")
        
        # Save TPA model
        print(f"Saving TPA model to {args.output}")
        torch.save({
            'model_state_dict': tpa_model.state_dict(),
            'config': model_config
        }, args.output)
        print("TPA model saved successfully")
        
        # Free memory
        del standard_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Test inference
        print(f"Running inference with prompt: '{args.prompt}'")
        tpa_model = tpa_model.to(device).eval()
        
        with torch.no_grad():
            # Format prompt for Gemma3 model
            prompt = [(args.prompt,)]
            
            # Generate text
            gen_start = time.time()
            outputs = tpa_model.generate(
                prompts=prompt,
                device=device,
                output_len=100,
                temperature=0.7,
                top_p=0.95,
                top_k=40
            )
            gen_time = time.time() - gen_start
            
            # Print results
            print("\n" + "="*50)
            print(f"PROMPT: {args.prompt}")
            print(f"RESULT: {outputs[0]}")
            print("="*50)
            print(f"Generation time: {gen_time:.2f} seconds")
            
            # Calculate memory savings
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
                tpa_kv_size = (args.k_rank + args.v_rank) * (num_heads + head_dim) * seq_len * bytes_per_element
                
                reduction = std_kv_size / tpa_kv_size
                print(f"\nMemory efficiency:")
                print(f"Standard KV cache: {std_kv_size/(1024**3):.2f} GB")
                print(f"TPA KV cache: {tpa_kv_size/(1024**3):.2f} GB")
                print(f"Reduction ratio: {reduction:.2f}x")
        
        print("\nTPA test completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()