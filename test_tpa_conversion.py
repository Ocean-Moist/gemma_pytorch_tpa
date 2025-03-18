"""Simple script to test TPA conversion with Gemma 1B model."""

import torch
import os
import argparse
import time
import tqdm
import math
from gemma import config, tokenizer
from gemma.model import GemmaForCausalLM
from gemma.tpa.tpa_model import GemmaTPAModel, create_tpa_kv_caches
from gemma.tpa.tpa_attention import GemmaTensorProductAttention

def factorize_qkv_weights(weight, head_dim, out_dim, hidden_size, rank):
    """
    Factorize weight matrix using SVD.
    
    Args:
        weight: Weight matrix to factorize
        head_dim: Dimension of attention head
        out_dim: Output dimension (number of heads)
        hidden_size: Model hidden size
        rank: Rank for factorization
        
    Returns:
        Tuple of (A_weight, B_weight) factorized matrices
    """
    # Reshape for factorization
    weight_reshaped = weight.reshape(out_dim, head_dim, hidden_size)
    
    # Flatten for SVD
    weight_flat = weight_reshaped.reshape(out_dim * head_dim, hidden_size)
    
    print(f"  Performing SVD on tensor of shape {weight_flat.shape}...")
    # Perform SVD
    U, S, Vh = torch.svd(weight_flat)
    
    # Use top-k singular values/vectors
    rank = min(rank, min(U.shape[1], Vh.shape[0]))
    
    # Scale singular values into the factors
    sqrt_S = torch.sqrt(S[:rank])
    U_scaled = U[:, :rank] * sqrt_S
    Vh_scaled = Vh[:rank] * sqrt_S.unsqueeze(1)
    
    # Reshape for A projections (out_dim, rank) and hidden_size
    A_weight = U_scaled.reshape(out_dim, head_dim, rank).permute(0, 2, 1).reshape(out_dim * rank, head_dim)
    B_weight = Vh_scaled.reshape(rank, hidden_size)
    
    return A_weight.transpose(0, 1), B_weight.transpose(0, 1)

def convert_model_to_tpa(standard_model, tpa_model, config):
    """
    Convert standard Gemma model to TPA model.
    
    Args:
        standard_model: Standard GemmaForCausalLM model
        tpa_model: GemmaTPAModel model to populate
        config: Model configuration
        
    Returns:
        Populated TPA model
    """
    # Copy token embeddings and other non-attention weights
    # This part would be model-specific implementation
    
    total_layers = len(standard_model.model.layers)
    
    # Configure progress bar
    progress_bar = tqdm.tqdm(total=total_layers, desc="Converting layers")
    
    # Process each layer
    for i, (std_layer, tpa_layer) in enumerate(zip(standard_model.model.layers, tpa_model.layers)):
        layer_start = time.time()
        
        # Copy non-attention weights (MLP, LayerNorms)
        tpa_layer.mlp.load_state_dict(std_layer.mlp.state_dict())
        tpa_layer.input_layernorm.load_state_dict(std_layer.input_layernorm.state_dict())
        tpa_layer.post_attention_layernorm.load_state_dict(std_layer.post_attention_layernorm.state_dict())
        
        if hasattr(tpa_layer, 'pre_feedforward_layernorm') and tpa_layer.pre_feedforward_layernorm is not None:
            tpa_layer.pre_feedforward_layernorm.load_state_dict(std_layer.pre_feedforward_layernorm.state_dict())
        
        if hasattr(tpa_layer, 'post_feedforward_layernorm') and tpa_layer.post_feedforward_layernorm is not None:
            tpa_layer.post_feedforward_layernorm.load_state_dict(std_layer.post_feedforward_layernorm.state_dict())
        
        # Get standard attention QKV projections
        qkv_weight = std_layer.self_attn.qkv_proj.weight
        q_size = config.num_attention_heads * config.head_dim
        kv_size = config.num_key_value_heads * config.head_dim
        
        # Split into Q, K, V
        q_weight = qkv_weight[:q_size]
        k_weight = qkv_weight[q_size:q_size+kv_size]
        v_weight = qkv_weight[q_size+kv_size:]
        
        # Factorize using SVD for TPA
        print(f"Layer {i+1}/{total_layers}: Factorizing Q matrix...")
        q_A, q_B = factorize_qkv_weights(
            q_weight, 
            config.head_dim, 
            config.num_attention_heads, 
            config.hidden_size, 
            config.q_rank
        )
        
        print(f"Layer {i+1}/{total_layers}: Factorizing K matrix...")
        k_A, k_B = factorize_qkv_weights(
            k_weight, 
            config.head_dim, 
            config.num_key_value_heads, 
            config.hidden_size, 
            config.k_rank
        )
        
        print(f"Layer {i+1}/{total_layers}: Factorizing V matrix...")
        v_A, v_B = factorize_qkv_weights(
            v_weight, 
            config.head_dim, 
            config.num_key_value_heads, 
            config.hidden_size, 
            config.v_rank
        )
        
        # Set factorized weights
        with torch.no_grad():
            tpa_layer.self_attn.W_A_q.weight.copy_(q_A)
            tpa_layer.self_attn.W_B_q.weight.copy_(q_B)
            tpa_layer.self_attn.W_A_k.weight.copy_(k_A)
            tpa_layer.self_attn.W_B_k.weight.copy_(k_B)
            tpa_layer.self_attn.W_A_v.weight.copy_(v_A)
            tpa_layer.self_attn.W_B_v.weight.copy_(v_B)
            
        # Copy output projection
        tpa_layer.self_attn.o_proj.load_state_dict(std_layer.self_attn.o_proj.state_dict())
        
        # Update progress bar
        layer_time = time.time() - layer_start
        remaining_layers = total_layers - (i + 1)
        est_remaining_time = layer_time * remaining_layers
        
        progress_bar.set_postfix({"Layer": f"{i+1}/{total_layers}", 
                                  "Est. time remaining": f"{est_remaining_time:.1f}s"})
        progress_bar.update(1)
    
    progress_bar.close()
    
    # Copy final layer norm
    if hasattr(standard_model.model, 'norm') and hasattr(tpa_model, 'norm'):
        tpa_model.norm.load_state_dict(standard_model.model.norm.state_dict())
    
    return tpa_model

def run_inference(model_path, prompt, max_tokens=100, temperature=0.7, device="cpu"):
    """
    Run inference with a TPA model.
    
    Args:
        model_path: Path to TPA model
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        device: Device to run on (cpu/cuda)
    """
    print(f"Loading TPA model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    if "config" in checkpoint:
        model_config = checkpoint["config"]
    else:
        print("Config not found in checkpoint, using default 1B config...")
        model_config = config.get_config_for_1b(dtype="float32")
    
    # Create tokenizer
    if os.path.exists("gemma_models/tokenizer.model"):
        tokenizer_path = "gemma_models/tokenizer.model"
    else:
        tokenizer_path = "tokenizer/tokenizer.model"
    
    model_tokenizer = tokenizer.Tokenizer(tokenizer_path)
    
    # Create TPA model
    model = GemmaForCausalLM(model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    
    # Tokenize prompt
    input_ids = model_tokenizer.encode(prompt)
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    # Generate
    print(f"Generating with prompt: '{prompt}'")
    generate_start = time.time()
    
    with torch.no_grad():
        # Simple greedy decoding for demo purposes
        for _ in tqdm.tqdm(range(max_tokens), desc="Generating"):
            outputs = model(input_ids_tensor)
            next_token_logits = outputs[:, -1, :]
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            input_ids_tensor = torch.cat([input_ids_tensor, next_token], dim=-1)
            
            # Stop if we generate EOS
            if next_token.item() == model_tokenizer.eos_id:
                break
    
    generate_time = time.time() - generate_start
    generated_text = model_tokenizer.decode(input_ids_tensor[0].tolist())
    
    print("\n" + "="*50)
    print("Generated text:")
    print(generated_text)
    print("="*50)
    print(f"Generation completed in {generate_time:.2f} seconds")

def main():
    parser = argparse.ArgumentParser(description="Test TPA conversion for Gemma 1B model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model weights")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save TPA model")
    parser.add_argument("--q_rank", type=int, default=6, help="Rank for query factorization")
    parser.add_argument("--k_rank", type=int, default=2, help="Rank for key factorization")
    parser.add_argument("--v_rank", type=int, default=2, help="Rank for value factorization")
    parser.add_argument("--run_inference", action="store_true", help="Run inference after conversion")
    parser.add_argument("--prompt", type=str, default="What are large language models?", 
                        help="Prompt for inference")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum tokens to generate")
    
    args = parser.parse_args()
    
    # Create Gemma 1B configuration
    print("Creating model configuration...")
    model_config = config.get_config_for_1b(dtype="float32")
    
    # Add TPA specific configuration parameters
    model_config.q_rank = args.q_rank
    model_config.k_rank = args.k_rank
    model_config.v_rank = args.v_rank
    
    print(f"TPA configuration: q_rank={args.q_rank}, k_rank={args.k_rank}, v_rank={args.v_rank}")
    
    start_time = time.time()
    print(f"Loading model from {args.model_path}...")
    model_data = torch.load(args.model_path, map_location="cpu")
    print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    
    # Create standard Gemma model
    print("Creating standard model instance...")
    standard_model = GemmaForCausalLM(model_config)
    
    # Load weights
    print("Loading weights into model...")
    load_start = time.time()
    standard_model.load_state_dict(model_data["model_state_dict"], strict=False)
    print(f"Weights loaded in {time.time() - load_start:.2f} seconds")
    
    # Create TPA model
    print("Creating TPA model...")
    tpa_model = GemmaTPAModel(model_config)
    
    # Convert standard model to TPA model
    print("Converting model to TPA format...")
    convert_start = time.time()
    convert_model_to_tpa(standard_model, tpa_model, model_config)
    conversion_time = time.time() - convert_start
    print(f"Conversion completed in {conversion_time:.2f} seconds")
    
    # Create wrapper model with token embeddings and output layer
    tpa_wrapper = GemmaForCausalLM(model_config)
    tpa_wrapper.load_state_dict(standard_model.state_dict(), strict=False)
    tpa_wrapper.model = tpa_model
    
    # Save TPA model
    print(f"Saving converted model to {args.save_path}...")
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save({
        'model_state_dict': tpa_wrapper.state_dict(),
        'config': model_config
    }, args.save_path)
    print(f"Model saved successfully!")
    
    # Run inference if requested
    if args.run_inference:
        run_inference(
            model_path=args.save_path,
            prompt=args.prompt,
            max_tokens=args.max_tokens
        )
    else:
        print("\nTo run inference with this model, use:")
        print(f"python test_tpa_conversion.py --model_path {args.save_path} --save_path {args.save_path} --run_inference --prompt \"Your prompt here\"")

if __name__ == "__main__":
    main()