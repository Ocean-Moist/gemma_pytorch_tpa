"""Demo script for running inference with TPA-based Gemma models."""

import torch
import argparse
from gemma import config as gemma_config
from gemma.gemma3_model import Gemma3ForMultimodalLM
from gemma.tpa.gemma3_tpa_model import Gemma3ForMultimodalLMwithTPA
from PIL import Image
import os

def load_image(image_path):
    """Load image from path."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file {image_path} not found")
    return Image.open(image_path).convert("RGB")

def main():
    parser = argparse.ArgumentParser(description="Run Gemma3 with TPA inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model weights")
    parser.add_argument("--model_variant", type=str, default="4b", choices=["1b", "4b", "12b", "27b"], help="Model variant")
    parser.add_argument("--prompt", type=str, default="Hello, world!", help="Text prompt for model")
    parser.add_argument("--image", type=str, default=None, help="Optional path to image file for multimodal prompting")
    parser.add_argument("--output_len", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.9, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=64, help="Top-k sampling parameter")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run inference on (cuda/cpu)")
    parser.add_argument("--convert_from_standard", action="store_true", 
                        help="Whether to convert from standard Gemma weights to TPA weights")
    parser.add_argument("--save_tpa_model", type=str, default=None,
                        help="Path to save the converted TPA model weights (if convert_from_standard is True)")
    parser.add_argument("--q_rank", type=int, default=6, help="Rank for query factorization in TPA")
    parser.add_argument("--k_rank", type=int, default=2, help="Rank for key factorization in TPA")
    parser.add_argument("--v_rank", type=int, default=2, help="Rank for value factorization in TPA")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    
    # Create TPA-compatible model configuration
    if args.model_variant == "1b":
        config = gemma_config.get_config_for_1b(dtype="float32" if device.type == "cpu" else "bfloat16")
    elif args.model_variant == "4b":
        config = gemma_config.get_config_for_4b(dtype="float32" if device.type == "cpu" else "bfloat16")
    elif args.model_variant == "12b":
        config = gemma_config.get_config_for_12b(dtype="float32" if device.type == "cpu" else "bfloat16")
    elif args.model_variant == "27b":
        config = gemma_config.get_config_for_27b_v3(dtype="float32" if device.type == "cpu" else "bfloat16")
    
    # Add TPA specific configuration parameters
    config.q_rank = args.q_rank
    config.k_rank = args.k_rank
    config.v_rank = args.v_rank
    
    # Create and load model
    if args.convert_from_standard:
        print(f"Loading standard Gemma model from {args.model_path}...")
        standard_model = Gemma3ForMultimodalLM(config)
        standard_model.load_weights(args.model_path)
        standard_model.eval()
        
        print("Converting to TPA model...")
        model = Gemma3ForMultimodalLMwithTPA(config)
        model.convert_from_standard_weights(standard_model)
        
        if args.save_tpa_model:
            print(f"Saving TPA model to {args.save_tpa_model}...")
            os.makedirs(os.path.dirname(args.save_tpa_model), exist_ok=True)
            torch.save({'model_state_dict': model.state_dict()}, args.save_tpa_model)
            
        # Clear standard model from memory
        del standard_model
        torch.cuda.empty_cache() if device.type == "cuda" else None
        
    else:
        print(f"Loading TPA Gemma model from {args.model_path}...")
        model = Gemma3ForMultimodalLMwithTPA(config)
        model.load_weights(args.model_path)
    
    model.to(device)
    model.eval()
    
    # Prepare prompt
    if args.image:
        print(f"Loading image from {args.image}...")
        image = load_image(args.image)
        prompt = [(args.prompt, image)]
    else:
        prompt = [(args.prompt,)]
    
    # Generate response
    print("Generating response...")
    outputs = model.generate(
        prompts=prompt,
        device=device,
        output_len=args.output_len,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k
    )
    
    # Print the generated text
    print("\nGenerated response:")
    print("="*50)
    print(outputs[0])
    print("="*50)
    
    # Print memory usage statistics
    if device.type == "cuda":
        memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # Convert to GB
        memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)  # Convert to GB
        print(f"\nMemory stats:")
        print(f"Allocated: {memory_allocated:.2f} GB")
        print(f"Reserved:  {memory_reserved:.2f} GB")

if __name__ == "__main__":
    main()