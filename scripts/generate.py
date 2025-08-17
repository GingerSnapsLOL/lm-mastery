#!/usr/bin/env python3
"""
Text generation script for trained models
"""

import argparse
import sys
import os
import torch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lm_mastery.utils.io import get_checkpoint_path
from lm_mastery.utils.device import setup_device, get_dtype
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_text(
    model, 
    tokenizer, 
    prompt: str, 
    max_new_tokens: int = 200,
    min_new_tokens: int = 60,
    temperature: float = 0.9,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    no_repeat_ngram_size: int = 3
):
    """Generate text from a prompt"""
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and return
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Generate text from trained model")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--prompt", default="Hello, how are you? Can you tell me", 
                       help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=200, help="Maximum new tokens")
    parser.add_argument("--min-tokens", type=int, default=60, help="Minimum new tokens")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, 
                       help="Repetition penalty")
    
    args = parser.parse_args()
    
    # Setup device
    device = setup_device()
    dtype = get_dtype(device)
    
    print(f"=== TEXT GENERATION ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Prompt: {args.prompt}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=False)
    
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "</s>", "bos_token": "<s>"})
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, 
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    if torch.cuda.is_available() and not hasattr(model, 'device_map'):
        model.cuda()
    
    model.config.use_cache = True  # Enable cache for generation
    model.eval()
    
    # Generate text
    print("\nGenerating text...")
    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        min_new_tokens=args.min_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )
    
    # Print results
    print("\n--- GENERATED TEXT ---")
    print(generated_text)

if __name__ == "__main__":
    main()