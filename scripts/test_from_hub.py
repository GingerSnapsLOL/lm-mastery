#!/usr/bin/env python3
"""
Test model from Hugging Face Hub
"""

import argparse
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_from_hub(name: str, prompt: str, max_new_tokens: int = 200, temperature: float = 0.9):
    """Test a model from Hugging Face Hub"""
    
    print(f"=== TESTING FROM HUB ===")
    print(f"Model: {name}")
    print(f"Prompt: {prompt}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Temperature: {temperature}")
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(name, use_fast=False, trust_remote_code=True)
        
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        
        print(f"Tokenizer vocab size: {len(tokenizer)}")
        
        # Load model
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            name, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        if torch.cuda.is_available() and not hasattr(model, 'device_map'):
            model.cuda()
        
        model.config.use_cache = True
        model.eval()
        
        # Generate text
        print("Generating text...")
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode and print
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print("\n--- GENERATED TEXT ---")
        print(generated_text)
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Test model from Hugging Face Hub")
    parser.add_argument("name", help="Model name from Hub (e.g., 'username/model-name')")
    parser.add_argument("prompt", help="Input prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=200, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    
    args = parser.parse_args()
    
    test_from_hub(
        name=args.name,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature
    )

if __name__ == "__main__":
    main()
