"""
Text generation sampling utilities
"""

import torch
from typing import Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

def simple_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    min_new_tokens: int = 60,
    temperature: float = 0.9,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    no_repeat_ngram_size: int = 3
) -> str:
    """
    Simple text generation function
    
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum new tokens to generate
        min_new_tokens: Minimum new tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Repetition penalty
        no_repeat_ngram_size: N-gram repetition prevention
        
    Returns:
        Generated text
    """
    
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
