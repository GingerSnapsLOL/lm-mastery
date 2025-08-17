#!/usr/bin/env python3
"""
Model diagnostic script for debugging training issues
"""

import argparse
import sys
import os
import torch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lm_mastery.data.loaders import load_val_dataset
from lm_mastery.data.collators import FixedLenCollator
from lm_mastery.utils.io import get_checkpoint_path, get_tokenizer_path
from lm_mastery.utils.device import setup_device, get_dtype
from transformers import AutoModelForCausalLM, LlamaTokenizer

def run_diagnostics(model, tokenizer, dataset, device, max_length=1024):
    """Run comprehensive model diagnostics"""
    
    print("=== FORWARD DIAGNOSTICS (1 batch) ===")
    
    # Get one batch
    collator = FixedLenCollator(max_length=max_length, pad_token_id=tokenizer.pad_token_id)
    batch = collator([dataset[0]])
    
    # Move to device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(**batch)
        logits = outputs.logits
        loss = outputs.loss
    
    # Analyze logits
    logits_mean = logits.mean().item()
    logits_std = logits.std().item()
    logits_max_abs = logits.abs().max().item()
    
    has_nan_in_logits = torch.isnan(logits).any().item()
    has_inf_in_logits = torch.isinf(logits).any().item()
    
    print(f"logits mean/std/max_abs: {logits_mean:.6f} / {logits_std:.6f} / {logits_max_abs:.6f}")
    print(f"has_nan_in_logits: {has_nan_in_logits}")
    print(f"has_inf_in_logits: {has_inf_in_logits}")
    
    # Analyze loss
    if loss is not None:
        loss_value = loss.item()
        has_nan_loss = torch.isnan(loss).item()
        has_inf_loss = torch.isinf(loss).item()
        
        print(f"one-batch CE loss (nats): {loss_value:.6f}")
        print(f"has_nan_loss: {has_nan_loss}")
        print(f"has_inf_loss: {has_inf_loss}")
        
        if not has_nan_loss and not has_inf_loss:
            perplexity = torch.exp(loss).item()
            print(f"one-batch perplexity: {perplexity:.2f}")
    
    # Check embeddings
    embeddings = model.get_input_embeddings()
    if embeddings is not None:
        emb_mean = embeddings.weight.mean().item()
        emb_std = embeddings.weight.std().item()
        emb_max_abs = embeddings.weight.abs().max().item()
        
        print(f"embedding mean/std/max_abs: {emb_mean:.6f} / {emb_std:.6f} / {emb_max_abs:.6f}")
    
    # Check gradients (if any)
    if model.training:
        total_norm = 0.0
        param_count = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
            print(f"gradient norm: {total_norm:.6f}")
            print(f"parameters with gradients: {param_count}")

def main():
    parser = argparse.ArgumentParser(description="Run model diagnostics")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--val-dataset", default="val_big", help="Validation dataset name")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Setup device
    device = setup_device()
    dtype = get_dtype(device)
    
    # Load validation dataset
    print(f"Loading validation dataset: {args.val_dataset}")
    val_dataset = load_val_dataset(args.val_dataset)
    print(f"Loaded {len(val_dataset)} validation sequences")
    
    # Load tokenizer
    print("Loading tokenizer...")
    try:
        tokenizer = LlamaTokenizer.from_pretrained(args.checkpoint)
        print("Loaded tokenizer from checkpoint")
    except Exception:
        tokenizer_path = get_tokenizer_path("spm")
        tokenizer = LlamaTokenizer(vocab_file=tokenizer_path)
        print("Loaded tokenizer from SPM")
    
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "</s>", "bos_token": "<s>"})
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=dtype)
    if torch.cuda.is_available():
        model.cuda()
    
    model.config.use_cache = False
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Run diagnostics
    run_diagnostics(model, tokenizer, val_dataset, device, args.max_length)

if __name__ == "__main__":
    main()
