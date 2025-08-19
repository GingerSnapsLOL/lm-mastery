#!/usr/bin/env python3
"""
SFT training script for specific datasets (Dolly, UltraChat)
"""

import os
import sys
import torch
import argparse

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from lm_mastery.sft.formatters import alpaca_to_text, ultrachat_to_text

def main():
    ap = argparse.ArgumentParser(description="Train model with SFT on specific datasets")
    ap.add_argument("--base_ckpt", type=str, required=True, help="Base model checkpoint path")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory")
    ap.add_argument("--dataset", type=str, choices=["dolly", "ultrachat"], required=True, help="Dataset to use")
    ap.add_argument("--max_len", type=int, default=1024, help="Maximum sequence length (note: SFTTrainer handles this automatically)")
    ap.add_argument("--bsz", type=int, default=1, help="Batch size")
    ap.add_argument("--ga", type=int, default=16, help="Gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    ap.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    
    args = ap.parse_args()

    print(f"=== SFT TRAINING ===")
    print(f"Base checkpoint: {args.base_ckpt}")
    print(f"Output directory: {args.out_dir}")
    print(f"Dataset: {args.dataset}")
    print(f"Max length: {args.max_len}")
    print(f"Batch size: {args.bsz}")
    print(f"Grad accum: {args.ga}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")

    # tokenizer
    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.base_ckpt, use_fast=False)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})

    # data
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == "dolly":
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
        # Don't format here - SFTTrainer will handle it
    else:  # ultrachat
        # UltraChat has splits: ['train_sft', 'test_sft', 'train_gen', 'test_gen']
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")  # Fixed: use correct split name
        # Don't format here - SFTTrainer will handle it
    
    print(f"Loaded {len(ds)} examples")

    # model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_ckpt,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )
    model.resize_token_embeddings(len(tok))
    model.config.use_cache = False

    # training
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    targs = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.ga,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        logging_steps=20,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        report_to=[],
    )

    # Create SFT trainer
    print("Creating SFT trainer...")
    
    # Define formatting function for the dataset
    def format_dataset(examples):
        """Format examples to text format for SFT training"""
        if args.dataset == "dolly":
            return alpaca_to_text(examples)
        else:  # ultrachat
            return ultrachat_to_text(examples)
    
    trainer = SFTTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        processing_class=tok,  # Use processing_class instead of tokenizer in trl 0.21.0
        #formatting_func=format_dataset,  # Use formatting_func to format the dataset
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model()  # Use trainer's save method
    tok.save_pretrained(args.out_dir)
    
    print(f"Training completed! Model saved to: {args.out_dir}")

if __name__ == "__main__":
    main()
