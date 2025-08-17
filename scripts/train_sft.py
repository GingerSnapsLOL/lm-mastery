#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) script for instruction-following models
"""

import argparse
import sys
import os
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lm_mastery.data.collators import FixedLenCollator
from lm_mastery.utils.io import get_output_dir, get_checkpoint_path
from lm_mastery.utils.device import setup_device, get_dtype
from lm_mastery.utils.seed import set_seed
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
)
from trl import SFTTrainer

def format_alpaca_example(example):
    """Format example in Alpaca instruction format"""
    instruction = example.get("instruction", "").strip()
    input_text = example.get("input", "").strip()
    output = example.get("output", "").strip()
    
    prompt = f"### Instruction:\n{instruction}\n"
    if input_text:
        prompt += f"\n### Input:\n{input_text}\n"
    prompt += "\n### Response:\n"
    
    return {"text": prompt + output}

def format_chat_example(example):
    """Format example in chat format"""
    messages = example.get("messages", [])
    if not messages:
        return {"text": ""}
    
    # Simple chat format: <s>user: ...</s>assistant: ...</s>
    formatted = ""
    for i, msg in enumerate(messages):
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        
        if role == "user":
            formatted += f"<s>user: {content}</s>"
        elif role == "assistant":
            formatted += f"assistant: {content}</s>"
        else:
            formatted += f"{role}: {content}"
    
    return {"text": formatted}

def load_instruction_dataset(data_path: str, format_type: str = "alpaca"):
    """Load and format instruction dataset"""
    print(f"Loading instruction dataset from: {data_path}")
    
    # Load dataset
    if data_path.endswith('.jsonl'):
        # JSONL format
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        print(f"Loaded {len(data)} examples from JSONL")
    else:
        # Assume it's a directory with dataset files
        from datasets import load_dataset
        dataset = load_dataset("json", data_files=data_path, split="train")
        data = [example for example in dataset]
        print(f"Loaded {len(data)} examples from dataset")
    
    # Format examples
    if format_type == "alpaca":
        formatted_data = [format_alpaca_example(ex) for ex in data]
    elif format_type == "chat":
        formatted_data = [format_chat_example(ex) for ex in data]
    else:
        raise ValueError(f"Unknown format type: {format_type}")
    
    # Filter out empty examples
    formatted_data = [ex for ex in formatted_data if ex["text"].strip()]
    print(f"Formatted {len(formatted_data)} examples")
    
    return formatted_data

def main():
    parser = argparse.ArgumentParser(description="Train model with supervised fine-tuning")
    parser.add_argument("--checkpoint", required=True, help="Base model checkpoint path")
    parser.add_argument("--data", required=True, help="Instruction dataset path or JSONL file")
    parser.add_argument("--output-dir", help="Output directory (overrides default)")
    parser.add_argument("--model-name", default="sft_model", help="Model name for output")
    parser.add_argument("--format", choices=["alpaca", "chat"], default="alpaca", 
                       help="Dataset format type")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = setup_device()
    dtype = get_dtype(device)
    
    print(f"=== SFT TRAINING CONFIG ===")
    print(f"Base checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.data}")
    print(f"Format: {args.format}")
    print(f"Max length: {args.max_length}")
    print(f"Batch size: {args.batch_size}")
    print(f"Grad accum: {args.grad_accum}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Warmup ratio: {args.warmup_ratio}")
    
    # Load dataset
    train_data = load_instruction_dataset(args.data, args.format)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=False)
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, 
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    if torch.cuda.is_available() and not hasattr(model, 'device_map'):
        model.cuda()
    
    model.config.use_cache = False
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Create training arguments
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = get_output_dir(f"run_{args.model_name}")
    
    print(f"Output directory: {output_dir}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,
        logging_steps=20,
        save_strategy="epoch",
        dataloader_num_workers=0,
        report_to=[],
        remove_unused_columns=False,  # Important for SFT
    )
    
    # Create SFT trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        dataset_text_field="text",
        max_seq_length=args.max_length,
        args=training_args,
    )
    
    # Train
    print("\nStarting SFT training...")
    trainer.train()
    
    # Save model
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"SFT training completed! Model saved to: {output_dir}")

if __name__ == "__main__":
    main()
