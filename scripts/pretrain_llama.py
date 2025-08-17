#!/usr/bin/env python3
"""
Llama model pretraining script
"""

import argparse
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lm_mastery.data.loaders import load_train_dataset
from lm_mastery.data.collators import FixedLenCollator
from lm_mastery.train.pretrain import TrainingConfig, create_trainer, init_model_weights
from lm_mastery.utils.io import get_output_dir, get_tokenizer_path
from lm_mastery.utils.device import setup_device, print_device_info
from lm_mastery.utils.seed import set_seed
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

def main():
    parser = argparse.ArgumentParser(description="Train Llama model")
    parser.add_argument("--model-name", default="llama_baseline_109M", help="Model name for output")
    parser.add_argument("--train-dataset", default="train_big", help="Training dataset name")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--grad-accum", type=int, default=32, help="Gradient accumulation steps")
    parser.add_argument("--max-steps", type=int, default=9156, help="Maximum training steps")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.15, help="Warmup ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", help="Output directory (overrides default)")
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = setup_device()
    print_device_info(device)
    
    # Load dataset
    print(f"Loading training dataset: {args.train_dataset}")
    train_dataset = load_train_dataset(args.train_dataset)
    print(f"Loaded {len(train_dataset)} training sequences")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer_path = get_tokenizer_path("spm")
    tokenizer = LlamaTokenizer(vocab_file=tokenizer_path)
    
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "</s>", "bos_token": "<s>"})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    
    # Create model
    print("Creating Llama model...")
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=2048,
        rms_norm_eps=1e-5,
        rope_theta=5e5,
        max_position_embeddings=1024,
        tie_word_embeddings=True,
        use_cache=False,
    )
    
    model = LlamaForCausalLM(config)
    model.resize_token_embeddings(len(tokenizer))
    
    # Initialize weights conservatively
    init_model_weights(model, std=0.01)
    
    # Create data collator
    collator = FixedLenCollator(max_length=args.max_length, pad_token_id=tokenizer.pad_token_id)
    
    # Create training config
    training_config = TrainingConfig(
        model_name=args.model_name,
        model_type="llama",
        train_dataset=args.train_dataset,
        max_length=args.max_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
    )
    
    # Get output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = get_output_dir(f"run_{args.model_name}")
    
    print(f"Output directory: {output_dir}")
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        config=training_config,
        output_dir=output_dir,
        data_collator=collator
    )
    
    # Print training config
    print("=== TRAINING CONFIG ===")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.train_dataset} ({len(train_dataset)} sequences)")
    print(f"Max length: {args.max_length}")
    print(f"Batch size: {args.batch_size}")
    print(f"Grad accum: {args.grad_accum}")
    print(f"Max steps: {args.max_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Warmup ratio: {args.warmup_ratio}")
    print(f"Output dir: {output_dir}")
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save model
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"Training completed! Model saved to: {output_dir}")

if __name__ == "__main__":
    main()
