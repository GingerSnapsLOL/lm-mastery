#!/usr/bin/env python3
"""
Direct Preference Optimization (DPO) training script

Continues training from SFT models using preference datasets to improve response quality.
"""

import os
import sys
import torch
import argparse

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig

def main():
    ap = argparse.ArgumentParser(description="Train model with DPO on preference datasets")
    ap.add_argument("--base_ckpt", type=str, required=True, 
                   help="Base SFT model checkpoint path (e.g., run_llama_sft_mix_v4)")
    ap.add_argument("--out_dir", type=str, required=True, 
                   help="Output directory for DPO model")
    ap.add_argument("--dataset", type=str, default="ultrafeedback", 
                   choices=["ultrafeedback", "identity", "custom"], 
                   help="Preference dataset to use")
    ap.add_argument("--max_len", type=int, default=1024, 
                   help="Maximum sequence length")
    ap.add_argument("--max_prompt_len", type=int, default=768, 
                   help="Maximum prompt length")
    ap.add_argument("--bsz", type=int, default=1, 
                   help="Batch size per device")
    ap.add_argument("--ga", type=int, default=32, 
                   help="Gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=5e-6, 
                   help="Learning rate")
    ap.add_argument("--beta", type=float, default=0.1, 
                   help="DPO beta parameter (controls preference strength)")
    ap.add_argument("--epochs", type=int, default=1, 
                   help="Number of training epochs")
    ap.add_argument("--resume", type=str, default="", 
                   help="(optional) path to checkpoint-XXXX to resume from")
    ap.add_argument("--max_samples", type=int, default=0, 
                   help="Maximum samples to use (0 = use all)")
    
    args = ap.parse_args()

    print(f"=== DPO TRAINING ===")
    print(f"Base checkpoint: {args.base_ckpt}")
    print(f"Output directory: {args.out_dir}")
    print(f"Dataset: {args.dataset}")
    print(f"Max length: {args.max_len}")
    print(f"Max prompt length: {args.max_prompt_len}")
    print(f"Batch size: {args.bsz}")
    print(f"Grad accum: {args.ga}")
    print(f"Learning rate: {args.lr}")
    print(f"Beta: {args.beta}")
    print(f"Epochs: {args.epochs}")
    if args.max_samples > 0:
        print(f"Max samples: {args.max_samples}")

    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tok = AutoTokenizer.from_pretrained(args.base_ckpt, use_fast=False)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": tok.eos_token or "[PAD]"})

    model = AutoModelForCausalLM.from_pretrained(
        args.base_ckpt,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )
    model.resize_token_embeddings(len(tok))
    model.config.use_cache = False

    # Load preference dataset
    print(f"Loading {args.dataset} preference dataset...")
    
    if args.dataset == "ultrafeedback":
        # UltraFeedback binarized dataset
        raw_ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
        
# --- DPO data formatting for UltraFeedback (keeps your exact template) ---

        RESP_TMPL = "Assistant:\n"   # must appear in the prompt

        def _pick_user_and_assistant(block):
            """block is either a string or a list of {'role','content'} dicts."""
            if isinstance(block, str):
                return "", block.strip()
            if isinstance(block, list):
                user = next((m.get("content","") for m in block if m.get("role") in ("user","prompter")), "")
                asst = next((m.get("content","") for m in block if m.get("role") == "assistant"), "")
                return user.strip(), asst.strip()
            return "", ""

        def format_ultrafeedback_for_dpo(ex):
            u_c, a_c = _pick_user_and_assistant(ex.get("chosen", []))
            u_r, a_r = _pick_user_and_assistant(ex.get("rejected", []))

            # Build one prompt for both completions; fall back to ex['prompt'] if needed
            user_text = (u_c or u_r or ex.get("prompt","")).strip()
            prompt = f"User: {user_text}\n{RESP_TMPL}"

            # Keep the same completion boundary as SFT (you trained with a leading newline)
            if a_c and not a_c.startswith("\n"): a_c = "\n" + a_c
            if a_r and not a_r.startswith("\n"): a_r = "\n" + a_r

            return {"prompt": prompt, "chosen": a_c, "rejected": a_r}

                
        raw_ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
        ds = raw_ds.map(format_ultrafeedback_for_dpo, remove_columns=raw_ds.column_names)
        ds = ds.filter(lambda r: len(r["prompt"])>0 and len(r["chosen"])>8 and len(r["rejected"])>8)

        
    elif args.dataset == "identity":
        # Identity preference dataset for changing model behavior
        raw_ds = load_dataset("mrfakename/identity", split="train")
        
        def format_identity(ex):
            """Format identity examples for DPO"""
            conversations = ex.get("conversations", [])
            # Get the last human message as prompt
            prompt = next((msg["value"] for msg in reversed(conversations) 
                          if msg["from"] == "human"), "")
            
            # For identity training, we'll create synthetic preferences
            # This is a simplified version - you might want to customize this
            chosen = "I am a helpful AI assistant. I aim to be truthful and helpful."
            rejected = "I cannot answer that question."
            
            formatted_prompt = f"User: {prompt}\nAssistant:\n"
            return {
                "prompt": formatted_prompt,
                "chosen": chosen,
                "rejected": rejected
            }
        
        ds = raw_ds.map(format_identity, remove_columns=raw_ds.column_names)
        
    else:  # custom
        # Load custom preference dataset
        raw_ds = load_dataset("banghua/DL-DPO-Dataset", split="train")
        
        def format_custom(ex):
            """Format custom DPO dataset"""
            # Assuming the dataset has chosen/rejected fields
            prompt = ex.get("prompt", "").strip()
            chosen = ex.get("chosen", "").strip()
            rejected = ex.get("rejected", "").strip()
            
            if not prompt:
                prompt = "User: Please respond to this request.\nAssistant:\n"
            
            return {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            }
        
        ds = raw_ds.map(format_custom, remove_columns=raw_ds.column_names)

    # Filter out examples with very short responses
    ds = ds.filter(lambda r: len(r["chosen"].strip()) > 8 and len(r["rejected"].strip()) > 8)
    
    # Limit samples if specified
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    
    print(f"Loaded {len(ds)} preference examples")

    # DPO training configuration
    print("Setting up DPO training...")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dpo_config = DPOConfig(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.ga,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        beta=args.beta,
        max_prompt_length=args.max_prompt_len,
        max_length=args.max_len,
        bf16=torch.cuda.is_available(),
        fp16=False,
        logging_steps=20,
        save_strategy="epoch",
        report_to=[],
        gradient_checkpointing=True,
        dataloader_num_workers=0,
    )

    # Create DPO trainer
    print("Creating DPO trainer...")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Use the same model as reference (self-DPO)
        args=dpo_config,
        processing_class=tok,  # TRL 0.21+ uses processing_class
        train_dataset=ds,
    )

    # Start training
    print("Starting DPO training...")
    resume_path = args.resume if args.resume else None
    dpo_trainer.train(resume_from_checkpoint=resume_path)
    
    # Save the trained model
    print("Saving DPO model...")
    dpo_trainer.save_model()
    tok.save_pretrained(args.out_dir)
    
    print(f"DPO training completed! Model saved to: {args.out_dir}")

if __name__ == "__main__":
    main()
