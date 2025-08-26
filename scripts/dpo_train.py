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
    ap.add_argument("--lr", type=float, default=3e-7, 
                   help="Learning rate")
    ap.add_argument("--beta", type=float, default=0.05, 
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
    tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
    tok.padding_side = "right"        # <— use RIGHT padding for DPOTrainer
    tok.truncation_side = "left" 

    model = AutoModelForCausalLM.from_pretrained(
        args.base_ckpt,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )
    model.resize_token_embeddings(len(tok))
    model.config.use_cache = False
    from copy import deepcopy
# Option A: reload a fresh copy from disk
    ref_model = AutoModelForCausalLM.from_pretrained(args.base_ckpt, torch_dtype=torch.bfloat16, device_map="auto")
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    # Load preference dataset
    print(f"Loading {args.dataset} preference dataset...")
    
    if args.dataset == "ultrafeedback":
        # UltraFeedback binarized dataset
        raw_ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
        
# --- DPO data formatting for UltraFeedback (matches your SFT template) ---
# Your SFT template: TEMPLATE = "User: {q}\nAssistant:\n"
# This DPO script uses: "User: {user_text}\nAssistant:\n" + response

        RESP_TMPL = "Assistant:\n"   # must match your SFT template format

        def _pick_user_and_assistant(block):
            """block is either a string or a list of {'role','content'} dicts."""
            if isinstance(block, str):
                return "", block.strip()
            if isinstance(block, list):
                user = next((m.get("content","") for m in block if m.get("role") in ("user","prompter")), "")
                asst = next((m.get("content","") for m in block if m.get("role") == "assistant"), "")
                return user.strip(), asst.strip()
            return "", ""

        def format_ultrafeedback_for_dpo(ex, max_user_len=None):
            u_c, a_c = _pick_user_and_assistant(ex.get("chosen", []))
            u_r, a_r = _pick_user_and_assistant(ex.get("rejected", []))

            user_text = (u_c or u_r or ex.get("prompt","")).strip()
            if max_user_len is not None and len(user_text) > max_user_len:
                user_text = user_text[:max_user_len].rstrip()

            prompt = f"User: {user_text}\n{RESP_TMPL}"

            # Keep leading newline to match SFT convention
            if a_c and not a_c.startswith("\n"): a_c = "\n" + a_c.lstrip()
            if a_r and not a_r.startswith("\n"): a_r = "\n" + a_r.lstrip()

            return {"prompt": prompt, "chosen": a_c, "rejected": a_r}
                
        raw_ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
        ds = raw_ds.map(lambda ex: format_ultrafeedback_for_dpo(ex, max_user_len=None),
                        remove_columns=raw_ds.column_names)

        def _ok_row(r):
            if not r["prompt"]: return False
            ch = r["chosen"].strip()
            rj = r["rejected"].strip()
            if len(ch) <= 8 or len(rj) <= 8: return False
            if ch == rj: return False
            return True

        ds = ds.filter(_ok_row)

        
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
        # core schedule
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.ga,

        # stability
        weight_decay=0.01,
        max_grad_norm=0.5,

        # dtypes / ckpt
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # logging / saving
        logging_steps=20,
        logging_first_step=True,
        save_strategy="epoch",
        save_total_limit=2,
        report_to=[],  # no wandb unless you enable it
        # evaluation_strategy="no",  # set to "steps" only if you add eval_dataset

        # TRL / HF Trainer quirks
        remove_unused_columns=False,
        dataloader_num_workers=0,

        # DPO-specific
        beta=args.beta,
        loss_type="sigmoid",
        label_smoothing=0.0,
        precompute_ref_log_probs=True,  # optional speedup
        max_prompt_length=args.max_prompt_len,
        max_length=args.max_len,
    )

    print("Creating DPO trainer...")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,              # TRL will clone & freeze current model as reference
        args=dpo_config,
        processing_class=tok,        # TRL 0.21.0 uses processing_class parameter
        train_dataset=ds,
        # eval_dataset=val_ds,       # add when you wire validation
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
