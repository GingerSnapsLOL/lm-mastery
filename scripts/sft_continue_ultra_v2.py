# scripts/sft_continue_ultra_v2.py
import os, torch, argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
# from lm_mastery.sft.formatters import *

RESP_TMPL = "Assistant:\n"  # <- this must appear in the prompt

def ultrachat_to_pc(ex):
    msgs = ex.get("messages", [])
    users = [m["content"] for m in msgs if m.get("role")=="user"]
    assists = [m["content"] for m in msgs if m.get("role")=="assistant"]
    return {
        "prompt": f"User: {users[0].strip()}\n{RESP_TMPL}" if users else f"User:\n{RESP_TMPL}",
        "completion": "\n" + (assists[0].strip() if assists else "")
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", required=True, help="Path to your previous SFT ckpt (run_llama_sft_ultra)")
    ap.add_argument("--out_dir", required=True, help="New output dir, e.g. .../run_llama_sft_ultra_v2")
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--ga", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--resume", type=str, default="", help="(optional) path to a specific checkpoint-XXXX to resume")
    args = ap.parse_args()

    # data (UltraChat test/train_sft -> use train_sft for continued SFT)
    raw = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds = raw.map(ultrachat_to_pc, remove_columns=raw.column_names)
    # optional: filter bad/empty completions
    ds = ds.filter(lambda r: len(r["completion"].strip()) > 0)

    # tokenizer & model
    tok = AutoTokenizer.from_pretrained(args.base_ckpt, use_fast=False)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": tok.eos_token or "[PAD]"})
    model = AutoModelForCausalLM.from_pretrained(
        args.base_ckpt,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else None),
    )
    model.resize_token_embeddings(len(tok))
    model.config.use_cache = False

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = SFTConfig(
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
        # SFT-specific
        # DO NOT set dataset_text_field here (not a single "text" column)
        max_length=args.max_len,
        packing=False,
        completion_only_loss=True,   # <— only the completion part contributes to loss
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
    )

    resume_path = args.resume if args.resume else None
    trainer.train(resume_from_checkpoint=resume_path)
    trainer.save_model()
    tok.save_pretrained(args.out_dir)

if __name__ == "__main__":
    main()
