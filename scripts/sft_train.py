import os, torch, argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from lm_mastery.sft.formatters import alpaca_to_text, ultrachat_to_text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--dataset", type=str, choices=["dolly","ultrachat"], required=True)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--ga", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    # tokenizer
    tok = AutoTokenizer.from_pretrained(args.base_ckpt, use_fast=False)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token":"[PAD]"})

    # data
    if args.dataset == "dolly":
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
        ds = ds.map(alpaca_to_text, remove_columns=ds.column_names)
    else:
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train")
        ds = ds.map(ultrachat_to_text, remove_columns=ds.column_names)

    # model
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

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=args.max_len,
        args=targs,
    )
    trainer.train()
    model.save_pretrained(args.out_dir, safe_serialization=False)
    tok.save_pretrained(args.out_dir)

if __name__ == "__main__":
    main()
