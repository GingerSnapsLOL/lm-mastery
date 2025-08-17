# train_with_tiktoken.py

#python 01-pretraining-pipeline/train/train_with_tiktoken.py \
#  --processed_path "01-pretraining-pipeline/data/processed/train_o200k" \
#  --output_dir "01-pretraining-pipeline/results/checkpoints/run_o200k" \
#  --hidden_size 512 --num_layers 8 --num_heads 8 --intermediate_size 1536 \
#  --batch_size 1 --grad_accum 32 --epochs 1.0


import argparse, os, math
import torch
from datasets import load_from_disk
from transformers import (
    LlamaConfig, LlamaForCausalLM,
    Trainer, TrainingArguments, set_seed
)

class SimpleCollator:
    def __call__(self, features):
        # features: list of {"input_ids": [int,int,...]}
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        input_ids = torch.stack(input_ids, dim=0)
        # labels equal to inputs for causal LM (shift inside model)
        return {"input_ids": input_ids, "labels": input_ids.clone()}

def read_meta(ds_dir):
    meta = {"encoding": None, "vocab_size": None, "seq_len": None}
    path = os.path.join(ds_dir, "tiktoken_meta.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                k,v = line.strip().split("=",1)
                meta[k]= v
    if meta["vocab_size"] is not None:
        meta["vocab_size"] = int(meta["vocab_size"])
    if meta["seq_len"] is not None:
        meta["seq_len"] = int(meta["seq_len"])
    return meta

parser = argparse.ArgumentParser()
parser.add_argument("--processed_path", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--seed", type=int, default=42)
# model hyperparams (override defaults if you want)
parser.add_argument("--hidden_size", type=int, default=768)
parser.add_argument("--num_layers", type=int, default=12)
parser.add_argument("--num_heads", type=int, default=12)
parser.add_argument("--num_kv_heads", type=int, default=None)
parser.add_argument("--intermediate_size", type=int, default=2048)
parser.add_argument("--max_position_embeddings", type=int, default=None)  # default: use seq_len from meta
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--grad_accum", type=int, default=32)
parser.add_argument("--epochs", type=float, default=1.0)
args = parser.parse_args()

set_seed(args.seed)

# Load dataset (already tokenized into fixed-length sequences)
ds = load_from_disk(args.processed_path)

# Read tokenizer meta to set vocab/seq_len
meta = read_meta(args.processed_path)
vocab_size = meta["vocab_size"] or 200000
seq_len = args.max_position_embeddings or meta["seq_len"] or 2048

# Build a small LLaMA-style config; vocab_size comes from tiktoken
config = LlamaConfig(
    vocab_size=vocab_size,
    max_position_embeddings=seq_len,
    hidden_size=args.hidden_size,
    num_hidden_layers=args.num_layers,
    num_attention_heads=args.num_heads,
    num_key_value_heads=args.num_kv_heads or args.num_heads,  # set smaller for GQA
    intermediate_size=args.intermediate_size,
    rope_theta=500000,
    rms_norm_eps=1e-5
)

model = LlamaForCausalLM(config)

# No HF tokenizer needed: our collator just batches tensors
collator = SimpleCollator()

# Trainer
train_args = TrainingArguments(
    output_dir=args.output_dir,
    overwrite_output_dir=True,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    learning_rate=args.learning_rate,
    warmup_ratio=0.01,
    weight_decay=0.1,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    evaluation_strategy="no",
    fp16=torch.cuda.is_available(),  # BF16 if you want, on Linux/Ampere+
    report_to=["none"],
)

trainer = Trainer(
    model=model,
    args=train_args,
    data_collator=collator,
    train_dataset=ds,
)

trainer.train()
trainer.save_model(args.output_dir)

print("Done. Saved to:", args.output_dir)
print(f"[meta] vocab_size={vocab_size}, seq_len={seq_len}")