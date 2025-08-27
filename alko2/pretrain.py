\
import os, math, json, yaml, random, argparse
from typing import List
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments, default_data_collator

def set_torch():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

def load_tokenizer(tokenizer_path: str):
    tok = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok

def temperature_weights(weights: List[float], T: float) -> List[float]:
    if T is None or T <= 0:
        s = sum(weights); return [w/s for w in weights]
    wT = [w ** (1.0 / T) for w in weights]; s = sum(wT); return [w/s for w in wT]

def build_streaming_dataset(mix_yaml_path: str, tok):
    with open(mix_yaml_path, "r") as f:
        mix = yaml.safe_load(f)
    temp = float(mix.get("temperature", 1.0))
    sources = mix["sources"]
    datasets_list, weights = [], []
    for src in sources:
        t = src.get("type", "text"); w = float(src.get("weight", 1.0)); weights.append(w)
        if t == "text":
            pattern = src["data_files"]
            ds = load_dataset("text", data_files=pattern, split="train", streaming=True)
            datasets_list.append(ds)
        elif t == "hf":
            dataset_name = src["dataset"]; config = src.get("config", None); split = src.get("split", "train")
            ds = load_dataset(dataset_name, config, split=split, streaming=True)
            col = src.get("text_column", "text")
            if col != "text":
                ds = ds.rename_column(col, "text")
            datasets_list.append(ds)
        else:
            raise ValueError(f"Unknown source type: {t}")
    probs = temperature_weights(weights, temp)
    mixed = interleave_datasets(datasets_list, probabilities=probs, stopping_strategy="all_exhausted")
    def tok_map(ex):
        ids = tok(ex["text"]).input_ids + [tok.eos_token_id]
        return {"ids": ids}
    tokenized = mixed.map(tok_map, remove_columns=[c for c in mixed.features.keys()])
    return tokenized

class PackedDataset(IterableDataset):
    def __init__(self, token_iterable, seq_len: int, shuffle_buffer: int = 128):
        super().__init__(); self.src = token_iterable; self.seq_len = seq_len; self.shuffle_buffer = shuffle_buffer
    def __iter__(self):
        buf, token_buffer = [], []
        for ex in self.src:
            ids = ex["ids"]; buf.append(ids)
            if len(buf) >= self.shuffle_buffer:
                random.shuffle(buf)
                for sample in buf:
                    token_buffer.extend(sample)
                    while len(token_buffer) >= self.seq_len:
                        chunk = token_buffer[:self.seq_len]; token_buffer = token_buffer[self.seq_len:]
                        yield {"input_ids": torch.tensor(chunk), "labels": torch.tensor(chunk), "attention_mask": torch.ones(self.seq_len, dtype=torch.long)}
                buf.clear()
        for sample in buf:
            token_buffer.extend(sample)
            while len(token_buffer) >= self.seq_len:
                chunk = token_buffer[:self.seq_len]; token_buffer = token_buffer[self.seq_len:]
                yield {"input_ids": torch.tensor(chunk), "labels": torch.tensor(chunk), "attention_mask": torch.ones(self.seq_len, dtype=torch.long)}

def main():
    set_torch()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_config", required=True)
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--mix_yaml", required=True)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--micro_bsz", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.02)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_steps", type=int, default=50000)
    ap.add_argument("--log_steps", type=int, default=20)
    ap.add_argument("--save_steps", type=int, default=2000)
    ap.add_argument("--deepspeed", default=None)
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer_path)
    with open(args.model_config, "r") as f:
        cfg_json = json.load(f)
    cfg = LlamaConfig(**cfg_json)
    model = LlamaForCausalLM(cfg)
    model.gradient_checkpointing_enable(use_reentrant=False)

    token_stream = build_streaming_dataset(args.mix_yaml, tok)
    train_ds = PackedDataset(token_stream, seq_len=args.seq_len, shuffle_buffer=128)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.micro_bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=1,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available(),
        fp16=False,
        logging_steps=args.log_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        evaluation_strategy="no",
        dataloader_pin_memory=True,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        report_to=[],
        deepspeed=args.deepspeed,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, tokenizer=tok, data_collator=default_data_collator)
    trainer.train()
    trainer.save_state(); trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
