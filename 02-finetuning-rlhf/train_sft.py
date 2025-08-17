# train_sft.py
import json, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer

CKPT = r"01-pretraining-pipeline/results/checkpoints/run_llama_baseline_109M"
OUT  = r"01-pretraining-pipeline/results/checkpoints/run_llama_sft"

# Example: Alpaca-style JSONL with fields: instruction, input, output
DATA = r"data/alpaca_like.jsonl"

def format_example(ex):
    instr = ex.get("instruction","").strip()
    inp   = ex.get("input","").strip()
    out   = ex.get("output","").strip()
    prompt = f"### Instruction:\n{instr}\n"
    if inp: prompt += f"\n### Input:\n{inp}\n"
    prompt += "\n### Response:\n"
    return {"text": prompt + out}

def main():
    tok = AutoTokenizer.from_pretrained(CKPT, use_fast=False)
    if tok.pad_token is None: tok.add_special_tokens({"pad_token":"[PAD]"})
    ds = load_dataset("json", data_files=DATA, split="train").map(format_example)

    model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.bfloat16)
    model.resize_token_embeddings(len(tok))
    model.config.use_cache = False

    args = TrainingArguments(
        output_dir=OUT, per_device_train_batch_size=1, gradient_accumulation_steps=16,
        learning_rate=2e-5, num_train_epochs=3, bf16=True, fp16=False,
        lr_scheduler_type="cosine", warmup_ratio=0.03,
        weight_decay=0.0, logging_steps=20, save_strategy="epoch",
        dataloader_num_workers=0, report_to=[]
    )

    trainer = SFTTrainer(
        model=model, tokenizer=tok, train_dataset=ds,
        dataset_text_field="text", max_seq_length=1024, args=args,
    )
    trainer.train()
    model.save_pretrained(OUT, safe_serialization=False)
    tok.save_pretrained(OUT)

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()