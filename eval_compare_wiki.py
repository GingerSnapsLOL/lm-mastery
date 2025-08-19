# eval_compare_wiki.py
import os, math, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

import alko  # <-- registers AlkoConfig/AlkoForCausalLM with Auto*

# add your local checkpoint folder here
LOCAL_ALKO = r"01-pretraining-pipeline/results/checkpoints/run_alko_big"
LOCAL_LLAMA = r"01-pretraining-pipeline/results/checkpoints/run_llama_baseline_109M"
MODEL_IDS = [
    "distilgpt2",
    "gpt2",
    "gpt2-medium",
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    LOCAL_ALKO,
    LOCAL_LLAMA,
]

SEQ_LEN = 1024

def build_wikitext(tokenizer):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation").filter(
        lambda ex: len(ex["text"].strip()) > 0
    )
    def tok(batch): return tokenizer(batch["text"], add_special_tokens=False)
    tokd = ds.map(tok, batched=True, remove_columns=["text"])
    def group(batch):
        ids = sum(batch["input_ids"], [])
        n = (len(ids) // SEQ_LEN) * SEQ_LEN
        ids = ids[:n]
        return {"input_ids": [ids[i:i+SEQ_LEN] for i in range(0, n, SEQ_LEN)]}
    return tokd.map(group, batched=True, remove_columns=tokd.column_names)

class FixedLenCollator:
    def __call__(self, feats):
        x = torch.tensor([f["input_ids"] for f in feats], dtype=torch.long)
        return {"input_ids": x, "labels": x.clone()}

def eval_model(mid):
    is_local = os.path.isdir(mid)
    # tokenizer
    tok = AutoTokenizer.from_pretrained(mid) if not is_local else \
          AutoTokenizer.from_pretrained(mid, use_fast=False)  # will pick up your SPM
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else tok.unk_token
    ds = build_wikitext(tok)

    # model
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=dtype)
    if torch.cuda.is_available(): model.cuda()
    model.config.use_cache = False

    args = args = TrainingArguments(
            output_dir=f"tmp_eval_{mid.replace('/','_').replace(':','_')}",
            per_device_eval_batch_size=1,
            dataloader_num_workers=0,   # Windows-safe
            report_to=[],
        )

    trainer = Trainer(model=model, args=args, data_collator=FixedLenCollator())
    m = trainer.evaluate(eval_dataset=ds)
    loss = float(m["eval_loss"]); ppl = math.exp(loss)
    return {"model": mid, "loss": loss, "ppl": ppl, "samples": len(ds)}

def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    rows = [eval_model(mid) for mid in MODEL_IDS]
    rows.sort(key=lambda r: r["ppl"])
    w = max(len(r["model"]) for r in rows)
    print(f"{'Model'.ljust(w)}  Eval loss   PPL     (#seqs)")
    for r in rows:
        print(f"{r['model'].ljust(w)}  {r['loss']:.4f}   {r['ppl']:.2f}   ({r['samples']})")

if __name__ == "__main__":
    main()