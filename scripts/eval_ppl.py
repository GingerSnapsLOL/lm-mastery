import math, torch, argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.ckpt, use_fast=False)
if tok.pad_token is None: tok.add_special_tokens({"pad_token":"[PAD]"})
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation").filter(lambda ex: ex["text"].strip())

SEQ_LEN = 1024
def tok_map(b): return tok(b["text"], add_special_tokens=False)
td = ds.map(tok_map, batched=True, remove_columns=["text"])
def group(b):
    ids = sum(b["input_ids"], [])
    n = (len(ids)//SEQ_LEN)*SEQ_LEN
    return {"input_ids":[ids[i:i+SEQ_LEN] for i in range(0,n,SEQ_LEN)]}
g = td.map(group, batched=True, remove_columns=td.column_names)

class Coll: 
    def __call__(self, feats):
        import torch
        x = torch.tensor([f["input_ids"] for f in feats], dtype=torch.long)
        return {"input_ids": x, "labels": x.clone()}

model = AutoModelForCausalLM.from_pretrained(args.ckpt, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None)
if torch.cuda.is_available(): model.cuda()
model.config.use_cache = False
args_t = TrainingArguments("tmp_eval_sft", per_device_eval_batch_size=1, dataloader_num_workers=0, report_to=[])
m = Trainer(model=model, args=args_t, data_collator=Coll()).evaluate(eval_dataset=g)
print("Eval loss:", m["eval_loss"], "PPL:", math.exp(m["eval_loss"]))
