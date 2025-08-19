# eval_alko.py
import math, torch, os
from datasets import load_from_disk
from transformers import TrainingArguments, Trainer, LlamaTokenizer, AutoModelForCausalLM
import alko  # registers your custom model

CKPT = r"01-pretraining-pipeline/results/checkpoints/run_alko_big"
VAL  = r"01-pretraining-pipeline/data/processed/val_big.arrow"
SPM  = r"01-pretraining-pipeline/results/tokenizer/spm.model"

# tokenizer
try:
    tok = LlamaTokenizer.from_pretrained(CKPT)
except Exception:
    tok = LlamaTokenizer(vocab_file=SPM)
if tok.eos_token is None: tok.add_special_tokens({"eos_token":"</s>","bos_token":"<s>"})
if tok.pad_token is None: tok.pad_token = tok.eos_token

# data
ds = load_from_disk(VAL)

# collator (fixed-length)
class FixedLenCollator:
    def __call__(self, feats):
        import torch
        x = torch.tensor([f["input_ids"] for f in feats], dtype=torch.long)
        return {"input_ids": x, "labels": x.clone()}

collator = FixedLenCollator()

# model
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=dtype)
if torch.cuda.is_available(): model.cuda()
model.config.use_cache = False

# eval (perplexity)
args = TrainingArguments(output_dir="tmp_eval", per_device_eval_batch_size=1, report_to=[], dataloader_num_workers=0)
trainer = Trainer(model=model, args=args, data_collator=collator)
metrics = trainer.evaluate(eval_dataset=ds)
print("Eval loss:", metrics["eval_loss"])
print("Perplexity:", math.exp(metrics["eval_loss"]))

# a tiny sample
prompt = "You are AlkoForCausalLM. Explain in simple words what a transformer is."
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.9, top_p=0.95,
                     repetition_penalty=1.1, no_repeat_ngram_size=3,
                     pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
print(tok.decode(out[0], skip_special_tokens=True))
