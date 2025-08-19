import math, torch
from datasets import load_from_disk
from transformers import LlamaTokenizer
from transformers import Trainer, TrainingArguments
from alko.configuration_alko import AlkoConfig
from alko.modeling_alko import AlkoForCausalLM

CKPT = r"01-pretraining-pipeline/results/checkpoints/run_alko_wt103"
VAL = r"01-pretraining-pipeline/data/processed/val.arrow"
SPM = r"01-pretraining-pipeline/results/tokenizer/spm.model"

# tokenizer
try:
    tok = LlamaTokenizer.from_pretrained(CKPT)
except:
    tok = LlamaTokenizer(vocab_file=SPM)
if tok.eos_token is None: tok.add_special_tokens({"eos_token":"</s>","bos_token":"<s>"})
if tok.pad_token is None: tok.pad_token = tok.eos_token

# model
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = AlkoForCausalLM.from_pretrained(CKPT, torch_dtype=dtype)
if torch.cuda.is_available(): model.to("cuda")
model.config.use_cache = False

# data
ds = load_from_disk(VAL)

# fixed-length collator
class FixedLenCollator:
    def __call__(self, feats):
        x = torch.tensor([f["input_ids"] for f in feats], dtype=torch.long)
        return {"input_ids": x, "labels": x.clone()}

collator = FixedLenCollator()

args = TrainingArguments(output_dir="tmp_eval", per_device_eval_batch_size=1, report_to=[])
trainer = Trainer(model=model, args=args, data_collator=collator)
metrics = trainer.evaluate(eval_dataset=ds)
ppl = math.exp(metrics["eval_loss"])
print("Eval loss:", metrics["eval_loss"])
print("Perplexity:", ppl)
