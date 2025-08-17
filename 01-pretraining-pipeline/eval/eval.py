import argparse, math, os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer



parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", required=True)
parser.add_argument("--tokenizer_dir", required=True)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(device)
tokenizer = LlamaTokenizer(vocab_file=os.path.join(args.tokenizer_dir, "spm.model"))

ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
texts = [t for t in ds["text"] if t.strip()]
texts = texts[:512]

losses = []
for t in texts:
    enc = tokenizer(t, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc, labels=enc["input_ids"])
    losses.append(out.loss.item())

ppl = math.exp(sum(losses)/len(losses))
print(f"Validation perplexity (subset): {ppl:.2f}")
