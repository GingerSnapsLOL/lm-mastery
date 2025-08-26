# ppl_eval.py
import math, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

CKPT = r"01-pretraining-pipeline\results\checkpoints\run_llama_sft_mix_v4"
DATA = ("wikitext", "wikitext-2-raw-v1", "test")  # or your held-out corpus

def main():
    tok = AutoTokenizer.from_pretrained(CKPT, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.bfloat16, device_map="auto").eval()

    ds = load_dataset(*DATA[0:2], split=DATA[2])
    texts = [t for t in ds["text"] if len(t.strip()) > 0][:1000]

    max_len = 1024
    nll, tok_count = 0.0, 0
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
        with torch.no_grad():
            out = model(**enc, labels=enc.input_ids)
            nll += out.loss.item() * enc.input_ids.numel()
            tok_count += enc.input_ids.numel()

    ppl = math.exp(nll / tok_count)
    print(f"PPL (approx): {ppl:.2f}")

if __name__ == "__main__":
    main()
