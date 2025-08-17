import torch, argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--prompt", default="You are a helpful assistant. Explain transformers in simple words.")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.ckpt, use_fast=False)
if tok.pad_token is None:
    tok.add_special_tokens({"pad_token":"[PAD]"})
model = AutoModelForCausalLM.from_pretrained(args.ckpt, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None)
model.config.use_cache = False
if torch.cuda.is_available(): model.cuda()

ids = tok(args.prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
out = model.generate(**ids, max_new_tokens=256, min_new_tokens=64, do_sample=True,
                     temperature=0.8, top_p=0.95, top_k=50, no_repeat_ngram_size=3,
                     repetition_penalty=1.1, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
print(tok.decode(out[0], skip_special_tokens=True))