#!/usr/bin/env python3
import argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

TEMPLATE = "User: {q}\nAssistant:\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.ckpt, use_fast=False)
    # avoid resizing embeddings at inference time:
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else None),
        device_map=("auto" if torch.cuda.is_available() else None),
    )
    model.eval(); model.config.use_cache = True

    print("Type 'exit' to quit.")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"exit","quit"}: break

        prompt = TEMPLATE.format(q=q)
        enc = tok(prompt, return_tensors="pt")
        enc = {k: v.to(model.device) for k, v in enc.items()}
        in_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        new_txt = tok.decode(out[0, in_len:], skip_special_tokens=True).strip()
        print(f"\nAlko-SFT: {new_txt}")

if __name__ == "__main__":
    main()
