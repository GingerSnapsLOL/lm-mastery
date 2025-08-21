#!/usr/bin/env python3
import argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": tok.eos_token or "[PAD]"})
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else None),
        device_map=("auto" if torch.cuda.is_available() else None),
    )
    model.config.use_cache = True
    model.eval()

    print("Type 'exit' to quit.")
    history = []

    while True:
        user = input("\nYou: ").strip()
        if user.lower() in {"exit", "quit"}: break
        # Keep the same template you used during SFT:
        prompt = f"User: {user}\nAssistant:"
        enc = tok(prompt, return_tensors="pt")  # add_special_tokens=True by default
        input_ids = enc["input_ids"].to(model.device)
        attn = enc["attention_mask"].to(model.device)
        in_len = input_ids.shape[1]

        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

        new_tokens = out[0, in_len:]
        text = tok.decode(new_tokens, skip_special_tokens=True).strip()
        print(f"\nAlko-SFT: {text}")

if __name__ == "__main__":
    main()
