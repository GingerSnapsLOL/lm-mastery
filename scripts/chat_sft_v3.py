#!/usr/bin/env python3
"""
Interactive chat for SFT v3-style models.

Prompt template used (matches training):
  User: {question}
  Assistant:

Notes:
- We always pass attention_mask.
- We auto-truncate the prompt to fit model context.
- We DO NOT resize embeddings at inference. If your checkpoint
  already has a [PAD] token, great. If not, we fall back to using EOS
  as pad to avoid resizing warnings (attention_mask still prevents issues).
"""

import argparse
import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

TEMPLATE_TURN = "User: {q}\nAssistant:\n"
STOP_HINT = "\nUser:"  # optional cutting point in decoded text

def get_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():  # mac
        return "mps", None
    return "cpu", None

def build_dialog(history, max_turns=8):
    """
    history: list of (user, assistant) where assistant may be None for current turn
    Returns the concatenated prompt string ending with 'Assistant:\\n'
    """
    buf = []
    # keep only the last N turns
    for u, a in history[-max_turns:]:
        if a is None:
            buf.append(TEMPLATE_TURN.format(q=u))
        else:
            buf.append(TEMPLATE_TURN.format(q=u) + a.strip() + "\n")
    # ensure last turn ends with Assistant:\n (awaiting model continuation)
    if not buf or not buf[-1].endswith("Assistant:\n"):
        if history and history[-1][1] is None:
            buf[-1] = TEMPLATE_TURN.format(q=history[-1][0])
        else:
            # start a fresh empty prompt if somehow missing
            buf.append(TEMPLATE_TURN.format(q=""))
    return "".join(buf)

def left_truncate_to_ctx(tokenizer, text, ctx_limit, reserve=128):
    """
    Left-truncate tokenized prompt to fit within ctx_limit - reserve.
    Keeps the *end* of the prompt so we preserve the most recent turns.
    """
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    max_len = max(8, ctx_limit - max(8, reserve))
    if len(ids) <= max_len:
        return text, len(ids)
    # drop from the left
    ids = ids[-max_len:]
    return tokenizer.decode(ids, skip_special_tokens=False), len(ids)

def generate_reply(model, tokenizer, prompt_text, max_new=256, temperature=0.7, top_p=0.9, top_k=50,
                   repetition_penalty=1.1, no_repeat_ngram_size=3):
    device = next(model.parameters()).device
    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    enc = {k: v.to(device) for k, v in enc.items()}
    in_len = enc["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_text = tokenizer.decode(out[0, in_len:], skip_special_tokens=True)
    # Optional: cut at the next "User:" so we don't bleed into next turn
    cut = new_text.find(STOP_HINT)
    if cut != -1:
        new_text = new_text[:cut]
    return new_text.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path or HF id of your SFT checkpoint")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=3)
    ap.add_argument("--reserve_ctx", type=int, default=128, help="tokens reserved for the reply")
    ap.add_argument("--history_turns", type=int, default=8, help="how many past turns to keep")
    args = ap.parse_args()

    device, dtype = get_device_and_dtype()

    print(f"[load] checkpoint: {args.ckpt}")
    tok = AutoTokenizer.from_pretrained(args.ckpt, use_fast=False)

    # Avoid resizing at inference:
    # If checkpoint already has PAD, great. Otherwise, tie PAD to EOS to avoid OOV.
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt,
        torch_dtype=dtype,
        device_map=("auto" if torch.cuda.is_available() else None),
    )
    model.eval()
    model.config.use_cache = True

    ctx_limit = getattr(model.config, "max_position_embeddings", 1024)
    print(f"[info] context window: {ctx_limit} tokens")

    # conversation state: list of (user, assistant)
    history = []

    print("Type 'exit' to quit.\n")
    while True:
        user = input("You: ").strip()
        if user.lower() in {"exit", "quit"}:
            break

        # append pending user turn
        history.append((user, None))

        # build prompt from history and truncate if needed
        prompt = build_dialog(history, max_turns=args.history_turns)
        prompt, prompt_len = left_truncate_to_ctx(tok, prompt, ctx_limit, reserve=args.reserve_ctx)

        # generate
        reply = generate_reply(
            model, tok, prompt,
            max_new=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size
        )

        # store assistant turn
        history[-1] = (user, reply)
        print(f"\nAssistant: {reply}\n")

if __name__ == "__main__":
    main()