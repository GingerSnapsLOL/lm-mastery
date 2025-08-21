#!/usr/bin/env python3




import math, argparse, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

def split_pc(messages):
    users = [m["content"] for m in messages if m.get("role")=="user"]
    assists = [m["content"] for m in messages if m.get("role")=="assistant"]
    return (users[0].strip() if users else ""), (assists[0].strip() if assists else "")

def tok_len(tok, text):  # length in tokens
    return len(tok(text, add_special_tokens=False)["input_ids"])

def build_batch(tok, pairs, seq_len, min_resp=32):
    """Return input_ids, attention_mask, labels with prompt masked and at least min_resp completion tokens."""
    batch_texts, prompt_lens = [], []
    for prompt, completion in pairs:
        pre = f"User: {prompt}\nAssistant:"
        comp = " " + completion if completion else ""

        p_ids = tok(pre, add_special_tokens=False)["input_ids"]
        c_ids = tok(comp, add_special_tokens=False)["input_ids"]

        # ensure at least min_resp completion tokens survive
        max_prompt_len = max(0, seq_len - min_resp - 1)  # -1 for safety
        if len(p_ids) + len(c_ids) > seq_len:
            # trim from the LEFT of the prompt (keep the end closest to completion)
            trim = max(0, (len(p_ids) + len(c_ids)) - seq_len)
            keep = max(0, len(p_ids) - trim)
            keep = min(keep, max_prompt_len)
            p_ids = p_ids[-keep:]

        # if still too long, truncate completion from RIGHT but keep min_resp
        total = len(p_ids) + len(c_ids)
        if total > seq_len:
            c_keep = max(min_resp, seq_len - len(p_ids))
            c_ids = c_ids[:c_keep]

        text_ids = p_ids + c_ids
        prompt_lens.append(len(p_ids))
        batch_texts.append(text_ids)

    maxlen = max(len(x) for x in batch_texts)
    # pad
    input_ids = []
    attn = []
    labels = []
    pad_id = tok.pad_token_id
    for ids, p_len in zip(batch_texts, prompt_lens):
        pad = [pad_id] * (maxlen - len(ids))
        arr = ids + pad
        msk = [1]*len(ids) + [0]*len(pad)
        lab = arr[:]  # copy
        # mask prompt part
        for i in range(p_len):
            lab[i] = -100
        input_ids.append(arr)
        attention = msk
        labels.append(lab)

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attn if (attn:=attention) else attention, dtype=torch.long),  # keep name 'attention'
        torch.tensor(labels, dtype=torch.long),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--max_eval", type=int, default=2000)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--min_resp", type=int, default=32)
    args = ap.parse_args()

    test = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
    pairs = []
    for r in (test if args.max_eval is None else test.select(range(min(args.max_eval, len(test))))):
        p,c = split_pc(r["messages"])
        if p or c: pairs.append((p,c))

    results = []
    for ckpt in args.ckpts:
        tok = AutoTokenizer.from_pretrained(ckpt, use_fast=False)
        if tok.pad_token is None:
            tok.add_special_tokens({"pad_token": tok.eos_token or "[PAD]"})
        model = AutoModelForCausalLM.from_pretrained(
            ckpt,
            torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else None),
        )
        if torch.cuda.is_available(): model.cuda()
        model.eval(); model.config.use_cache = False

        total_nll, total_tokens = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(pairs), args.bsz):
                batch = pairs[i:i+args.bsz]
                input_ids, attention_mask, labels = build_batch(tok, batch, args.seq_len, args.min_resp)
                input_ids = input_ids.to(model.device)
                attention_mask = attention_mask.to(model.device)
                labels = labels.to(model.device)

                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                # out.loss averaged over non-ignored tokens
                sup = (labels != -100).sum().item()
                if sup == 0:
                    continue
                total_nll += out.loss.item() * sup
                total_tokens += sup

        avg_nll = (total_nll / total_tokens) if total_tokens > 0 else float("nan")
        ppl = math.exp(avg_nll) if (avg_nll == avg_nll and avg_nll < 30) else float("inf")
        results.append((ckpt, avg_nll, ppl))

    print("\nModel".ljust(64), "Loss".ljust(10), "PPL")
    for name, loss, ppl in sorted(results, key=lambda x: (x[1] if x[1]==x[1] else 1e9)):
        loss_str = f"{loss:.4f}" if loss==loss else "nan"
        ppl_str = f"{ppl:.2f}" if ppl!=float('inf') else "inf"
        print(name.ljust(64), loss_str.ljust(10), ppl_str)

if __name__ == "__main__":
    main()
