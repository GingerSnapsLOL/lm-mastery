#!/usr/bin/env python3
import math, argparse, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

RESP_TMPL = "Assistant:\n"   # must appear in prompt
SEQ_DEFAULT = 1024

def split_pc(messages):
    u = next((m["content"] for m in messages if m.get("role")=="user"), "")
    a = next((m["content"] for m in messages if m.get("role")=="assistant"), "")
    return u.strip(), a.strip()

def build_masked_batch(tok, pairs, seq_len, min_resp=32):
    """Return input_ids, attention_mask, labels (prompt masked)"""
    enc_seqs = []
    prompt_lens = []
    pad_id = tok.pad_token_id

    for prompt, completion in pairs:
        pre = f"User: {prompt}\n{RESP_TMPL}"
        comp = "\n" + completion if completion else ""

        p_ids = tok(pre, add_special_tokens=False)["input_ids"]
        c_ids = tok(comp, add_special_tokens=False)["input_ids"]

        # Trim prompt so at least min_resp completion tokens remain
        max_prompt_len = max(0, seq_len - min_resp - 1)
        if len(p_ids) + len(c_ids) > seq_len:
            trim = (len(p_ids) + len(c_ids)) - seq_len
            keep = max(0, len(p_ids) - trim)
            keep = min(keep, max_prompt_len)
            p_ids = p_ids[-keep:]

        if len(p_ids) + len(c_ids) > seq_len:
            c_keep = max(min_resp, seq_len - len(p_ids))
            c_ids = c_ids[:c_keep]

        ids = p_ids + c_ids
        enc_seqs.append(ids)
        prompt_lens.append(len(p_ids))

    maxlen = max(len(x) for x in enc_seqs)
    input_ids, attention_mask, labels = [], [], []
    for ids, p_len in zip(enc_seqs, prompt_lens):
        pad_len = maxlen - len(ids)
        arr = ids + [pad_id]*pad_len
        att = [1]*len(ids) + [0]*pad_len
        lab = arr[:]  # copy
        for i in range(p_len):  # mask prompt
            lab[i] = -100
        input_ids.append(arr); attention_mask.append(att); labels.append(lab)

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="List of model checkpoints")
    ap.add_argument("--max_eval", type=int, default=1500)
    ap.add_argument("--seq_len", type=int, default=SEQ_DEFAULT)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--min_resp", type=int, default=32)
    args = ap.parse_args()

    test = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
    pairs = []
    for r in test.select(range(min(args.max_eval, len(test)))):
        u, a = split_pc(r["messages"])
        if u or a:
            pairs.append((u, a))

    results = []
    for ckpt in args.ckpts:
        tok = AutoTokenizer.from_pretrained(ckpt, use_fast=False)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token  # avoid resizing embeddings for eval
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
                input_ids, attention_mask, labels = build_masked_batch(
                    tok, batch, args.seq_len, args.min_resp
                )
                input_ids = input_ids.to(model.device)
                attention_mask = attention_mask.to(model.device)
                labels = labels.to(model.device)

                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                sup = (labels != -100).sum().item()
                if sup == 0:  # skip degenerate batches
                    continue
                total_nll += out.loss.item() * sup
                total_tokens += sup

        avg_nll = (total_nll / total_tokens) if total_tokens > 0 else float("nan")
        ppl = math.exp(avg_nll) if (avg_nll==avg_nll and avg_nll<30) else float("inf")
        results.append((ckpt, avg_nll, ppl))

    print("\nModel".ljust(64), "Loss".ljust(10), "PPL")
    for name, loss, ppl in sorted(results, key=lambda x: (x[1] if x[1]==x[1] else 1e9)):
        loss_s = f"{loss:.4f}" if loss==loss else "nan"
        ppl_s  = f"{ppl:.2f}"   if ppl!=float('inf') else "inf"
        print(name.ljust(64), loss_s.ljust(10), ppl_s)

if __name__ == "__main__":
    main()
