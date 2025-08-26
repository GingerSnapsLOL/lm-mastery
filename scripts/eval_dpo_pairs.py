# eval_dpo_pairs.py
import math, torch, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

RESP_TMPL = "Assistant:\n"  # must match training
MAX_LEN = 4096               # set to your model context
BATCH = 4                    # increase if VRAM allows

def prepare_eval_split():
    # Try test/val split; fall back to sampling from train if needed
    for split in ["test_prefs", "val_prefs", "validation", "test"]:
        try:
            ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split)
            if len(ds) > 0:
                break
        except Exception:
            ds = None
    if ds is None:
        ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs").select(range(5000))
    # Reuse your formatter
    def _pick(block):
        if isinstance(block, str): return "", block.strip()
        if isinstance(block, list):
            u = next((m.get("content","") for m in block if m.get("role") in ("user","prompter")), "")
            a = next((m.get("content","") for m in block if m.get("role") == "assistant"), "")
            return u.strip(), a.strip()
        return "", ""
    def fmt(ex):
        uc, ac = _pick(ex.get("chosen", []))
        ur, ar = _pick(ex.get("rejected", []))
        user_text = (uc or ur or ex.get("prompt","")).strip()
        prompt = f"User: {user_text}\n{RESP_TMPL}"
        if ac and not ac.startswith("\n"): ac = "\n" + ac
        if ar and not ar.startswith("\n"): ar = "\n" + ar
        return {"prompt": prompt, "chosen": ac, "rejected": ar}
    ds = ds.map(fmt, remove_columns=ds.column_names)
    ds = ds.filter(lambda r: len(r["prompt"])>0 and len(r["chosen"].strip())>8 and len(r["rejected"].strip())>8 and (r["chosen"].strip() != r["rejected"].strip()))
    return ds

def seq_logprob(model, tok, prompts, completions):
    # Returns sum logprobs of each completion conditioned on its prompt
    # One-by-one (robust) or small batches to avoid OOM
    outs = []
    for i in range(0, len(prompts), BATCH):
        batch_p = prompts[i:i+BATCH]
        batch_c = completions[i:i+BATCH]
        # Concatenate prompt+completion then mask to only completion tokens
        inputs = tok([p + c for p,c in zip(batch_p, batch_c)],
                     return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(DEVICE)
        with torch.no_grad():
            out = model(**inputs)
            # Shift for next-token prediction
            logits = out.logits[:, :-1, :].float()
            ids = inputs.input_ids[:, 1:]
            logprobs = torch.log_softmax(logits, dim=-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)

        # Build mask for completion tokens only
        # We re-tokenize prompts alone to get prompt lengths
        prompt_ids = tok(batch_p, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).input_ids.to(DEVICE)
        prompt_len = (prompt_ids != tok.pad_token_id).sum(dim=1)
        attn_mask = (inputs.input_ids[:, 1:] != tok.pad_token_id)
        seq_sum = []
        for j in range(len(batch_p)):
            start = int(prompt_len[j].item()) - 1  # -1 due to shift
            start = max(start, 0)
            m = attn_mask[j]
            # Mask out prompt part
            m[:start] = False
            seq_sum.append(logprobs[j][m].sum().item())
        outs.extend(seq_sum)
    return np.array(outs, dtype=np.float64)

def eval_model_on_pairs(model_path, ds):
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=DTYPE, device_map="auto")
    model.eval()

    prompts = ds["prompt"]
    chosen  = ds["chosen"]
    reject  = ds["rejected"]

    lp_c = seq_logprob(model, tok, prompts, chosen)
    lp_r = seq_logprob(model, tok, prompts, reject)
    wins = (lp_c > lp_r)
    margins = lp_c - lp_r

    result = {
        "n": int(len(ds)),
        "acc": float(wins.mean()),
        "margin_mean": float(margins.mean()),
        "margin_median": float(np.median(margins)),
        "margin_p95": float(np.percentile(margins, 95)),
        "len_chosen_tokens_mean": float(np.mean([len(tok(c).input_ids) for c in chosen])),
        "len_reject_tokens_mean": float(np.mean([len(tok(r).input_ids) for r in reject])),
    }
    return result

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="List of model dirs or HF IDs")
    args = ap.parse_args()

    ds = prepare_eval_split()
    all_results = {}
    for m in args.models:
        print(f"Evaluating {m} …")
        res = eval_model_on_pairs(m, ds)
        all_results[m] = res
        print(json.dumps({m: res}, indent=2))

    # Pretty print summary table
    print("\nSUMMARY")
    rows = []
    for k,v in all_results.items():
        rows.append((k, v["n"], v["acc"], v["margin_mean"], v["margin_p95"]))
    rows.sort(key=lambda x: x[2], reverse=True)
    colw = [max(len(str(r[i])) for r in rows + [("model", "n", "acc", "Δmean", "Δp95")]) for i in range(5)]
    header = ["model","n","acc","Δmean","Δp95"]
    fmt = "  ".join("{:<" + str(colw[i]) + "}" for i in range(5))
    print(fmt.format(*header))
    for r in rows:
        print(fmt.format(r[0], r[1], f"{r[2]:.4f}", f"{r[3]:.2f}", f"{r[4]:.2f}"))