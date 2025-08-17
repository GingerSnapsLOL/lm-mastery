# diagnostic.py
import argparse, os, math, sys
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, LlamaTokenizer

# Ensure your custom model is registered
import alko  # registers AlkoConfig/AlkoForCausalLM with Auto*

def global_grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            total += float(g.pow(2).sum())
    return math.sqrt(total) if total > 0 else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"01-pretraining-pipeline/results/checkpoints/run_alko_big")
    ap.add_argument("--ds",   default=r"01-pretraining-pipeline/data/processed/train_big.arrow")
    ap.add_argument("--tokenizer_dir", default=r"01-pretraining-pipeline/results/tokenizer")
    ap.add_argument("--sample_seqs", type=int, default=1000, help="how many sequences to scan for id range")
    ap.add_argument("--compute_grad_norm", action="store_true", help="do a single backward to report grad-norm")
    args = ap.parse_args()

    print("=== ENV ===")
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- Load dataset (packed Arrow folder) ---
    print("\n=== DATASET ===")
    ds = load_from_disk(args.ds)
    print("rows:", len(ds))
    first = ds[0]["input_ids"]
    seq_len = len(first)
    print("seq_len:", seq_len)

    # --- Load tokenizer ---
    print("\n=== TOKENIZER ===")
    try:
        tok = LlamaTokenizer.from_pretrained(args.ckpt)
        print("Loaded tokenizer from checkpoint")
    except Exception:
        tok = LlamaTokenizer(vocab_file=os.path.join(args.tokenizer_dir, "spm.model"))
        print("Loaded tokenizer from tokenizer_dir")
    if tok.eos_token is None:
        tok.add_special_tokens({"eos_token": "</s>", "bos_token": "<s>"})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print("vocab_size:", len(tok))
    print("eos_token_id:", tok.eos_token_id, "pad_token_id:", tok.pad_token_id)

    # --- Scan a subset for id range ---
    print("\n=== ID RANGE CHECK (first {} seqs) ===".format(min(args.sample_seqs, len(ds))))
    max_id = -1
    min_id = 10**9
    n = min(args.sample_seqs, len(ds))
    for i in range(n):
        ids = ds[i]["input_ids"]
        if not ids: continue
        if (m := max(ids)) > max_id: max_id = m
        if (m2 := min(ids)) < min_id: min_id = m2
    print("min_id:", min_id, "max_id:", max_id)
    vocab_ok = (max_id < len(tok)) and (min_id >= 0)
    print("vocab_ok:", vocab_ok)
    if not vocab_ok:
        print("!! Token IDs out of range. Check tokenizer vs dataset packing.")

    # --- Load model ---
    print("\n=== MODEL ===")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, torch_dtype=dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    model.config.use_cache = False  # simpler path for custom model
    print("params (M):", sum(p.numel() for p in model.parameters()) / 1e6)

    # --- One-batch forward diagnostics ---
    print("\n=== FORWARD DIAGNOSTICS (1 batch) ===")
    import numpy as np
    x = torch.tensor(np.array(ds[0:1]["input_ids"]), dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model(input_ids=x, labels=x)
        logits = out.logits
        loss = float(out.loss.detach().cpu())
    mean = float(logits.mean().detach().cpu())
    std  = float(logits.std().detach().cpu())
    mx   = float(logits.abs().max().detach().cpu())
    has_nan = torch.isnan(logits).any().item()
    print(f"logits mean/std/max_abs: {mean:.4f} / {std:.4f} / {mx:.4f}")
    print("has_nan_in_logits:", has_nan)
    print(f"one-batch CE loss (nats): {loss:.4f}")
    print(f"expected fresh-loss (nats) ~ ln(vocab) = {math.log(len(tok)):.2f}")

    # --- Optional grad norm test ---
    if args.compute_grad_norm:
        print("\n=== GRAD-NORM TEST (1 batch backward) ===")
        model.train()
        model.zero_grad(set_to_none=True)
        out = model(input_ids=x, labels=x)
        out.loss.backward()
        gnorm = global_grad_norm(model)
        print(f"global_grad_norm (pre-clip): {gnorm:.2f}")
        model.zero_grad(set_to_none=True)

    # --- Quick hints ---
    print("\n=== HINTS ===")
    if has_nan or mx > 1e3:
        print("* Logits are too large or contain NaNs → check causal mask dtype and additions (use float32 mask, add as att += mask.to(att.dtype)).")
    if not vocab_ok:
        print("* Tokenizer/dataset mismatch → repack with the same SentencePiece model you use at train time.")
    if loss > 100:
        print("* Loss >> ln(vocab). Something is off (mask/labels/dtype). Target fresh loss ~ 10–12 for vocab ~32k.")
    elif loss > 20:
        print("* Loss high but finite; continue debugging + lower LR, increase warmup, enable TF32, ensure grad clipping.")

if __name__ == "__main__":
    main()