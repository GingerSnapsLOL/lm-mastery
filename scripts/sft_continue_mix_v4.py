# scripts/sft_continue_ultra_v3.py
import os, argparse, hashlib, torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

RESP_TMPL = "Assistant:\n"  # must appear in prompt

def ultrachat_to_pc(ex):
    msgs = ex.get("messages", [])
    users = [m.get("content","").strip() for m in msgs if m.get("role")=="user"]
    assists = [m.get("content","").strip() for m in msgs if m.get("role")=="assistant"]
    prompt = f"User: {users[0]}\n{RESP_TMPL}" if users else f"User:\n{RESP_TMPL}"
    comp = ("\n" + assists[0]) if assists else ""
    return {"prompt": prompt, "completion": comp}

def hermes_to_pc(ex):
    # OpenHermes 2.5 uses instruction/chat style fields; handle common keys
    instr = (ex.get("instruction") or ex.get("prompt") or "").strip()
    out = (ex.get("output") or ex.get("response") or "").strip()
    return {"prompt": f"User: {instr}\n{RESP_TMPL}", "completion": "\n"+out}

def oasst_to_pc(ex):
    # oasst1 is a tree; top1/ENG subsets present flat messages; fall back if fields differ
    role = ex.get("role") or ex.get("message", {}).get("role")
    text = ex.get("text") or ex.get("message", {}).get("text") or ""
    if role == "assistant":
        return {"prompt": f"User:\n{RESP_TMPL}", "completion": "\n"+text.strip()}
    else:
        return {"prompt": f"User: {text.strip()}\n{RESP_TMPL}", "completion": "\n"}

def ascii_ratio(s, n=200):
    s = s[:n]
    if not s: return 0.0
    return sum(32 <= ord(c) < 127 for c in s) / len(s)

def clean_filter(r, min_chars=16, max_chars=4000):
    p, c = r["prompt"], r["completion"]
    if not c or len(c.strip()) < min_chars: return False
    L = len((p+c))
    if L < min_chars or L > max_chars: return False
    if ascii_ratio(c) < 0.7: return False
    return True

def dedup(ds):
    seen = set()
    keep = []
    for r in ds:
        h = hashlib.sha1((r["prompt"]+"||"+r["completion"]).encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h); keep.append(r)
    return Dataset.from_list(keep)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--out_dir",   required=True)
    ap.add_argument("--max_len",   type=int, default=1024)
    ap.add_argument("--bsz",       type=int, default=1)
    ap.add_argument("--ga",        type=int, default=16)
    ap.add_argument("--lr",        type=float, default=1e-5)
    ap.add_argument("--epochs",    type=int, default=1)
    ap.add_argument("--resume",    type=str, default="")
    ap.add_argument("--ultra_frac", type=float, default=0.6)
    ap.add_argument("--hermes_frac",type=float, default=0.25)
    ap.add_argument("--oasst_frac", type=float, default=0.15)
    args = ap.parse_args()

    # ---- load & map datasets (use num_proc=1 on Windows) ----
    ultra = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ultra = ultra.map(ultrachat_to_pc, remove_columns=ultra.column_names, num_proc=1)

    hermes = load_dataset("teknium/OpenHermes-2.5", split="train")
    hermes = hermes.map(hermes_to_pc, remove_columns=hermes.column_names, num_proc=1)

    # a small clean English top-1 set; fallback if missing
    try:
        oasst = load_dataset("SchubergPhilis/OpenAssistant-Top1-ENG-V1", split="train")
    except Exception:
        oasst = load_dataset("OpenAssistant/oasst1", split="train")
    oasst = oasst.map(oasst_to_pc, remove_columns=[c for c in oasst.column_names if c not in ("prompt","completion")], num_proc=1)

    # ---- filter & dedup ----
    ultra = ultra.filter(clean_filter)
    hermes = hermes.filter(clean_filter)
    oasst = oasst.filter(clean_filter)

    # (optional) cap sizes & mix by fractions
    total = min(len(ultra), 150_000) + min(len(hermes), 80_000) + min(len(oasst), 20_000)
    u_n = int(total * args.ultra_frac)
    h_n = int(total * args.hermes_frac)
    o_n = int(total * args.oasst_frac)

    ultra = ultra.shuffle(seed=42).select(range(min(u_n, len(ultra))))
    hermes = hermes.shuffle(seed=42).select(range(min(h_n, len(hermes))))
    oasst = oasst.shuffle(seed=42).select(range(min(o_n, len(oasst))))

    from datasets import concatenate_datasets
    ds = dedup(concatenate_datasets([ultra, hermes, oasst]).shuffle(seed=123))
    print(f"[data] final rows: {len(ds)}")

    # ---- tokenizer & model ----
    tok = AutoTokenizer.from_pretrained(args.base_ckpt, use_fast=False)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token":"[PAD]"})
    model = AutoModelForCausalLM.from_pretrained(
        args.base_ckpt,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else None),
    )
    model.resize_token_embeddings(len(tok))
    model.config.use_cache = False

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ctx = getattr(model.config, "max_position_embeddings", 1024)
    cfg = SFTConfig(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.ga,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        logging_steps=20,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        completion_only_loss=True,
        max_seq_length=min(args.max_len, ctx),
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,  # TRL 0.21+
    )

    resume = args.resume or None
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model()
    tok.save_pretrained(args.out_dir)
    print(f"[done] saved to {args.out_dir}")

if __name__ == "__main__":
    main()
