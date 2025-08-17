import argparse, os, sys
from datasets import load_dataset

def dump_text(iterable, fh, limit=None):
    n = 0
    for ex in iterable:
        # FineWeb-Edu has a 'text' field. Fall back if needed.
        txt = ex.get("text") or ex.get("content") or ""
        if txt:
            fh.write(txt.replace("\r", "") + "\n\n")
            n += 1
            if limit and n >= limit:
                break
    return n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fw_docs", type=int, default=500_000, help="FineWeb-Edu docs for train+val")
    ap.add_argument("--val_ratio", type=float, default=0.01, help="val split fraction")
    ap.add_argument("--add_wt103", action="store_true", help="try to append WikiText-103")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "train_big.txt")
    val_path   = os.path.join(args.out_dir, "validation_big.txt")

    with open(train_path, "w", encoding="utf-8") as ftr, open(val_path, "w", encoding="utf-8") as fval:
        # ---------- FineWeb-Edu (script-less; OK on datasets v3) ----------
        # Use a smaller config if you prefer: name="sample-10BT" | "sample-100BT"
        fw = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)  # v3-friendly
        val_take = max(1, int(args.fw_docs * args.val_ratio))
        wrote_val = dump_text(fw.take(val_take), fval)
        wrote_train = dump_text(fw.skip(val_take).take(args.fw_docs - val_take), ftr)
        print(f"[fineweb-edu] train docs: {wrote_train} | val docs: {wrote_val}")

        # ---------- (Optional) WikiText-103 (may fail on v3 if it relies on a script) ----------
        if args.add_wt103:
            try:
                wt = load_dataset("wikitext", "wikitext-103-raw-v1")
                wt_train = wt["train"]
                wt_val = wt["validation"]
                wrote_wt_val = dump_text(wt_val, fval)
                wrote_wt_train = dump_text(wt_train, ftr)
                print(f"[wikitext-103] train docs: {wrote_wt_train} | val docs: {wrote_wt_val}")
            except Exception as e:
                print(f"[wikitext-103] skipped (datasets v3 likely removed scripts): {e}", file=sys.stderr)

    print("Wrote:", train_path, val_path)

if __name__ == "__main__":
    main()