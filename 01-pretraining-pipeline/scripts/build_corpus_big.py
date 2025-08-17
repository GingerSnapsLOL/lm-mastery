from datasets import load_dataset
import argparse, random, os

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", required=True)
parser.add_argument("--val_ratio", type=float, default=0.01)
parser.add_argument("--max_owt_docs", type=int, default=500_000)  # ~quick start; raise later
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
train_path = os.path.join(args.out_dir, "train_big.txt")
val_path   = os.path.join(args.out_dir, "validation_big.txt")

def dump(ds_iter, limit=None, fh=None):
    n=0
    for ex in ds_iter:
        text = ex.get("text") or ex.get("content") or ""
        if text: fh.write(text.replace("\r","") + "\n\n")
        n+=1
        if limit and n>=limit: break
    return n

with open(train_path, "w", encoding="utf-8") as ftr, open(val_path, "w", encoding="utf-8") as fval:
    # WikiText-103
    wt = load_dataset("wikitext", "wikitext-103-raw-v1")
    dump(wt["train"], fh=ftr)
    dump(wt["validation"], fh=fval)

    # OpenWebText
    try:
        owt = load_dataset("openwebtext", split="train", streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading OpenWebText dataset: {e}")
        owt = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

    # small held-out slice for val
    val_take = int(args.max_owt_docs * args.val_ratio)
    dump(owt.take(val_take), fh=fval)  # small val slice
    dump(owt.skip(val_take).take(args.max_owt_docs - val_take), fh=ftr)



print("Wrote:", train_path, val_path)