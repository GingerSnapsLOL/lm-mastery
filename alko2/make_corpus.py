import os, sys, re, argparse, shutil, json, gzip
from pathlib import Path
from typing import List, Dict, Iterable, Optional

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:
    pa = None
    pq = None

def iter_files(roots: List[str], exts: List[str]) -> Iterable[Path]:
    for r in roots:
        if not r: 
            continue
        p = Path(r)
        if not p.exists():
            continue
        for ext in exts:
            yield from p.rglob(f"*{ext}")

def classify(path: Path) -> str:
    s = str(path).lower()
    if any(k in s for k in ["fineweb", "refinedweb", "commoncrawl", "ccnet"]):
        return "fineweb"
    if any(k in s for k in ["book", "gutenberg"]):
        return "books"
    if any(k in s for k in ["code", "github", "the-stack", "starcoder"]):
        return "code"
    if any(k in s for k in ["math", "openwebmath", "gsm"]):
        return "math"
    if any(k in s for k in ["wiki", "wikipedia"]):
        return "wikipedia"
    return "misc"

def safe_copy_or_link(src: Path, dst: Path, use_symlinks: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if use_symlinks:
        try:
            os.symlink(src, dst)
            return
        except Exception:
            pass  # fallback to copy
    shutil.copy2(src, dst)

def jsonl_to_txt(src: Path, dst: Path, text_keys: List[str]):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with (gzip.open if src.suffix == ".gz" else open)(src, "rt", encoding="utf-8", errors="ignore") as f, \
         open(dst, "w", encoding="utf-8") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                for k in text_keys:
                    if k in obj and isinstance(obj[k], str):
                        out.write(obj[k].strip() + "\n")
                        break
            except Exception:
                continue

def parquet_to_txt(src: Path, dst: Path, text_keys: List[str]):
    if pq is None:
        print(f"[WARN] pyarrow not installed; cannot convert {src}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        table = pq.read_table(src)
        cols = {c.lower(): c for c in table.schema.names}
        key = None
        for k in text_keys:
            if k.lower() in cols:
                key = cols[k.lower()]; break
        if key is None:
            print(f"[WARN] none of text_keys {text_keys} found in {src}")
            return
        with open(dst, "w", encoding="utf-8") as out:
            arr = table[key]
            for v in arr.to_pylist():
                if isinstance(v, str):
                    out.write(v.strip() + "\n")
    except Exception as e:
        print(f"[WARN] failed to read {src}: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="Folders to scan (recursively) for data")
    ap.add_argument("--out_dir", required=True, help="Unified corpus output folder")
    ap.add_argument("--use_symlinks", action="store_true", help="Try to symlink instead of copy")
    ap.add_argument("--text_keys", default="text,content,body,document", help="Comma list of candidate text keys for JSONL/Parquet")
    ap.add_argument("--max_files_per_bucket", type=int, default=0, help="Optional cap per bucket (0 = no cap)")
    ap.add_argument("--write_yaml", action="store_true", help="Write data_mixture.yaml next to out_dir")
    args = ap.parse_args()

    text_keys = [k.strip() for k in args.text_keys.split(",") if k.strip()]

    out = Path(args.out_dir)
    for bucket in ["fineweb","wikipedia","books","code","math","misc"]:
        (out / bucket).mkdir(parents=True, exist_ok=True)

    exts = [".txt", ".jsonl", ".jsonl.gz", ".ndjson", ".ndjson.gz", ".parquet", ".arrow"]
    counts = {k:0 for k in ["fineweb","wikipedia","books","code","math","misc"]}
    cap = args.max_files_per_bucket

    for src in iter_files(args.roots, exts):
        bucket = classify(src)
        if cap and counts[bucket] >= cap:
            continue
        if src.suffix.lower() == ".txt":
            dst = out / bucket / src.name
            safe_copy_or_link(src, dst, args.use_symlinks)
            counts[bucket]+=1
        elif src.suffix.lower() in [".jsonl", ".ndjson"] or any(str(src).lower().endswith(s) for s in [".jsonl.gz", ".ndjson.gz"]):
            dst = out / bucket / (src.stem.replace(".gz","") + ".txt")
            jsonl_to_txt(src, dst, text_keys)
            counts[bucket]+=1
        elif src.suffix.lower() == ".parquet":
            dst = out / bucket / (src.stem + ".txt")
            parquet_to_txt(src, dst, text_keys)
            counts[bucket]+=1
        elif src.suffix.lower() == ".arrow":
            print(f"[SKIP] Raw .arrow file (needs dataset loader): {src}")
        else:
            print(f"[SKIP] Unknown ext: {src}")

    print("[OK] Bucket counts:", counts)

    if args.write_yaml:
        yaml = f'''temperature: 0.5
sources:
  - name: fineweb
    type: text
    data_files: "{str(out / 'fineweb' / '**' / '*.txt').replace('\\', '/')}"
    weight: 0.60
    newline_separated: true
  - name: wikipedia
    type: text
    data_files: "{str(out / 'wikipedia' / '**' / '*.txt').replace('\\', '/')}"
    weight: 0.10
    newline_separated: true
  - name: books
    type: text
    data_files: "{str(out / 'books' / '**' / '*.txt').replace('\\', '/')}"
    weight: 0.10
    newline_separated: true
  - name: code
    type: text
    data_files: "{str(out / 'code' / '**' / '*.txt').replace('\\', '/')}"
    weight: 0.10
    newline_separated: true
  - name: math
    type: text
    data_files: "{str(out / 'math' / '**' / '*.txt').replace('\\', '/')}"
    weight: 0.10
    newline_separated: true
'''
        with open(out.parent / "data_mixture.yaml", "w", encoding="utf-8") as f:
            f.write(yaml)
        print("[OK] Wrote:", out.parent / "data_mixture.yaml")

if __name__ == "__main__":
    main()
