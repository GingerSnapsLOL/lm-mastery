# 01-pretraining-pipeline/scripts/pack_dataset.py
import argparse, os, glob, numpy as np
import sentencepiece as spm
from datasets import Dataset, Features, Sequence, Value

parser = argparse.ArgumentParser()
parser.add_argument("--tokenizer_dir", required=True)
parser.add_argument("--input_glob", required=True)
parser.add_argument("--seq_len", type=int, default=1024)
parser.add_argument("--output_path", required=True)
args = parser.parse_args()

sp = spm.SentencePieceProcessor()
tok_path = os.path.join(args.tokenizer_dir, "spm.model")
assert os.path.exists(tok_path), f"Missing tokenizer model: {tok_path}"
sp.load(tok_path)

eos_id = sp.eos_id()
if eos_id < 0:  # fallback if missing
    eos_id = 2

ids = []
for path in glob.glob(args.input_glob):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.extend(sp.encode(line, out_type=int))
            ids.append(eos_id)

n = (len(ids) // args.seq_len) * args.seq_len
ids = ids[:n]
arr = np.asarray(ids, dtype=np.int32).reshape(-1, args.seq_len)

features = Features({"input_ids": Sequence(Value("int32"))})
ds = Dataset.from_dict({"input_ids": arr.tolist()}, features=features)
os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
ds.save_to_disk(args.output_path)

print(f"Saved packed dataset to {args.output_path} | sequences={len(ds)} | seq_len={args.seq_len}")
