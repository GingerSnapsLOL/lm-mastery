# pack_dataset_o200k_harmony.py

#python 01-pretraining-pipeline/scripts/pack_dataset_o200k_harmony.py \
#  --input_glob "01-pretraining-pipeline/data/raw/*.txt" \
#  --seq_len 2048 \
#  --output_path "01-pretraining-pipeline/data/processed/train_o200k"


import argparse, os, glob, numpy as np
import tiktoken
from datasets import Dataset, Features, Sequence, Value

parser = argparse.ArgumentParser()
parser.add_argument("--input_glob", required=True, help="Glob for raw .txt files")
parser.add_argument("--seq_len", type=int, default=2048)
parser.add_argument("--encoding", default="o200k_harmony")
parser.add_argument("--output_path", required=True, help="HF dataset directory")
args = parser.parse_args()

enc = tiktoken.get_encoding(args.encoding)
vocab_size = enc.n_vocab

# Read all lines and encode with tiktoken
ids = []
newline = enc.encode("\n")
for path in glob.glob(args.input_glob):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            # encode_ordinary forbids special tokens (good for plain text pretrain)
            ids.extend(enc.encode_ordinary(line))
            # put a newline boundary between samples
            ids.extend(newline)

# Trim to multiple of seq_len and reshape
n = (len(ids) // args.seq_len) * args.seq_len
ids = ids[:n]
arr = np.asarray(ids, dtype=np.int32).reshape(-1, args.seq_len)

# Save as HF dataset with a single "input_ids" sequence column
features = Features({"input_ids": Sequence(Value("int32"))})
ds = Dataset.from_dict({"input_ids": arr.tolist()}, features=features)
os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
ds.save_to_disk(args.output_path)

# Store tiny metadata for training script
meta_path = os.path.join(args.output_path, "tiktoken_meta.txt")
with open(meta_path, "w", encoding="utf-8") as m:
    m.write(f"encoding={args.encoding}\n")
    m.write(f"vocab_size={vocab_size}\n")
    m.write(f"seq_len={args.seq_len}\n")

print(f"Saved packed dataset to {args.output_path} | sequences={len(ds)} | seq_len={args.seq_len} | vocab={vocab_size}")