import os, argparse, glob
import numpy as np
import sentencepiece as spm
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset, concatenate_datasets, Dataset

def write_parquet_shard(shard_ids_np: np.ndarray, out_dir: str, shard_idx: int):
    """
    shard_ids_np: [N, seq_len] int32
    Writes one Parquet file with column 'input_ids' = list<int32>.
    """
    n, seq_len = shard_ids_np.shape
    values = pa.array(shard_ids_np.reshape(-1), type=pa.int32())
    offsets = pa.array(np.arange(0, (n * seq_len) + 1, seq_len, dtype=np.int32))
    list_array = pa.ListArray.from_arrays(offsets, values)  # [N] of list<int32>
    table = pa.Table.from_arrays([list_array], names=["input_ids"])
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"shard-{shard_idx:05d}.parquet")
    pq.write_table(table, path)
    return path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer_dir", required=True)
    ap.add_argument("--input_file", required=True)   # big .txt
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--output_path", required=True)  # e.g. data/processed/train_big.arrow
    ap.add_argument("--shard_seqs", type=int, default=10000)  # sequences per shard (~40MB per shard)
    args = ap.parse_args()

    # load SPM
    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(args.tokenizer_dir, "spm.model"))
    eos_id = sp.eos_id()
    if eos_id < 0:
        eos_id = 2

    buf = []  # rolling flat token buffer
    shard_idx = 0
    seq_len = args.seq_len
    tgt_tokens = args.shard_seqs * seq_len

    tmp_dir = args.output_path + "_parquet"
    os.makedirs(tmp_dir, exist_ok=True)
    written = []

    def flush():
        nonlocal buf, shard_idx
        if len(buf) < tgt_tokens:
            return
        n = (len(buf) // seq_len) * seq_len
        if n == 0:
            return
        arr = np.asarray(buf[:n], dtype=np.int32).reshape(-1, seq_len)
        buf = buf[n:]
        path = write_parquet_shard(arr, tmp_dir, shard_idx)
        written.append(path)
        shard_idx += 1

    # stream lines → tokens
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            ids = sp.encode(s, out_type=int)
            ids.append(eos_id)
            buf.extend(ids)
            if len(buf) >= tgt_tokens:
                flush()

    # final partial shard
    n = (len(buf) // seq_len) * seq_len
    if n > 0:
        arr = np.asarray(buf[:n], dtype=np.int32).reshape(-1, seq_len)
        path = write_parquet_shard(arr, tmp_dir, shard_idx)
        written.append(path)

    # build a single HF dataset folder from all parquet shards
    files = sorted(glob.glob(os.path.join(tmp_dir, "*.parquet")))
    ds = load_dataset("parquet", data_files=files, split="train")  # v3-friendly
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    ds.save_to_disk(args.output_path)

    # optional: cleanup parquet shards to save disk space
    # import shutil; shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"Saved packed dataset to {args.output_path}")
    print(f"Shards: {len(files)} | seq_len={seq_len}")

if __name__ == "__main__":
    main()
