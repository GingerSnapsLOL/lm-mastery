import os
from datasets import load_dataset

out_dir = "01-pretraining-pipeline/data/raw"
os.makedirs(out_dir, exist_ok=True)
ds = load_dataset("wikitext", "wikitext-103-raw-v1")
# ds = load_dataset("wikitext", "wikitext-2-raw-v1")

def write_split(split_name):
    split = ds[split_name]["text"]
    path = os.path.join(out_dir, f"{split_name}.txt")
    with open(path, "w", encoding="utf-8") as f:
        for line in split:
            if line.strip():
                f.write(line.strip() + "\n")

for split in ["train", "validation", "test"]:
    write_split(split)

print(f"Wrote raw text to {out_dir}. Replace with your own corpus when ready.")
