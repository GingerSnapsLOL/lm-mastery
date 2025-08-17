# 🧱 LLM Pretraining Starter (Small-Scale)

This is a starter scaffold to pretrain a **small decoder-only model** (e.g., 125M–350M params) **from scratch** using Hugging Face Transformers + (optional) DeepSpeed.
It includes scripts for **tokenizer training**, **dataset preparation/packing**, **training**, and **evaluation**.

> Start small on a single GPU, then scale up to multi-GPU with DeepSpeed/FSDP later.

## Quickstart

```bash
# 1) Create env (conda or venv)
conda create -n llm-pretrain python=3.10 -y
conda activate llm-pretrain

# 2) Install deps
pip install -r shared/requirements.txt

# 3) Prepare data (downloads a small text dataset as example)
python 01-pretraining-pipeline/scripts/prepare_data.py

# 4) Train tokenizer on your corpus
python 01-pretraining-pipeline/scripts/build_tokenizer.py   --input_glob "01-pretraining-pipeline/data/raw/*.txt"   --vocab_size 32000   --model_type bpe   --output "01-pretraining-pipeline/results/tokenizer"

# 5) Pack dataset to fixed-length token sequences
python 01-pretraining-pipeline/scripts/pack_dataset.py   --tokenizer_dir "01-pretraining-pipeline/results/tokenizer"   --input_glob "01-pretraining-pipeline/data/raw/*.txt"   --seq_len 1024   --output_path "01-pretraining-pipeline/data/processed/train.arrow"

# 6) Train from scratch (HF Trainer)
python 01-pretraining-pipeline/train/train.py   --tokenizer_dir "01-pretraining-pipeline/results/tokenizer"   --processed_path "01-pretraining-pipeline/data/processed/train.arrow"   --model_config "01-pretraining-pipeline/config/model_config.yaml"   --output_dir "01-pretraining-pipeline/results/checkpoints/run1"   --deepspeed "01-pretraining-pipeline/train/ds_zero2.json"

# (Optional) Evaluate perplexity
python 01-pretraining-pipeline/eval/eval.py   --model_dir "01-pretraining-pipeline/results/checkpoints/run1"   --tokenizer_dir "01-pretraining-pipeline/results/tokenizer"
```

## Notes
- The default config targets **single-GPU** training; use DeepSpeed config to enable ZeRO-2 offload/sharding.
- Swap the example dataset with your own corpus for meaningful results.
- For speed on Linux + NVIDIA, consider installing FlashAttention (optional) and set `use_flash_attn=True` in the training script.
