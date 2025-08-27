#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TOKENIZER_DIR="$THIS_DIR/tokenizer"
MODEL_CFG="$THIS_DIR/alko2_model_config.json"
MIX_YAML="$THIS_DIR/data_mixture.yaml"
DS_CFG="$THIS_DIR/ds_zero3.json"
OUT="$THIS_DIR/out_pretrain"

mkdir -p "$OUT" "$TOKENIZER_DIR"

# Example: build tokenizer first
# python build_tokenizer.py --input_glob "/PATH/TO/corpus/**/*.txt" --out_dir "$TOKENIZER_DIR" --vocab_size 50000

accelerate launch --deepspeed "$DS_CFG" pretrain.py \
  --model_config "$MODEL_CFG" \
  --tokenizer_path "$TOKENIZER_DIR" \
  --mix_yaml "$MIX_YAML" \
  --seq_len 2048 \
  --micro_bsz 1 \
  --grad_accum 128 \
  --lr 3e-4 \
  --warmup_ratio 0.02 \
  --weight_decay 0.1 \
  --output_dir "$OUT" \
  --max_steps 10000 \
  --log_steps 20 \
  --save_steps 1000 \
  --deepspeed "$DS_CFG"
