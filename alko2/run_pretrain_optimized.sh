#!/bin/bash

# Optimized Alko2 Pretraining Script
# Choose your configuration based on available hardware

# Configuration options:
# 1. 0.83B model (lighter, faster training)
# 2. 1.1B model (better quality, recommended)

# Model size selection
MODEL_SIZE="1.1b"  # Change to "0.83b" for lighter model

# Data configuration
MIX_YAML="path/to/your/data_mixture.yaml"  # Update this path
TOKENIZER_PATH="path/to/your/tokenizer"     # Update this path
OUTPUT_DIR="results/checkpoints/alko2_${MODEL_SIZE}"

# Training parameters
SEQ_LEN=2048
MICRO_BSZ=2
GRAD_ACCUM=64
LR=3e-4
MAX_STEPS=50000
WARMUP_RATIO=0.02
WEIGHT_DECAY=0.1

# DeepSpeed configuration (choose based on your setup)
# Option 1: ZeRO-3 with CPU offload (safest for large models)
DS_CONFIG="ds_zero3_optimized.json"
# Option 2: ZeRO-2 GPU-only (faster, requires more VRAM)
# DS_CONFIG="ds_zero2_gpu_only.json"

# Model configuration
if [ "$MODEL_SIZE" = "1.1b" ]; then
    MODEL_CONFIG="alko2_llama1b_enhanced.json"
else
    MODEL_CONFIG="alko2_llama1b.json"
fi

echo "Starting Alko2 pretraining with ${MODEL_SIZE} model..."
echo "Model config: ${MODEL_CONFIG}"
echo "DeepSpeed config: ${DS_CONFIG}"
echo "Output directory: ${OUTPUT_DIR}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run training with accelerate
accelerate launch \
    --config_file ds_zero3.json \
    pretrain.py \
    --model_config "${MODEL_CONFIG}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --mix_yaml "${MIX_YAML}" \
    --seq_len ${SEQ_LEN} \
    --micro_bsz ${MICRO_BSZ} \
    --grad_accum ${GRAD_ACCUM} \
    --lr ${LR} \
    --warmup_ratio ${WARMUP_RATIO} \
    --weight_decay ${WEIGHT_DECAY} \
    --output_dir "${OUTPUT_DIR}" \
    --max_steps ${MAX_STEPS} \
    --log_steps 20 \
    --save_steps 2000 \
    --deepspeed "${DS_CONFIG}"

echo "Training completed! Check results in: ${OUTPUT_DIR}"

