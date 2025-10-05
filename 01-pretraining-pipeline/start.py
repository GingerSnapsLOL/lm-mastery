
# This porject about pretraining pipeline.
# --------------------------------
# Training & scaling
# FSDP / ZeRO / DeepSpeed, gradient checkpointing, activation/optimizer sharding
# Megatron-LM layout (tensor + pipeline parallel)
# Tokenizers (tiktoken/SentencePiece), dataset packing, mixed precision (bf16/fp8)

# Build data→train loop for a 1–3B decoder model (Megatron/DeepSpeed). Log throughput, stability, loss curves.

# 1. Prepare data
# 2. Train model
# 3. Evaluate model


# Version 2
# decoder-only transformer with the common 2024–2025 upgrades (FlashAttention-3,
#  GQA, SwiGLU, RMSNorm, RoPE with long-context scaling, QK-Norm).





    




