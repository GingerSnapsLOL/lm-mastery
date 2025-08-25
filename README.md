# LM Mastery - Language Model Training and Evaluation Toolkit

A comprehensive toolkit for training and evaluating language models with a focus on stability, reproducibility, and ease of use. Includes pre-training, supervised fine-tuning (SFT), evaluation, and interactive chat capabilities.

## 🏗️ Repository Structure

```
lm-mastery/
├─ pyproject.toml                    # Package configuration
├─ README.md                         # This file
├─ .vscode/                         # VS Code configuration
├─ data/                            # (optional) symlink to large data
├─ scripts/                         # CLI wrappers and training scripts
│  ├─ pretrain_alko.py             # Train Alko model
│  ├─ pretrain_llama.py            # Train Llama model
│  ├─ eval_ppl.py                  # Evaluate perplexity
│  ├─ eval_compare_wiki.py         # Compare models on Wiki
│  ├─ generate.py                   # Generate text
│  ├─ diagnostic.py                 # Model diagnostics
│  ├─ sft_train.py                  # General SFT training (Dolly, UltraChat)
│  ├─ sft_continue_ultra_v2.py     # Continue UltraChat SFT training v2
│  ├─ sft_continue_ultra_v3.py     # Continue UltraChat SFT training v3
│  ├─ sft_continue_mix_v4.py       # Mixed dataset SFT training (UltraChat + OpenHermes + OASST)
│  ├─ dpo_train.py                  # Direct Preference Optimization training
│  ├─ chat_sft.py                   # Basic chat interface for SFT models
│  ├─ chat_sft_v2.py               # Chat interface for v2-style models
│  ├─ chat_sft_v3.py               # Advanced chat interface with conversation history
│  └─ eval_sft_ultra_v2.py         # Evaluate SFT models on UltraChat test set
├─ src/
│  └─ lm_mastery/                  # Main package
│     ├─ __init__.py               # Package initialization
│     ├─ configs/                  # Configuration files
│     │  ├─ alko_base.yaml        # Alko model config
│     │  └─ train_llama_109m.yaml # Llama training config
│     ├─ models/                   # Model implementations
│     │  ├─ alko/                  # Custom Alko model
│     │  │  ├─ __init__.py
│     │  │  ├─ configuration_alko.py
│     │  │  └─ modeling_alko.py
│     │  └─ llama/                 # Llama model tweaks
│     ├─ data/                     # Data utilities
│     │  ├─ packers.py             # Arrow/parquet packers
│     │  ├─ loaders.py             # Robust dataset loading
│     │  └─ collators.py           # Data collation
│     ├─ train/                    # Training utilities
│     │  ├─ pretrain.py            # Shared trainer setup
│     │  └─ schedules.py           # Learning rate schedules
│     ├─ eval/                     # Evaluation utilities
│     │  ├─ ppl.py                 # Perplexity evaluation
│     │  └─ compare.py             # Model comparison
│     ├─ gen/                      # Generation utilities
│     │  └─ sampler.py             # Text generation
│     ├─ sft/                      # SFT utilities
│     │  ├─ formatters.py          # Dataset formatting functions
│     │  └─ __init__.py
│     └─ utils/                    # Utility functions
│        ├─ io.py                  # Path handling, safe save
│        ├─ seed.py                # Random seed management
│        ├─ device.py              # Device setup, dtype
│        └─ logging.py             # Logging configuration
└─ 01-pretraining-pipeline/        # (optional legacy)
   ├─ data/processed/…             # Keep datasets here
   └─ results/checkpoints/…        # Keep checkpoints here
```

## 🚀 Quick Start

### 1. Install Package (Editable)

```bash
# Install in editable mode
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

### 2. Train Alko Model

```bash
# Basic training
python scripts/pretrain_alko.py

# Custom parameters
python scripts/pretrain_alko.py \
  --model-name alko_custom \
  --train-dataset train_big \
  --max-length 1024 \
  --batch-size 1 \
  --grad-accum 32 \
  --learning-rate 5e-5 \
  --warmup-ratio 0.15
```

### 3. Train Llama Model

```bash
# Basic training
python scripts/pretrain_llama.py

# Custom parameters
python scripts/pretrain_llama.py \
  --model-name llama_custom \
  --train-dataset train_big \
  --max-length 1024 \
  --batch-size 1 \
  --grad-accum 32 \
  --learning-rate 5e-5 \
  --warmup-ratio 0.15
```

### 4. Supervised Fine-Tuning (SFT)

#### **Basic SFT Training**
```bash
# Train on Dolly dataset
python scripts/sft_train.py \
  --base_ckpt "results/checkpoints/run_alko_big" \
  --out_dir "results/checkpoints/run_alko_dolly" \
  --dataset dolly \
  --bsz 1 \
  --ga 16 \
  --lr 2e-5 \
  --epochs 3

# Train on UltraChat dataset
python scripts/sft_train.py \
  --base_ckpt "results/checkpoints/run_llama_baseline_109M" \
  --out_dir "results/checkpoints/run_llama_ultrachat" \
  --dataset ultrachat \
  --bsz 1 \
  --ga 16 \
  --lr 2e-5 \
  --epochs 3
```

#### **Continue UltraChat Training**
```bash
# Continue from v2 to v3
python scripts/sft_continue_ultra_v3.py \
  --base_ckpt "results/checkpoints/run_llama_sft_ultra_v2" \
  --out_dir "results/checkpoints/run_llama_sft_ultra_v3" \
  --max_len 2048 \
  --bsz 1 \
  --ga 32 \
  --lr 1e-5 \
  --epochs 2
```

#### **Mixed Dataset Training (Recommended)**
```bash
# Train on mixed high-quality datasets
python scripts/sft_continue_mix_v4.py \
  --base_ckpt "results/checkpoints/run_llama_sft_ultra_v3" \
  --out_dir "results/checkpoints/run_llama_sft_mix_v4" \
  --max_len 2048 \
  --bsz 1 \
  --ga 32 \
  --lr 2e-5 \
  --epochs 2 \
  --ultra_frac 0.6 \
  --hermes_frac 0.25 \
  --oasst_frac 0.15
```

### **5. Direct Preference Optimization (DPO)**

#### **DPO Training from SFT Model**
```bash
# Train with UltraFeedback dataset (recommended)
python scripts/dpo_train.py \
  --base_ckpt "results/checkpoints/run_llama_sft_mix_v4" \
  --out_dir "results/checkpoints/run_llama_dpo_ultra" \
  --dataset ultrafeedback \
  --max_len 1024 \
  --max_prompt_len 768 \
  --bsz 1 \
  --ga 32 \
  --lr 5e-6 \
  --beta 0.1 \
  --epochs 1

# Train with custom preference dataset
python scripts/dpo_train.py \
  --base_ckpt "results/checkpoints/run_llama_sft_mix_v4" \
  --out_dir "results/checkpoints/run_llama_dpo_custom" \
  --dataset custom \
  --max_len 1024 \
  --bsz 1 \
  --ga 32 \
  --lr 5e-6 \
  --beta 0.2 \
  --epochs 1
```

### **6. Interactive Chat**

#### **Basic Chat Interface**
```bash
# Simple chat for any SFT model
python scripts/chat_sft.py \
  --ckpt "results/checkpoints/run_llama_sft_ultra_v3"
```

#### **Advanced Chat with History**
```bash
# Chat with conversation history and context management
python scripts/chat_sft_v3.py \
  --ckpt "results/checkpoints/run_llama_sft_mix_v4" \
  --max_new_tokens 512 \
  --temperature 0.7 \
  --history_turns 8
```

### 6. Evaluation

#### **Perplexity Evaluation**
```bash
# Evaluate model perplexity
python scripts/eval_ppl.py --checkpoint results/checkpoints/run_alko_big

# Run diagnostics
python scripts/diagnostic.py --checkpoint results/checkpoints/run_alko_big
```

#### **SFT Model Evaluation**
```bash
# Compare multiple SFT models on UltraChat test set
python scripts/eval_sft_ultra_v2.py \
  --ckpts "results/checkpoints/run_llama_sft_ultra_v2" \
           "results/checkpoints/run_llama_sft_ultra_v3" \
           "results/checkpoints/run_llama_sft_mix_v4" \
  --max_eval 1500 \
  --seq_len 1024
```

## 🔧 Key Features

### **Robust Dataset Loading**
- **Automatic Feature Type Fixing**: Handles `List` → `Sequence` compatibility issues
- **Multiple Fallback Strategies**: Arrow → Parquet → Raw file reading
- **Cross-version Compatibility**: Works with different datasets library versions

### **Training Stability**
- **Conservative Weight Initialization**: Prevents extreme initial values
- **Loss Clipping**: Automatically handles NaN/Inf values
- **Gradient Clipping**: Aggressive clipping for stability
- **Mixed Precision**: BF16/FP16 support with automatic detection

### **SFT Training Pipeline**
- **Completion-Only Loss**: Focuses training on response quality
- **Mixed Dataset Training**: Combines UltraChat, OpenHermes, and OASST
- **Progressive Training**: v2 → v3 → mix_v4 pipeline for quality improvement
- **Context-Aware Length**: Automatically respects model position embeddings

### **DPO Training**
- **Preference Learning**: Trains on chosen vs rejected response pairs
- **Multiple Datasets**: UltraFeedback, Identity, and custom preference datasets
- **Self-Reference**: Uses same model as reference for efficient training
- **Quality Improvement**: Builds upon SFT models for better response alignment

### **Interactive Chat**
- **Conversation History**: Maintains context across multiple turns
- **Smart Truncation**: Preserves recent conversation while fitting context
- **Template Matching**: Uses exact training prompt format
- **Memory Efficient**: Automatic context management

### **Comprehensive Evaluation**
- **Perplexity Metrics**: Standard language model evaluation
- **SFT-Specific Metrics**: Completion-only loss evaluation
- **Model Comparison**: Side-by-side performance analysis
- **Robust Testing**: Handles edge cases and errors gracefully

### **Modular Architecture**
- **Reusable Components**: Shared training, evaluation, and data utilities
- **Configuration Files**: YAML-based model and training configurations
- **Environment Variables**: Configurable paths via environment variables

## 📁 Data Organization

### **Dataset Paths**
- **Training**: `01-pretraining-pipeline/data/processed/train_big.arrow`
- **Validation**: `01-pretraining-pipeline/data/processed/val_big.arrow`
- **Parquet Fallback**: `01-pretraining-pipeline/data/processed/train_big.arrow_parquet/`

### **Model Checkpoints**
- **Alko**: `01-pretraining-pipeline/results/checkpoints/run_alko_big/`
- **Llama**: `01-pretraining-pipeline/results/checkpoints/run_llama_baseline_109M/`
- **SFT UltraChat v2**: `results/checkpoints/run_llama_sft_ultra_v2/`
- **SFT UltraChat v3**: `results/checkpoints/run_llama_sft_ultra_v3/`
- **SFT Mixed v4**: `results/checkpoints/run_llama_sft_mix_v4/`

### **Environment Variables**
```bash
export LM_MASTERY_DATA_DIR="path/to/data"
export LM_MASTERY_CHECKPOINT_DIR="path/to/checkpoints"
export LM_MASTERY_TOKENIZER_DIR="path/to/tokenizers"
export LM_MASTERY_OUTPUT_DIR="path/to/outputs"
```

## 🎯 Usage Examples

### **SFT Training Pipeline**

```bash
# Step 1: Initial SFT on UltraChat
python scripts/sft_train.py \
  --base_ckpt "results/checkpoints/run_llama_baseline_109M" \
  --out_dir "results/checkpoints/run_llama_sft_ultra_v1" \
  --dataset ultrachat \
  --epochs 3

# Step 2: Continue training with better parameters
python scripts/sft_continue_ultra_v3.py \
  --base_ckpt "results/checkpoints/run_llama_sft_ultra_v1" \
  --out_dir "results/checkpoints/run_llama_sft_ultra_v3" \
  --max_len 2048 \
  --ga 32 \
  --lr 1e-5 \
  --epochs 2

# Step 3: Mixed dataset training for final quality
python scripts/sft_continue_mix_v4.py \
  --base_ckpt "results/checkpoints/run_llama_sft_ultra_v3" \
  --out_dir "results/checkpoints/run_llama_sft_mix_v4" \
  --max_len 2048 \
  --ga 32 \
  --lr 2e-5 \
  --epochs 2

# Step 4: DPO training for preference alignment
python scripts/dpo_train.py \
  --base_ckpt "results/checkpoints/run_llama_sft_mix_v4" \
  --out_dir "results/checkpoints/run_llama_dpo_final" \
  --dataset ultrafeedback \
  --max_len 1024 \
  --bsz 1 \
  --ga 32 \
  --lr 5e-6 \
  --beta 0.1 \
  --epochs 1
```

### **Interactive Chat Sessions**

```bash
# Start chat with your trained model
python scripts/chat_sft_v3.py \
  --ckpt "results/checkpoints/run_llama_sft_mix_v4" \
  --temperature 0.7 \
  --max_new_tokens 512

# Example conversation:
# You: What is machine learning?
# Assistant: Machine learning is a subset of artificial intelligence...
# You: Can you explain neural networks?
# Assistant: Neural networks are computational models inspired by...
```

### **Model Evaluation**

```bash
# Evaluate perplexity
python scripts/eval_ppl.py --checkpoint results/checkpoints/run_alko_big

# Compare SFT models
python scripts/eval_sft_ultra_v2.py \
  --ckpts "results/checkpoints/run_llama_sft_ultra_v2" \
           "results/checkpoints/run_llama_sft_ultra_v3" \
           "results/checkpoints/run_llama_sft_mix_v4"
```

## 🛠️ Development

### **Adding New SFT Datasets**

1. **Create Formatter**: Add function to `src/lm_mastery/sft/formatters.py`
2. **Update Scripts**: Modify SFT training scripts to use new dataset
3. **Test Integration**: Verify with small dataset subset

### **Adding New Preference Datasets**

1. **Create Formatter**: Add dataset loading and formatting logic to `scripts/dpo_train.py`
2. **Update Choices**: Add new dataset option to the `--dataset` argument
3. **Test Format**: Ensure chosen/rejected pairs are properly formatted

### **Adding New Models**

1. **Create Model Module**: `src/lm_mastery/models/your_model/`
2. **Add Configuration**: `src/lm_mastery/configs/your_model.yaml`
3. **Create Training Script**: `scripts/pretrain_your_model.py`

### **Running Tests**

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Code formatting
black src/ scripts/

# Linting
flake8 src/ scripts/
```

## 🔍 Troubleshooting

### **Common Issues**

1. **Feature Type Errors**: Automatically handled by robust loaders
2. **Training Instability**: Use conservative initialization and loss clipping
3. **Memory Issues**: Enable gradient checkpointing and use mixed precision
4. **SFT Quality Issues**: Use completion-only loss and mixed datasets

### **Debugging**

```bash
# Run diagnostics
python scripts/diagnostic.py --checkpoint your/checkpoint

# Check dataset loading
python -c "from lm_mastery.data.loaders import load_train_dataset; print(load_train_dataset('train_big'))"

# Test SFT formatters
python -c "from lm_mastery.sft.formatters import alpaca_to_text; print(alpaca_to_text({'instruction': 'test', 'response': 'test'}))"
```

## 📚 Dependencies

- **PyTorch**: >=2.0.0
- **Transformers**: >=4.30.0
- **Datasets**: >=2.10.0
- **TRL**: >=0.21.0 (for SFT training)
- **PyArrow**: >=10.0.0
- **Pandas**: >=1.5.0
- **NumPy**: >=1.21.0
- **PyYAML**: >=6.0

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch
3. **Make** your changes
4. **Add** tests if applicable
5. **Submit** a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face** for the transformers and TRL libraries
- **PyTorch** team for the deep learning framework
- **Datasets** team for data loading utilities
- **OpenHermes** and **OASST** teams for high-quality instruction datasets

