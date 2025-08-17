# LM Mastery - Language Model Training and Evaluation Toolkit

A comprehensive toolkit for training and evaluating language models with a focus on stability, reproducibility, and ease of use.

## 🏗️ Repository Structure

```
lm-mastery/
├─ pyproject.toml                    # Package configuration
├─ README.md                         # This file
├─ .vscode/                         # VS Code configuration
├─ data/                            # (optional) symlink to large data
├─ scripts/                         # Thin CLI wrappers only
│  ├─ pretrain_alko.py             # Train Alko model
│  ├─ pretrain_llama.py            # Train Llama model
│  ├─ eval_ppl.py                  # Evaluate perplexity
│  ├─ eval_compare_wiki.py         # Compare models on Wiki
│  ├─ generate.py                   # Generate text
│  └─ diagnostic.py                 # Model diagnostics
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

### 4. Evaluate Model

```bash
# Evaluate perplexity
python scripts/eval_ppl.py --checkpoint results/checkpoints/run_alko_big

# Run diagnostics
python scripts/diagnostic.py --checkpoint results/checkpoints/run_alko_big
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

### **Environment Variables**
```bash
export LM_MASTERY_DATA_DIR="path/to/data"
export LM_MASTERY_CHECKPOINT_DIR="path/to/checkpoints"
export LM_MASTERY_TOKENIZER_DIR="path/to/tokenizers"
export LM_MASTERY_OUTPUT_DIR="path/to/outputs"
```

## 🎯 Usage Examples

### **Custom Training Configuration**

```python
from lm_mastery.train.pretrain import TrainingConfig

config = TrainingConfig(
    model_name="my_model",
    learning_rate=1e-4,
    warmup_ratio=0.1,
    max_grad_norm=0.5,
    batch_size=2,
    gradient_accumulation_steps=16
)
```

### **Robust Dataset Loading**

```python
from lm_mastery.data.loaders import load_packed_dataset

# Automatically handles feature type issues
dataset = load_packed_dataset("path/to/dataset")
```

### **Safe Model Training**

```python
from lm_mastery.train.pretrain import create_trainer, init_model_weights

# Conservative initialization
init_model_weights(model, std=0.01)

# Safe trainer with loss clipping
trainer = create_trainer(model, tokenizer, dataset, config, output_dir)
```

## 🛠️ Development

### **Adding New Models**

1. **Create Model Module**: `src/lm_mastery/models/your_model/`
2. **Add Configuration**: `src/lm_mastery/configs/your_model.yaml`
3. **Create Training Script**: `scripts/pretrain_your_model.py`

### **Adding New Utilities**

1. **Create Module**: `src/lm_mastery/your_module/`
2. **Update `__init__.py`**: Add imports and exports
3. **Add Tests**: Create corresponding test files

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

### **Debugging**

```bash
# Run diagnostics
python scripts/diagnostic.py --checkpoint your/checkpoint

# Check dataset loading
python -c "from lm_mastery.data.loaders import load_train_dataset; print(load_train_dataset('train_big'))"
```

## 📚 Dependencies

- **PyTorch**: >=2.0.0
- **Transformers**: >=4.30.0
- **Datasets**: >=2.10.0
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

- **Hugging Face** for the transformers library
- **PyTorch** team for the deep learning framework
- **Datasets** team for data loading utilities

