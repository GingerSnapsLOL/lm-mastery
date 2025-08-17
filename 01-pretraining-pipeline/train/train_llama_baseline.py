# train_llama_baseline.py
import os, math, yaml, torch, glob
from datasets import load_from_disk, load_dataset, Dataset, Features, Sequence, Value
from transformers import (
    LlamaConfig, LlamaForCausalLM, LlamaTokenizer,
    TrainingArguments, Trainer, set_seed
)

# ---------- paths ----------
TOK = r"01-pretraining-pipeline/results/tokenizer/spm.model"
TRAIN = r"01-pretraining-pipeline/data/processed/train_big.arrow"
OUT = r"01-pretraining-pipeline/results/checkpoints/run_llama_baseline_109M"

def load_train_dataset(train_dir: str):
    """Load training dataset with fallback strategies for 'List' feature type issue"""
    # 1) try normal path (datasets v3)
    try:
        return load_from_disk(train_dir)
    except Exception as e:
        print("[train] load_from_disk failed ->", e)
        
        # Try to fix the old 'List' feature type issue
        if "Feature type 'List' not found" in str(e):
            print("[train] attempting to fix 'List' feature type issue...")
            try:
                # Look for dataset_info.json and fix the feature type
                info_file = os.path.join(train_dir, "dataset_info.json")
                if os.path.exists(info_file):
                    import json
                    with open(info_file, 'r') as f:
                        info = json.load(f)
                    
                    print(f"[train] original features: {info.get('features', 'Not found')}")
                    
                    # Replace 'List' with 'Sequence' in features
                    if 'features' in info:
                        features_str = json.dumps(info['features'])
                        if '"List"' in features_str:
                            features_str = features_str.replace('"List"', '"Sequence"')
                            info['features'] = json.loads(features_str)
                            
                            # Write back the fixed info
                            with open(info_file, 'w') as f:
                                json.dump(info, f, indent=2)
                            
                            print(f"[train] fixed features: {info['features']}")
                            print("[train] fixed dataset_info.json, trying to load again...")
                            return load_from_disk(train_dir)
                        else:
                            print("[train] no 'List' found in features, issue might be elsewhere")
                    else:
                        print("[train] no features found in dataset_info.json")
                else:
                    print("[train] dataset_info.json not found")
            except Exception as fix_e:
                print(f"[train] feature type fix failed -> {fix_e}")

    # 2) try parquet fallback (if your packer kept shards)
    pq_dir = train_dir + "_parquet"
    pq_files = glob.glob(os.path.join(pq_dir, "*.parquet"))
    if pq_files:
        print("[train] loading parquet shards fallback")
        try:
            # Try to load parquet with explicit features to avoid 'List' type issue
            return load_dataset("parquet", data_files=pq_files, split="train", features=Features({"input_ids": Sequence(Value("int32"))}))
        except Exception as e:
            print(f"[train] parquet loading failed -> {e}")
            print("[train] trying to read parquet files directly...")
            
            # Try to read parquet files directly with pandas
            try:
                import pandas as pd
                all_rows = []
                for f in sorted(pq_files):
                    try:
                        df = pd.read_parquet(f)
                        if 'input_ids' in df.columns:
                            all_rows.extend(df['input_ids'].tolist())
                    except Exception as read_e:
                        print(f"[train] failed to read parquet {f}: {read_e}")
                        continue
                
                if all_rows:
                    print(f"[train] loaded {len(all_rows)} sequences from parquet files")
                    feats = Features({"input_ids": Sequence(Value("int32"))})
                    return Dataset.from_dict({"input_ids": all_rows}, features=feats)
            except ImportError:
                print("[train] pandas not available, skipping parquet direct read")
            except Exception as pd_e:
                print(f"[train] pandas parquet read failed -> {pd_e}")

    # 3) last resort: read arrow file(s) directly and rebuild features explicitly
    print("[train] reading raw .arrow files and rebuilding dataset")
    arrow_files = glob.glob(os.path.join(train_dir, "data", "*.arrow")) or glob.glob(os.path.join(train_dir, "*.arrow"))
    assert arrow_files, f"No .arrow files found under {train_dir}"
    
    all_rows = []
    for f in sorted(arrow_files):
        try:
            import pyarrow.ipc as pa_ipc
            with open(f, "rb") as fh:
                rb = pa_ipc.open_file(fh)
                tbl = rb.read_all()
            col = tbl.column("input_ids")  # list<int32>
            all_rows.extend(col.to_pylist())  # list[list[int]]
        except Exception as e:
            print(f"[train] failed to read {f}: {e}")
            continue
    
    if not all_rows:
        raise ValueError("No data could be loaded from any source")
    
    print(f"[train] loaded {len(all_rows)} sequences from arrow files")
    
    # Use the correct feature type for current datasets version
    feats = Features({"input_ids": Sequence(Value("int32"))})
    return Dataset.from_dict({"input_ids": all_rows}, features=feats)

# ---------- data ----------
ds = load_train_dataset(TRAIN)
print(f"[train] loaded dataset with {len(ds)} sequences")

# fixed-length collator
class FixedLenCollator:
    def __call__(self, feats):
        import torch
        x = torch.tensor([f["input_ids"] for f in feats], dtype=torch.long)
        return {"input_ids": x, "labels": x.clone()}

collator = FixedLenCollator()

# ---------- tokenizer ----------
tok = LlamaTokenizer(vocab_file=TOK)
if tok.eos_token is None:
    tok.add_special_tokens({"eos_token":"</s>","bos_token":"<s>"})
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# ---------- model config (~109M params) ----------
cfg = LlamaConfig(
    vocab_size=len(tok),             # 32000
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,          # head_dim = 64
    intermediate_size=2048,          # ~2.67x
    rms_norm_eps=1e-5,
    rope_theta=5e5,
    max_position_embeddings=1024,
    tie_word_embeddings=True,
    use_cache=False,                 # for grad ckpt & stability
)

model = LlamaForCausalLM(cfg)
model.resize_token_embeddings(len(tok))

# Initialize weights more conservatively to prevent extreme values
def init_weights(module):
    if isinstance(module, torch.nn.Linear):
        # Much more conservative initialization
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)  # Reduced from default
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Embedding):
        # Much more conservative initialization
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)  # Reduced from default

print("[train] applying conservative weight initialization...")
model.apply(init_weights)

# ---------- training knobs ----------
SEQ_LEN = 1024
BATCH_SIZE = 1
GRAD_ACCUM = 32
TOKENS_PER_STEP = BATCH_SIZE * GRAD_ACCUM * SEQ_LEN  # 32,768
TARGET_TOKENS = 300_000_000                           # ~9,156 steps
MAX_STEPS = (TARGET_TOKENS + TOKENS_PER_STEP - 1) // TOKENS_PER_STEP

# numerics
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
set_seed(42)

# Check if bf16 is available
bf16_available = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

print(f"=== TRAINING CONFIG ===")
print(f"BF16 available: {bf16_available}")
print(f"Will use: {'BF16' if bf16_available else 'FP16' if torch.cuda.is_available() else 'FP32'}")
print(f"Learning rate: 5e-5")
print(f"Warmup ratio: 15%")
print(f"Max grad norm: 0.25")
print(f"Adam betas: (0.9, 0.95)")

args = TrainingArguments(
    output_dir=OUT,
    overwrite_output_dir=True,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    max_steps=MAX_STEPS,
    learning_rate=5e-5,  # Much lower learning rate for stability
    warmup_ratio=0.15,   # 15% warmup for better stability
    weight_decay=0.001,  # Much lower weight decay
    adam_beta1=0.9, adam_beta2=0.95,
    max_grad_norm=0.25,  # Much more aggressive gradient clipping
    lr_scheduler_type="cosine",
    logging_steps=20,
    bf16=bf16_available,  # Use bf16 if available
    fp16=False,
    gradient_checkpointing=True,
    dataloader_num_workers=0,        # Windows-safe
    report_to=[],
)

# optional: tiny z-loss for logit-scale stability
def z_loss(logits, coeff=1e-4):
    return (logits.logsumexp(dim=-1) ** 2).mean() * coeff

# Trainer with CE loss (default). If you want z-loss, override compute_loss.
class SafeCETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Safety check: clip loss if it's too extreme
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: Loss is {loss}, replacing with safe value")
            loss = torch.tensor(10.0, device=loss.device, dtype=loss.dtype)
        
        # Clip extremely high loss values
        if loss > 100.0:
            print(f"WARNING: Loss {loss} too high, clipping to 100")
            loss = torch.clamp(loss, max=100.0)
        
        # Optional: add z-loss for logit-scale stability (uncomment if needed)
        # loss = loss + z_loss(outputs.logits)
        
        return (loss, outputs) if return_outputs else loss

# train
trainer = SafeCETrainer(model=model, args=args, data_collator=collator, train_dataset=ds)
model.config.pad_token_id = tok.pad_token_id
model.gradient_checkpointing_enable()
trainer.train()

# save once at the end (plain PyTorch weights)
os.makedirs(OUT, exist_ok=True)
model.save_pretrained(OUT, safe_serialization=False)
tok.save_pretrained(OUT)
print(f"Saved baseline to: {OUT}")
print(f"Steps: {MAX_STEPS} (~{MAX_STEPS * TOKENS_PER_STEP:,} tokens)")
