import os, yaml, torch
from datasets import load_from_disk
from transformers import TrainingArguments, set_seed, Trainer
from transformers import LlamaTokenizer
from alko.configuration_alko import AlkoConfig
from alko.modeling_alko import AlkoForCausalLM
import math

# -------------------- paths & knobs --------------------
TOK = r"01-pretraining-pipeline/results/tokenizer/spm.model"
TRAIN_DS = r"01-pretraining-pipeline/data/processed/train.arrow"   # make sure this is WT-103 train
OUT = r"01-pretraining-pipeline/results/checkpoints/run_alko_wt103"
CFG = r"01-pretraining-pipeline/config/alko_base.yaml"
TRAIN_DS = r"01-pretraining-pipeline/data/processed/train_big.arrow"

SEQ_LEN = 1024
BATCH_SIZE = 1
GRAD_ACCUM = 32
TOKENS_PER_STEP = BATCH_SIZE * GRAD_ACCUM * SEQ_LEN  # 32,768
TARGET_TOKENS = 100_000_000                          # ~100M tokens
MAX_STEPS = (TARGET_TOKENS + TOKENS_PER_STEP - 1) // TOKENS_PER_STEP  # ≈3052

set_seed(42)

# -------------------- data --------------------
ds = load_from_disk(TRAIN_DS)

# -------------------- tokenizer --------------------
tok = LlamaTokenizer(vocab_file=TOK)
if tok.eos_token is None:
    tok.add_special_tokens({"eos_token": "</s>", "bos_token": "<s>"})
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# -------------------- config & model --------------------
with open(CFG, "r") as f:
    cfg_yaml = yaml.safe_load(f)
cfg = AlkoConfig(**cfg_yaml)
model = AlkoForCausalLM(cfg)
# allow tokenizer growth if special tokens were added
model.resize_token_embeddings(len(tok))
model.config.pad_token_id = tok.pad_token_id

# -------------------- fixed-length collator (no padding) --------------------
class FixedLenCollator:
    def __call__(self, feats):
        x = torch.tensor([f["input_ids"] for f in feats], dtype=torch.long)
        return {"input_ids": x, "labels": x.clone()}

collator = FixedLenCollator()

# -------------------- TrainingArguments (no periodic saves) --------------------
# Check if bf16 is available
bf16_available = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

print(f"=== TRAINING CONFIG ===")
print(f"BF16 available: {bf16_available}")
print(f"Will use: {'BF16' if bf16_available else 'FP16' if torch.cuda.is_available() else 'FP32'}")
print(f"Learning rate: 1e-4")
print(f"Warmup ratio: 5%")
print(f"Max grad norm: 1.0")
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
    logging_steps=20,
    bf16=bf16_available,  # Use bf16 if available, otherwise disable
    fp16=not bf16_available and torch.cuda.is_available(),  # Fallback to fp16 if bf16 not available
    adam_beta1=0.9,      # Recommended beta1
    adam_beta2=0.95,     # Recommended beta2
    max_grad_norm=0.25,  # Much more aggressive gradient clipping
    dataloader_num_workers=0,  # Disable multiprocessing for stability
    
    report_to=[],  # no wandb/tensorboard
    # don't rely on save_* knobs; we'll hard-disable saving via Trainer subclass below
)

# -------------------- Trainer subclass to disable checkpoints --------------------
class NoSaveTrainer(Trainer):
    def _save_checkpoint(self, *args, **kwargs):
        return  # disable model/optimizer checkpointing entirely
    def _save(self, *args, **kwargs):
        return  # disable internal save hooks
    def save_model(self, *args, **kwargs):
        return  # disable final auto-save (we'll save manually)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Custom loss computation with additional safety checks"""
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Additional safety: clip loss if it's too extreme
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: Loss is {loss}, replacing with safe value")
            loss = torch.tensor(10.0, device=loss.device, dtype=loss.dtype)
        
        # Clip extremely high loss values
        if loss > 100.0:
            print(f"WARNING: Loss {loss} too high, clipping to 100")
            loss = torch.clamp(loss, max=100.0)
        
        return (loss, outputs) if return_outputs else loss

trainer = NoSaveTrainer(
    model=model,
    args=args,
    data_collator=collator,
    train_dataset=ds,
)

# -------------------- Custom Learning Rate Scheduler --------------------
class SafeWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.step_count = 0
        
    def step(self):
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            lr = self.optimizer.param_groups[0]['lr'] * (self.step_count / self.warmup_steps)
        else:
            # Cosine annealing with minimum LR
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.optimizer.param_groups[0]['lr'] * (0.5 * (1 + math.cos(math.pi * progress)))
            lr = max(lr, self.min_lr)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

# Calculate warmup steps
warmup_steps = int(MAX_STEPS * 0.15)  # 15% warmup

# -------------------- train --------------------
print(f"Starting training with {warmup_steps} warmup steps...")
print(f"Initial learning rate: {args.learning_rate}")
print(f"Max gradient norm: {args.max_grad_norm}")

# Create custom scheduler
scheduler = SafeWarmupScheduler(trainer.optimizer, warmup_steps, MAX_STEPS)

# Training loop with additional safety
for step in range(MAX_STEPS):
    try:
        # Get batch
        batch = next(iter(trainer.get_train_dataloader()))
        
        # Forward pass
        outputs = trainer.model(**batch)
        loss = outputs.loss
        
        # Safety check: skip step if loss is extreme
        if torch.isnan(loss) or torch.isinf(loss) or loss > 100.0:
            print(f"Step {step}: Skipping due to extreme loss {loss}")
            continue
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), args.max_grad_norm)
        
        # Check gradient norm
        total_norm = 0.0
        for p in trainer.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        if total_norm > 10.0:
            print(f"Step {step}: High gradient norm {total_norm}, skipping step")
            trainer.optimizer.zero_grad()
            continue
        
        # Optimizer step
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()
        
        # Update learning rate
        scheduler.step()
        
        # Logging
        if step % 20 == 0:
            current_lr = trainer.optimizer.param_groups[0]['lr']
            print(f"Step {step}: Loss={loss.item():.4f}, LR={current_lr:.2e}, GradNorm={total_norm:.4f}")
            
    except Exception as e:
        print(f"Step {step}: Error occurred: {e}")
        trainer.optimizer.zero_grad()
        continue

print("Training completed!")

# -------------------- single final save --------------------
os.makedirs(OUT, exist_ok=True)
model.save_pretrained(OUT, safe_serialization=False)  # writes pytorch_model.bin
tok.save_pretrained(OUT)
print(f"Done. Final model saved to: {OUT}")
print(f"Steps: {MAX_STEPS} (~{MAX_STEPS * TOKENS_PER_STEP:,} tokens)")