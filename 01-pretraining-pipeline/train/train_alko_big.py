
def main():
    import os, yaml, torch
    from datasets import load_from_disk
    from transformers import TrainingArguments, set_seed, Trainer
    from transformers import LlamaTokenizer
    from alko.configuration_alko import AlkoConfig
    from alko.modeling_alko import AlkoLLM
    import math

    # -------------------- enable TF32 --------------------
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
  

    # -------------------- paths & knobs --------------------
    TOK = r"01-pretraining-pipeline/results/tokenizer/spm.model"
    CFG = r"01-pretraining-pipeline/config/alko_base.yaml"

    # --- paths ---
    TRAIN_DS = r"01-pretraining-pipeline/data/processed/train_big.arrow"
    OUT      = r"01-pretraining-pipeline/results/checkpoints/run_alko_big"

    # --- token budget ---
    SEQ_LEN = 1024
    BATCH_SIZE = 1
    GRAD_ACCUM = 32
    TOKENS_PER_STEP = BATCH_SIZE * GRAD_ACCUM * SEQ_LEN  # 32,768
    TARGET_TOKENS = 300_000_000                          # change if you want
    MAX_STEPS = (TARGET_TOKENS + TOKENS_PER_STEP - 1) // TOKENS_PER_STEP  # ≈ 9,156

    set_seed(1488)

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
    model = AlkoLLM(cfg)

    # allow tokenizer growth if special tokens were added
    model.resize_token_embeddings(len(tok))
    model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False

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
        adam_beta1=0.9, 
        adam_beta2=0.95,
        max_grad_norm=0.25,  # Much more aggressive gradient clipping
        lr_scheduler_type="cosine",   # cosine LR schedule for smoother decay
        gradient_checkpointing=False,  # reduce memory usage for large models
        logging_steps=20,
        fp16=False,
        bf16=bf16_available,  # Use bf16 if available
        dataloader_num_workers=0,
        dataloader_persistent_workers=False,
        report_to=[],
    )

    # -------------------- Trainer subclass to disable checkpoints --------------------
    class NoSaveTrainer(Trainer):
        def _save_checkpoint(self, *args, **kwargs):
            return  # disable model/optimizer checkpointing entirely
        def _save(self, *args, **kwargs):
            return  # disable internal save hooks
        def save_model(self, *args, **kwargs):
            return  # disable final auto-save (we'll save manually)
        
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
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

    # -------------------- train --------------------
    print(f"Starting training...")
    print(f"Initial learning rate: {args.learning_rate}")
    print(f"Max gradient norm: {args.max_grad_norm}")
    
    # Use standard trainer.train() with our custom safety measures
    trainer.train()

    print("Training completed!")

    # -------------------- single final save --------------------
    os.makedirs(OUT, exist_ok=True)
    model.save_pretrained(OUT, safe_serialization=False)  # writes pytorch_model.bin
    tok.save_pretrained(OUT)
    print(f"Done. Final model saved to: {OUT}")
    print(f"Steps: {MAX_STEPS} (~{MAX_STEPS * TOKENS_PER_STEP:,} tokens)")



if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()            # Windows-safe
    main()