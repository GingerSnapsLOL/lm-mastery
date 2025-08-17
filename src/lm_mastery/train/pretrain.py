"""
Shared pretraining utilities and configurations
"""

import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any
from transformers import (
    TrainingArguments, Trainer, 
    AutoModelForCausalLM, AutoTokenizer
)

@dataclass
class TrainingConfig:
    """Training configuration dataclass"""
    # Model
    model_name: str
    model_type: str = "alko"  # "alko" or "llama"
    
    # Data
    train_dataset: str = "train_big"
    val_dataset: str = "val_big"
    max_length: int = 1024
    
    # Training
    batch_size: int = 1
    gradient_accumulation_steps: int = 32
    max_steps: int = 9156
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.15
    weight_decay: float = 0.001
    max_grad_norm: float = 0.25
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    
    # Mixed precision
    use_bf16: bool = True
    use_fp16: bool = False
    
    # Other
    seed: int = 42
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 0
    
    def to_training_args(self, output_dir: str) -> TrainingArguments:
        """Convert to TrainingArguments"""
        return TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            warmup_ratio=self.warmup_ratio,
            weight_decay=self.weight_decay,
            adam_beta1=self.adam_beta1,
            adam_beta2=self.adam_beta2,
            max_grad_norm=self.max_grad_norm,
            lr_scheduler_type="cosine",
            logging_steps=20,
            bf16=self.use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=self.use_fp16 and torch.cuda.is_available() and not self.use_bf16,
            gradient_checkpointing=self.gradient_checkpointing,
            dataloader_num_workers=self.dataloader_num_workers,
            report_to=[],
        )

class SafeTrainer(Trainer):
    """Trainer with safety measures for numerical stability"""
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss with safety checks"""
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
        
        return (loss, outputs) if return_outputs else loss

def create_trainer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset,
    config: TrainingConfig,
    output_dir: str,
    data_collator=None
) -> SafeTrainer:
    """Create a trainer with the given configuration"""
    
    # Create training arguments
    training_args = config.to_training_args(output_dir)
    
    # Create trainer
    trainer = SafeTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    # Setup model
    model.config.pad_token_id = tokenizer.pad_token_id
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    return trainer

def init_model_weights(model: AutoModelForCausalLM, std: float = 0.01):
    """Initialize model weights conservatively"""
    def init_weights(module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
    
    print(f"Applying conservative weight initialization (std={std})...")
    model.apply(init_weights)
