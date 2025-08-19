"""
Perplexity evaluation utilities
"""

import math
import torch
from typing import Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

def evaluate_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    max_length: int = 1024,
    batch_size: int = 1
) -> Dict[str, float]:
    """
    Evaluate model perplexity on a dataset
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        dataset: Dataset to evaluate on
        max_length: Maximum sequence length
        batch_size: Evaluation batch size
        
    Returns:
        Dictionary with evaluation metrics
    """
    
    # Create evaluation arguments
    eval_args = TrainingArguments(
        output_dir="tmp_eval",
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=0,
        report_to=[],
    )
    
    # Create trainer for evaluation
    trainer = Trainer(
        model=model,
        args=eval_args,
        data_collator=lambda x: {
            "input_ids": torch.tensor([seq[:max_length] for seq in x["input_ids"]], dtype=torch.long),
            "labels": torch.tensor([seq[:max_length] for seq in x["input_ids"]], dtype=torch.long)
        }
    )
    
    # Evaluate
    metrics = trainer.evaluate(eval_dataset=dataset)
    
    eval_loss = metrics["eval_loss"]
    perplexity = math.exp(eval_loss)
    
    return {
        "eval_loss": eval_loss,
        "perplexity": perplexity
    }
