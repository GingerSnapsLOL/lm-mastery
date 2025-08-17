"""
Learning rate schedules and utilities
"""

import math
import torch
from typing import Optional

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1
):
    """
    Create a learning rate scheduler with cosine annealing and warmup
    
    Args:
        optimizer: The optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        num_cycles: Number of cosine cycles
        last_epoch: Last epoch number
    
    Returns:
        Learning rate scheduler
    """
    from transformers import get_cosine_schedule_with_warmup as hf_cosine_schedule
    
    return hf_cosine_schedule(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        last_epoch=last_epoch
    )

def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1
):
    """
    Create a learning rate scheduler with linear decay and warmup
    
    Args:
        optimizer: The optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        last_epoch: Last epoch number
    
    Returns:
        Learning rate scheduler
    """
    from transformers import get_linear_schedule_with_warmup as hf_linear_schedule
    
    return hf_linear_schedule(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        last_epoch=last_epoch
    )

def calculate_warmup_steps(total_steps: int, warmup_ratio: float) -> int:
    """Calculate warmup steps from ratio"""
    return int(total_steps * warmup_ratio)

def calculate_learning_rate_schedule(
    initial_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1
) -> torch.Tensor:
    """
    Calculate learning rate schedule manually
    
    Args:
        initial_lr: Initial learning rate
        warmup_steps: Number of warmup steps
        total_steps: Total number of steps
        min_lr_ratio: Minimum LR as ratio of initial LR
    
    Returns:
        Tensor of learning rates for each step
    """
    lrs = []
    min_lr = initial_lr * min_lr_ratio
    
    for step in range(total_steps):
        if step < warmup_steps:
            # Linear warmup
            lr = initial_lr * (step + 1) / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr = min_lr + (initial_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        lrs.append(lr)
    
    return torch.tensor(lrs)
