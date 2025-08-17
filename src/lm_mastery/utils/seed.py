"""
Random seed utilities for reproducibility
"""

import torch
import random
import numpy as np
import os
from typing import Optional

def set_seed(seed: int = 42, deterministic: bool = True):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed set to {seed} (deterministic: {deterministic})")

def get_random_seed() -> int:
    """Get a random seed for experiments"""
    return random.randint(1, 1000000)
