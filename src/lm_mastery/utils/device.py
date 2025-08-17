"""
Device and numerical precision utilities
"""

import torch
from typing import Optional, Tuple

def setup_device(device: Optional[str] = None, allow_tf32: bool = True) -> torch.device:
    """Setup device with optimal settings"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Enable TF32 for better performance on Ampere+ GPUs
    if allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    return torch.device(device)

def get_dtype(device: torch.device, prefer_bf16: bool = True) -> torch.dtype:
    """Get optimal dtype for the device"""
    if device.type == "cuda":
        if prefer_bf16 and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            return torch.float16
    else:
        return torch.float32

def get_mixed_precision_config(device: torch.device, prefer_bf16: bool = True) -> Tuple[bool, bool]:
    """Get mixed precision configuration for training"""
    if device.type != "cuda":
        return False, False
    
    bf16_available = torch.cuda.is_bf16_supported()
    
    if prefer_bf16 and bf16_available:
        return True, False  # bf16=True, fp16=False
    elif torch.cuda.is_available():
        return False, True  # bf16=False, fp16=True
    else:
        return False, False  # bf16=False, fp16=False

def print_device_info(device: torch.device):
    """Print device information and capabilities"""
    print(f"=== DEVICE INFO ===")
    print(f"Device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
        print(f"BF16 support: {torch.cuda.is_bf16_supported()}")
        print(f"TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
    else:
        print("Using CPU")
