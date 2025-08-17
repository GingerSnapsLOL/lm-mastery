"""
IO utilities for paths, safe saving, and Windows compatibility
"""

import os
import json
from pathlib import Path
from typing import Optional, Union

# Base paths - can be overridden by environment variables
DEFAULT_DATA_DIR = "01-pretraining-pipeline/data/processed"
DEFAULT_CHECKPOINT_DIR = "01-pretraining-pipeline/results/checkpoints"
DEFAULT_TOKENIZER_DIR = "01-pretraining-pipeline/results/tokenizer"

def get_data_path(dataset_name: str = "train_big", data_dir: Optional[str] = None) -> str:
    """Get path to dataset directory"""
    if data_dir is None:
        data_dir = os.getenv("LM_MASTERY_DATA_DIR", DEFAULT_DATA_DIR)
    
    # Support both .arrow and .arrow_parquet formats
    arrow_path = os.path.join(data_dir, f"{dataset_name}.arrow")
    parquet_path = os.path.join(data_dir, f"{dataset_name}.arrow_parquet")
    
    # Prefer parquet if available (more robust)
    if os.path.exists(parquet_path):
        return parquet_path
    elif os.path.exists(arrow_path):
        return arrow_path
    else:
        raise FileNotFoundError(f"Dataset {dataset_name} not found in {data_dir}")

def get_checkpoint_path(model_name: str, checkpoint_dir: Optional[str] = None) -> str:
    """Get path to model checkpoint directory"""
    if checkpoint_dir is None:
        checkpoint_dir = os.getenv("LM_MASTERY_CHECKPOINT_DIR", DEFAULT_CHECKPOINT_DIR)
    
    checkpoint_path = os.path.join(checkpoint_dir, model_name)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)
    
    return checkpoint_path

def get_tokenizer_path(tokenizer_name: str = "spm", tokenizer_dir: Optional[str] = None) -> str:
    """Get path to tokenizer file"""
    if tokenizer_dir is None:
        tokenizer_dir = os.getenv("LM_MASTERY_TOKENIZER_DIR", DEFAULT_TOKENIZER_DIR)
    
    # Support both .model and full path
    if not tokenizer_name.endswith('.model'):
        tokenizer_path = os.path.join(tokenizer_dir, f"{tokenizer_name}.model")
    else:
        tokenizer_path = tokenizer_name
    
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
    return tokenizer_path

def safe_save(obj, filepath: Union[str, Path], **kwargs):
    """Safely save an object with error handling"""
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if hasattr(obj, 'save_pretrained'):
            obj.save_pretrained(str(filepath), **kwargs)
        elif hasattr(obj, 'save'):
            obj.save(str(filepath), **kwargs)
        else:
            raise ValueError(f"Object {type(obj)} has no save method")
        
        print(f"Successfully saved to: {filepath}")
        
    except Exception as e:
        print(f"Error saving to {filepath}: {e}")
        raise

def get_output_dir(base_name: str, output_dir: Optional[str] = None) -> str:
    """Get output directory for training results"""
    if output_dir is None:
        output_dir = os.getenv("LM_MASTERY_OUTPUT_DIR", "results")
    
    output_path = os.path.join(output_dir, base_name)
    os.makedirs(output_path, exist_ok=True)
    return output_path
