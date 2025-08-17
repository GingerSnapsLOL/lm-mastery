"""
Data collation utilities for training
"""

import torch
from typing import Dict, List, Any

class FixedLenCollator:
    """Fixed-length sequence collator for causal language modeling"""
    
    def __init__(self, max_length: int = 1024, pad_token_id: int = 0):
        self.max_length = max_length
        self.pad_token_id = pad_token_id
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate features into batches
        
        Args:
            features: List of feature dictionaries with 'input_ids'
            
        Returns:
            Dictionary with 'input_ids' and 'labels' tensors
        """
        # Extract input_ids from features
        input_ids = [f["input_ids"] for f in features]
        
        # Pad sequences to max_length
        padded_ids = []
        for seq in input_ids:
            if len(seq) > self.max_length:
                # Truncate if too long
                seq = seq[:self.max_length]
            elif len(seq) < self.max_length:
                # Pad if too short
                seq = seq + [self.pad_token_id] * (self.max_length - len(seq))
            padded_ids.append(seq)
        
        # Convert to tensors
        input_tensor = torch.tensor(padded_ids, dtype=torch.long)
        labels_tensor = input_tensor.clone()
        
        # Create attention mask (ignore padding tokens)
        attention_mask = (input_tensor != self.pad_token_id).long()
        
        return {
            "input_ids": input_tensor,
            "labels": labels_tensor,
            "attention_mask": attention_mask
        }

class VariableLenCollator:
    """Variable-length sequence collator with dynamic padding"""
    
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate features with dynamic padding
        
        Args:
            features: List of feature dictionaries with 'input_ids'
            
        Returns:
            Dictionary with padded tensors
        """
        # Find max length in this batch
        max_len = max(len(f["input_ids"]) for f in features)
        
        # Pad all sequences to max length
        padded_ids = []
        for seq in features["input_ids"]:
            if len(seq) < max_len:
                seq = seq + [self.pad_token_id] * (max_len - len(seq))
            padded_ids.append(seq)
        
        # Convert to tensors
        input_tensor = torch.tensor(padded_ids, dtype=torch.long)
        labels_tensor = input_tensor.clone()
        
        # Create attention mask
        attention_mask = (input_tensor != self.pad_token_id).long()
        
        return {
            "input_ids": input_tensor,
            "labels": labels_tensor,
            "attention_mask": attention_mask
        }
