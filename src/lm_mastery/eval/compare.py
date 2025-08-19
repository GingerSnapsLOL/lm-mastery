"""
Model comparison utilities
"""

import torch
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

def compare_models_on_wiki(
    models: List[AutoModelForCausalLM],
    tokenizers: List[AutoTokenizer],
    wiki_dataset: Dataset,
    max_length: int = 1024
) -> Dict[str, Any]:
    """
    Compare multiple models on WikiText dataset
    
    Args:
        models: List of models to compare
        tokenizers: List of tokenizers
        wiki_dataset: WikiText dataset
        max_length: Maximum sequence length
        
    Returns:
        Dictionary with comparison results
    """
    
    results = {}
    
    for i, (model, tokenizer) in enumerate(zip(models, tokenizers)):
        model_name = f"model_{i}"
        print(f"Evaluating {model_name}...")
        
        # Simple evaluation - calculate average loss on a few samples
        model.eval()
        total_loss = 0.0
        num_samples = min(10, len(wiki_dataset))
        
        with torch.no_grad():
            for j in range(num_samples):
                sample = wiki_dataset[j]
                input_ids = torch.tensor(sample["input_ids"][:max_length], dtype=torch.long).unsqueeze(0)
                
                if torch.cuda.is_available():
                    input_ids = input_ids.cuda()
                    model = model.cuda()
                
                outputs = model(input_ids, labels=input_ids)
                total_loss += outputs.loss.item()
        
        avg_loss = total_loss / num_samples
        results[model_name] = {
            "avg_loss": avg_loss,
            "perplexity": torch.exp(torch.tensor(avg_loss)).item()
        }
    
    return results
