"""
Evaluation utilities
"""

from .ppl import evaluate_perplexity
from .compare import compare_models_on_wiki

__all__ = ["evaluate_perplexity", "compare_models_on_wiki"]
