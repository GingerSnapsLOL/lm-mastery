"""
Logging utilities for training and evaluation
"""

import logging
import sys
from typing import Optional

def setup_logging(level: int = logging.INFO, 
                  log_file: Optional[str] = None,
                  format_string: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration"""
    
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create logger
    logger = logging.getLogger("lm_mastery")
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str = "lm_mastery") -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name)
