"""
Tensor Parallel training module
Contains distributed training functionality
"""

from .TP_training import *

# If TP_training.py has specific functions/classes, list them here
__all__ = [
    # Add specific exports from TP_training.py
    # For example: 'train_model_tp', 'setup_distributed', etc.
    'train_epoch',
    'validate',
    'train_model',
    'main'
]