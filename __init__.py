"""
MNIST Digit Classification with Tensor Parallelism
Main package initialization
"""

__version__ = "1.0.0"
__author__ = "Shashank"

# Import utilities first to avoid circular imports
from .utilities import *

# Import training modules
try:
    # If Base_training is a directory with __init__.py
    from .Base_training import *
except ImportError:
    # If Base_training.py is a file, import it directly
    from . import Base_training

from .Training_Tensor_Parallelism import *

# Make key components available at package level
__all__ = [
    # Model components
    'Model',
    'Attention', 
    'MLP',
    'PatchEmbedding',
    
    # Data loading
    'CustomDataset',
    'mnist_transform',
    'get_dataloaders',
    
    # Utility functions
    'setup_device',
    'count_parameters', 
    'save_checkpoint',
    'load_checkpoint',
]