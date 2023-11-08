# This file defines general purpose functions

# Import necessary libraries
import numpy as np
import torch
import random
import os

# Define a function to set random seeds for reproducibility
def set_seed(seed: int = 42):
    # Set random seed for Python's random module
    random.seed(seed)
    
    # Set environment variable for Python's hash seed
    os.environ['PYHTONHASHSEED'] = str(seed)
    
    # Set random seed for NumPy
    np.random.seed(seed)
    
    # Set random seed for PyTorch on CPU
    torch.manual_seed(seed)
    
    # Set random seed for PyTorch on GPU (if available)
    torch.cuda.manual_seed(seed)
    
    # Ensure deterministic behavior for CuDNN (CuDNN is a GPU-accelerated library)
    torch.backends.cudnn.deterministic = True