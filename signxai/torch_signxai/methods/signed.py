"""Implementation of SIGN thresholding methods for PyTorch."""
import torch
import numpy as np


def calculate_sign_mu(relevance_map, mu=0.0):
    """Calculate binary sign-based relevance map.
    
    Args:
        relevance_map: Relevance map tensor or numpy array
        mu: Threshold for considering a value positive/negative (default 0.0)
        
    Returns:
        Sign-based relevance map (-1, 0, +1 values)
    """
    if isinstance(relevance_map, torch.Tensor):
        # PyTorch tensor case
        sign_map = torch.zeros_like(relevance_map)
        sign_map[relevance_map > mu] = 1.0
        sign_map[relevance_map < -mu] = -1.0
        return sign_map
    else:
        # Numpy array case
        sign_map = np.zeros_like(relevance_map)
        sign_map[relevance_map > mu] = 1.0
        sign_map[relevance_map < -mu] = -1.0
        return sign_map