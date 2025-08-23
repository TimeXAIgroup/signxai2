"""Implementation of SIGN thresholding methods for PyTorch."""
import torch
import numpy as np


def calculate_sign_mu(relevance_map, mu=0.0, vlow=-1, vhigh=1):
    """Calculate binary sign-based relevance map to match TensorFlow behavior.
    
    Args:
        relevance_map: Relevance map tensor or numpy array
        mu: Threshold for considering a value positive/negative (default 0.0)
        vlow: Value for elements below threshold (default -1)
        vhigh: Value for elements at or above threshold (default 1)
        
    Returns:
        Sign-based relevance map with TensorFlow-compatible behavior
    """
    if isinstance(relevance_map, torch.Tensor):
        # PyTorch tensor case - match TensorFlow behavior exactly
        sign_map = torch.full_like(relevance_map, float(vlow))
        sign_map[relevance_map >= mu] = float(vhigh)
        return sign_map
    else:
        # Numpy array case - match TensorFlow behavior exactly
        sign_map = np.full_like(relevance_map, vlow)
        sign_map[relevance_map >= mu] = vhigh
        return sign_map