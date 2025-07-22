#!/usr/bin/env python3
"""TF-exact VarGrad x Input implementation to match TensorFlow behavior exactly."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union
from .tf_exact_vargrad_hook import TFExactVarGradAnalyzer


class TFExactVarGradXInputAnalyzer:
    """TF-exact VarGrad x Input analyzer that matches TensorFlow implementation exactly.
    
    TensorFlow implementation: vargrad_x_input = vargrad * input
    """
    
    def __init__(self, model: nn.Module, noise_scale: float = 0.2, augment_by_n: int = 50):
        """Initialize TF-exact VarGrad x Input analyzer.
        
        Args:
            model: PyTorch model
            noise_scale: Standard deviation of noise to add (TF default: 0.2)
            augment_by_n: Number of noisy samples to generate (TF default: 50)
        """
        self.vargrad_analyzer = TFExactVarGradAnalyzer(
            model=model,
            noise_scale=noise_scale,
            augment_by_n=augment_by_n
        )
    
    def analyze(self, input_tensor: torch.Tensor, target_class: Optional[Union[int, torch.Tensor]] = None, **kwargs) -> np.ndarray:
        """Analyze input using TF-exact VarGrad x Input algorithm.
        
        Args:
            input_tensor: Input tensor to analyze
            target_class: Target class index
            **kwargs: Additional arguments
            
        Returns:
            VarGrad x Input attribution as numpy array
        """
        # Get VarGrad attribution
        vargrad_result = self.vargrad_analyzer.analyze(input_tensor, target_class, **kwargs)
        
        # Convert input to numpy if needed
        if isinstance(input_tensor, torch.Tensor):
            input_np = input_tensor.detach().cpu().numpy()
        else:
            input_np = input_tensor
        
        # Handle batch dimension
        if input_np.ndim == 4:  # (batch, channels, height, width)
            input_np = input_np[0]  # Remove batch dimension
        elif input_np.ndim == 3:  # (channels, height, width) - already correct
            pass
        
        # Multiply VarGrad result by input exactly as TensorFlow does: v * x
        result = vargrad_result * input_np
        
        return result


def create_tf_exact_vargrad_x_input_analyzer(model: nn.Module, **kwargs):
    """Create TF-exact VarGrad x Input analyzer with TensorFlow-compatible parameters.
    
    Args:
        model: PyTorch model
        **kwargs: Additional arguments
        
    Returns:
        TFExactVarGradXInputAnalyzer instance
    """
    # Use TensorFlow defaults
    noise_scale = kwargs.get('noise_scale', 0.2)  # TF default
    augment_by_n = kwargs.get('augment_by_n', 50)  # TF default
    
    return TFExactVarGradXInputAnalyzer(
        model=model,
        noise_scale=noise_scale,
        augment_by_n=augment_by_n
    )