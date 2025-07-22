#!/usr/bin/env python3
"""TF-exact VarGrad x Input x Sign implementation to match TensorFlow behavior exactly."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union
from .tf_exact_vargrad_hook import TFExactVarGradAnalyzer


class TFExactVarGradXInputXSignAnalyzer:
    """TF-exact VarGrad x Input x Sign analyzer that matches TensorFlow implementation exactly.
    
    TensorFlow implementation: vargrad_x_input_x_sign = vargrad * input * sign(input)
    where sign(input) = np.nan_to_num(input / np.abs(input), nan=1.0)
    """
    
    def __init__(self, model: nn.Module, noise_scale: float = 0.2, augment_by_n: int = 50):
        """Initialize TF-exact VarGrad x Input x Sign analyzer.
        
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
        """Analyze input using TF-exact VarGrad x Input x Sign algorithm.
        
        Args:
            input_tensor: Input tensor to analyze
            target_class: Target class index
            **kwargs: Additional arguments
            
        Returns:
            VarGrad x Input x Sign attribution as numpy array
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
        
        # Calculate sign exactly as TensorFlow does: np.nan_to_num(x / np.abs(x), nan=1.0)
        # This handles division by zero by setting result to 1.0 where input is 0
        input_abs = np.abs(input_np)
        with np.errstate(divide='ignore', invalid='ignore'):
            sign = input_np / input_abs
        sign = np.nan_to_num(sign, nan=1.0, posinf=1.0, neginf=-1.0)
        
        # Apply TensorFlow formula: v * x * s
        # VarGrad result is already channel-summed (224, 224)
        # Input and sign are (3, 224, 224), so we need to apply the formula channel-wise then sum
        result_channels = []
        for c in range(input_np.shape[0]):  # For each channel
            channel_result = vargrad_result * input_np[c] * sign[c]
            result_channels.append(channel_result)
        
        # Sum across channels to get final (224, 224) result
        result = np.sum(result_channels, axis=0)
        
        return result


def create_tf_exact_vargrad_x_input_x_sign_analyzer(model: nn.Module, **kwargs):
    """Create TF-exact VarGrad x Input x Sign analyzer with TensorFlow-compatible parameters.
    
    Args:
        model: PyTorch model
        **kwargs: Additional arguments
        
    Returns:
        TFExactVarGradXInputXSignAnalyzer instance
    """
    # Use TensorFlow defaults
    noise_scale = kwargs.get('noise_scale', 0.2)  # TF default
    augment_by_n = kwargs.get('augment_by_n', 50)  # TF default
    
    return TFExactVarGradXInputXSignAnalyzer(
        model=model,
        noise_scale=noise_scale,
        augment_by_n=augment_by_n
    )