"""
StdxEpsilon rule implementation for Zennit and PyTorch.
This custom rule implements the StdxEpsilonRule from TensorFlow iNNvestigate.
"""
import torch
import numpy as np
from zennit.rules import Epsilon
from zennit.core import Stabilizer


class StdxEpsilon(Epsilon):
    """
    StdxEpsilon rule from the TensorFlow iNNvestigate implementation.
    This rule is similar to Epsilon rule but uses a multiple of the 
    standard deviation of the input as epsilon for stabilization.
    
    Args:
        stdfactor (float, optional): Factor to multiply the standard deviation by.
            Default: 0.25.
        bias (bool, optional): Whether to include bias in the computation.
            Default: True.
    """
    
    def __init__(self, stdfactor=0.25, bias=True):
        """
        Initialize StdxEpsilon rule with the standard deviation factor.
        
        Args:
            stdfactor (float, optional): Factor to multiply the standard deviation by.
                Default: 0.25.
            bias (bool, optional): Whether to include bias in the computation.
                Default: True.
        """
        # Initialize with a default epsilon (will be overridden dynamically)
        super().__init__(epsilon=1e-6, zero_params=[] if bias else ['bias'])
        self.stdfactor = stdfactor
        self.bias = bias
        
    def gradient_mapper(self, input_tensor, output_gradient):
        """
        Custom gradient mapper that calculates epsilon based on input standard deviation.
        
        Args:
            input_tensor (torch.Tensor): Input tensor to the layer.
            output_gradient (torch.Tensor): Gradient from the next layer.
            
        Returns:
            torch.Tensor: Modified gradient based on StdxEpsilon rule.
        """
        # Calculate epsilon based on standard deviation of input
        std_val = torch.std(input_tensor).item()
        epsilon = max(std_val * self.stdfactor, 1e-12)
        
        # Use Zennit's stabilizer to apply epsilon properly
        stabilized_input = Stabilizer.ensure(input_tensor, epsilon=epsilon)
        
        # Apply the gradient computation with the dynamic epsilon
        return output_gradient / stabilized_input
    
    def copy(self):
        """Return a copy of this hook that preserves our custom attributes."""
        return StdxEpsilon(stdfactor=self.stdfactor, bias=self.bias)