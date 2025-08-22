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
        Matches TensorFlow's StdxEpsilonRule implementation exactly.
        
        Args:
            input_tensor (torch.Tensor): Input tensor to the layer.
            output_gradient (torch.Tensor): Gradient from the next layer.
            
        Returns:
            torch.Tensor: Modified gradient based on StdxEpsilon rule.
        """
        # Calculate epsilon based on standard deviation of THIS layer's input (exact TF match)
        # Use the full tensor std, not just a single value
        std_val = torch.std(input_tensor)
        epsilon = std_val * self.stdfactor
        
        # Ensure epsilon is a tensor with the same device as input
        if not isinstance(epsilon, torch.Tensor):
            epsilon = torch.tensor(epsilon, device=input_tensor.device, dtype=input_tensor.dtype)
        
        # Apply stabilization: input + sign(input) * epsilon
        # This matches TensorFlow's approach more precisely
        stabilized_input = input_tensor + torch.sign(input_tensor) * epsilon
        
        # Prevent division by zero with a small fallback
        stabilized_input = torch.where(torch.abs(stabilized_input) < 1e-12, 
                                     torch.sign(stabilized_input) * 1e-12, 
                                     stabilized_input)
        
        # Apply the gradient computation with the stabilized input
        return output_gradient / stabilized_input
    
    def copy(self):
        """Return a copy of this hook that preserves our custom attributes."""
        return StdxEpsilon(stdfactor=self.stdfactor, bias=self.bias)