"""
TensorFlow-exact implementation of LRP SIGN Epsilon rule with epsilon=1.0 for PyTorch.

This module implements PyTorch hooks that exactly replicate TensorFlow iNNvestigate's
lrpsign_epsilon_1 method behavior with enhanced precision.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Callable, List, Union
from zennit.core import Hook, Composite


class TFExactLRPSignEpsilon1Hook(Hook):
    """
    Hook that exactly replicates TensorFlow iNNvestigate's LRP SIGN Epsilon implementation with epsilon=1.0.
    
    This implements the Epsilon rule with SIGN input layer rule exactly as done in TensorFlow:
    1. Apply standard epsilon stabilization for hidden layers (epsilon=1.0)
    2. Apply SIGN rule at input layer
    3. Match TensorFlow's exact numerical operations with enhanced precision
    """
    
    def __init__(self, epsilon: float = 1.0, is_input_layer: bool = False):
        super().__init__()
        self.epsilon = epsilon
        self.is_input_layer = is_input_layer
    
    def backward(self, module: nn.Module, grad_input: tuple, grad_output: tuple) -> tuple:
        """Backward hook implementing TF-exact LRP SIGN Epsilon rule."""
        
        if len(grad_input) == 0 or grad_input[0] is None:
            return grad_input
        
        # Use double precision for enhanced numerical accuracy
        original_dtype = grad_input[0].dtype
        
        with torch.no_grad():
            # Get module input (stored during forward pass)
            input_tensor = getattr(module, '_lrp_input', None)
            if input_tensor is None:
                return grad_input
            
            # Convert to double precision for calculations
            input_tensor = input_tensor.to(torch.float64)
            grad_input_computed = grad_input[0].to(torch.float64)
            
            if self.is_input_layer:
                # Apply SIGN rule for input layer (TensorFlow exact implementation)
                # SIGN rule: R = sign(x) * grad if |x| > threshold, else grad
                # Use very small threshold for numerical stability
                threshold = 1e-15
                signs = torch.where(
                    torch.abs(input_tensor) < threshold,
                    torch.ones_like(input_tensor),
                    input_tensor / torch.abs(input_tensor)
                )
                relevance = signs * grad_input_computed
            else:
                # Apply epsilon rule for hidden layers
                # Get weights if available
                weight = getattr(module, 'weight', None)
                bias = getattr(module, 'bias', None)
                
                if weight is not None:
                    weight = weight.to(torch.float64)
                    
                    # Stabilize the denominator with epsilon (TensorFlow approach)
                    # Forward pass for denominator calculation
                    if isinstance(module, nn.Linear):
                        z = torch.nn.functional.linear(input_tensor, weight, bias)
                    elif isinstance(module, nn.Conv2d):
                        z = torch.nn.functional.conv2d(
                            input_tensor, weight, bias,
                            module.stride, module.padding, module.dilation, module.groups
                        )
                    else:
                        # Fallback: use input gradient directly
                        relevance = grad_input_computed
                        return (relevance.to(original_dtype),) + grad_input[1:]
                    
                    # Add epsilon stabilization
                    z_eps = z + self.epsilon * torch.sign(z)
                    z_eps = torch.where(torch.abs(z_eps) < 1e-15, 
                                       torch.sign(z_eps) * 1e-15, z_eps)
                    
                    # Compute relevance using stabilized denominator
                    s = grad_input_computed / z_eps
                    
                    # Backward pass to compute input relevance
                    if isinstance(module, nn.Linear):
                        relevance = torch.nn.functional.linear(s.transpose(-1, -2), weight.transpose(-1, -2)).transpose(-1, -2)
                        if len(input_tensor.shape) != len(relevance.shape):
                            relevance = relevance.squeeze(0)
                    elif isinstance(module, nn.Conv2d):
                        relevance = torch.nn.functional.conv_transpose2d(
                            s, weight, None,
                            module.stride, module.padding, 
                            output_padding=0, groups=module.groups, dilation=module.dilation
                        )
                    else:
                        relevance = grad_input_computed
                    
                    # Element-wise multiplication with input
                    relevance = relevance * input_tensor
                else:
                    # No weights available - use gradient directly
                    relevance = grad_input_computed
            
            # Convert back to original precision
            relevance = relevance.to(original_dtype)
            
        return (relevance,) + grad_input[1:]
    
    def forward(self, module: nn.Module, input: tuple, output: Any) -> Any:
        """Forward hook to store input for backward pass."""
        if len(input) > 0:
            # Store input for backward pass
            setattr(module, '_lrp_input', input[0].detach())
        return output


def create_tf_exact_lrpsign_epsilon_1_composite(model: nn.Module) -> Composite:
    """
    Create a composite with TF-exact LRP SIGN Epsilon hooks for epsilon=1.0.
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        Composite with appropriate hooks for each layer
    """
    
    # Find all layers in the model
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            layers.append((name, module))
    
    # Create hooks for each layer
    hook_map = []
    
    for i, (name, module) in enumerate(layers):
        # First layer (input layer) gets SIGN rule
        is_input_layer = (i == 0)
        
        # Create hook with epsilon=1.0
        hook = TFExactLRPSignEpsilon1Hook(
            epsilon=1.0,
            is_input_layer=is_input_layer
        )
        
        hook_map.append((module, hook))
    
    return Composite(hook_map=hook_map)