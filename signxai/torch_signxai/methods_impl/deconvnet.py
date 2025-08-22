"""PyTorch implementation of DeconvNet."""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Union, Optional, Callable, Dict, Any


class DeconvNetReLU(torch.autograd.Function):
    """DeconvNet ReLU activation.
    
    This modified ReLU passes the gradient if the gradient from the next layer is positive,
    regardless of the input value.
    """
    
    @staticmethod
    def forward(ctx, input_tensor):
        ctx.save_for_backward(input_tensor)
        return input_tensor.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        # For DeconvNet, we only consider the gradient, not the input value
        # If gradient is positive, pass it, otherwise zero
        grad_input = grad_output.clone()
        grad_input[grad_output < 0] = 0
        return grad_input


class DeconvNetReLUModule(nn.Module):
    """Module wrapper for the DeconvNetReLU function."""
    
    def forward(self, x):
        return DeconvNetReLU.apply(x)


def replace_relu_with_deconvnet_relu(model):
    """Replace all ReLU activations with DeconvNetReLU.
    
    Args:
        model: PyTorch model
        
    Returns:
        Modified model with DeconvNet ReLU activations
    """
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            setattr(model, name, DeconvNetReLUModule())
        else:
            replace_relu_with_deconvnet_relu(module)
    return model


def build_deconvnet_model(model):
    """Build a DeconvNet model by replacing ReLU activations.
    
    Args:
        model: PyTorch model
        
    Returns:
        DeconvNet model for backpropagation
    """
    # Create a copy of the model to avoid modifying the original
    deconvnet_model = type(model)()
    deconvnet_model.load_state_dict(model.state_dict())
    deconvnet_model.eval()
    
    # Replace ReLU with DeconvNet ReLU
    replace_relu_with_deconvnet_relu(deconvnet_model)
    
    return deconvnet_model


def deconvnet(model, input_tensor, target_class=None):
    """Generate DeconvNet attribution map.
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor (requires_grad=True)
        target_class: Target class index (None for argmax)
        
    Returns:
        Gradient attribution map
    """
    # Ensure input has gradient
    input_tensor = input_tensor.requires_grad_(True)
    
    # Forward pass
    model.zero_grad()
    output = model(input_tensor)
    
    # Select target class
    if target_class is None:
        target_class = output.argmax(dim=1)
    
    # Create one-hot encoding
    if output.dim() == 2:  # Batch output
        one_hot = torch.zeros_like(output)
        one_hot.scatter_(1, target_class.unsqueeze(1) if isinstance(target_class, torch.Tensor) else target_class, 1.0)
    else:  # Single output
        one_hot = torch.zeros_like(output)
        one_hot[target_class] = 1.0
    
    # Backward pass
    output.backward(gradient=one_hot)
    
    # Get gradients
    return input_tensor.grad.detach()