"""
TensorFlow-exact implementation of LRP SIGN Epsilon StdX with mu parameter for PyTorch.

This module implements PyTorch hooks that exactly replicate TensorFlow iNNvestigate's
lrpsign_epsilon_std_x_mu methods behavior.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Callable, List, Union
from zennit.core import Hook, Composite
from zennit.attribution import Gradient
import zennit.layer as zlayer


class TFExactLRPSignEpsilonStdXMuHook(Hook):
    """
    Hook that exactly replicates TensorFlow iNNvestigate's LRP SIGN Epsilon StdX with mu implementation.
    
    This implements the StdxEpsilonRule with SIGNmu input layer rule exactly as done in TensorFlow:
    1. Calculate eps = std(input) * stdfactor per layer
    2. Apply sign-aware stabilization: zs + ((zs >= 0) * 2 - 1) * eps  
    3. Compute relevance using gradient-like operation
    4. Apply SIGNmu rule at input layer with mu parameter
    """
    
    def __init__(self, stdfactor: float = 0.25, mu: float = 0.5, is_input_layer: bool = False):
        super().__init__()
        self.stdfactor = stdfactor
        self.mu = mu
        self.is_input_layer = is_input_layer
    
    def backward(self, module: nn.Module, grad_input: tuple, grad_output: tuple) -> tuple:
        """
        Implements TensorFlow iNNvestigate's StdxEpsilonRule with SIGNmu backward pass.
        """
        # Get the stored input and output from forward hook
        if not hasattr(module, 'input') or not hasattr(module, 'output'):
            return grad_input
            
        input_tensor = module.input
        output_tensor = module.output
        grad_out = grad_output[0]
        
        if input_tensor is None or output_tensor is None or grad_out is None:
            return grad_input
            
        # Convert to TF format (B,H,W,C) for consistent std calculation with TensorFlow
        if len(input_tensor.shape) == 4:  # Conv layer (B,C,H,W) -> (B,H,W,C)
            input_tf_format = input_tensor.permute(0, 2, 3, 1)
            # Use exactly the same std calculation as TensorFlow
            eps = torch.std(input_tf_format, unbiased=False).item() * self.stdfactor
        else:  # Linear layer - use as is
            eps = torch.std(input_tensor, unbiased=False).item() * self.stdfactor
            
        # Compute raw pre-activation output (Zs in TF)
        if isinstance(module, nn.Conv2d):
            zs = nn.functional.conv2d(
                input_tensor, module.weight, module.bias,
                module.stride, module.padding, module.dilation, module.groups
            )
        elif isinstance(module, nn.Linear):
            zs = nn.functional.linear(input_tensor, module.weight, module.bias)
        else:
            zs = output_tensor
            
        # Apply TF-exact sign-aware stabilization
        # TensorFlow: (K.greater_equal(x, 0) * 2 - 1) * eps
        sign_mask = (zs >= 0).float() * 2 - 1  # +1 for x>=0, -1 for x<0
        stabilized_zs = zs + sign_mask * eps
        
        # TF-exact SafeDivide implementation
        stabilized_zs = torch.where(
            torch.abs(stabilized_zs) < 1e-12,
            torch.sign(stabilized_zs) * 1e-12,
            stabilized_zs
        )
        
        # Compute relevance ratio
        relevance_ratio = grad_out / stabilized_zs
        
        # Gradient-like operation (replicate TensorFlow's tape.gradient)
        if isinstance(module, nn.Conv2d):
            # For conv layers, use conv_transpose (equivalent to TF gradient)
            grad_input_computed = nn.functional.conv_transpose2d(
                relevance_ratio, module.weight, None,
                module.stride, module.padding, 0, module.groups, module.dilation
            )
        elif isinstance(module, nn.Linear):
            # For linear layers, use matrix multiplication
            if relevance_ratio.dim() == 2 and module.weight.dim() == 2:
                grad_input_computed = torch.mm(relevance_ratio, module.weight)
            else:
                grad_input_computed = nn.functional.linear(relevance_ratio, module.weight.t())
        else:
            # For other layers, pass through
            grad_input_computed = relevance_ratio
            
        # Final relevance: input * gradient (TF: Multiply([ins, tmp2]))
        if self.is_input_layer:
            # Apply SIGNmu rule for input layer
            # TF SIGNmu rule: np.nan_to_num(ins / np.abs(ins), nan=1.0) + mu * np.ones_like(ins)
            signs = torch.where(
                torch.abs(input_tensor) < 1e-12,
                torch.ones_like(input_tensor),
                input_tensor / torch.abs(input_tensor)
            )
            # SIGNmu = SIGN + mu * ones
            signmu = signs + self.mu * torch.ones_like(input_tensor)
            relevance = signmu * grad_input_computed
        else:
            relevance = input_tensor * grad_input_computed
            
        return (relevance,) + grad_input[1:]
    
    def forward(self, module: nn.Module, input: tuple, output: tuple) -> tuple:
        """Store input and output for backward pass."""
        module.input = input[0]
        module.output = output[0] if isinstance(output, tuple) else output
        return output


class TFExactLRPSignEpsilonStdXMuComposite:
    """
    Composite that applies TF-exact LRP SIGN Epsilon StdX with mu rules to all layers.
    """
    
    def __init__(self, stdfactor: float = 0.25, mu: float = 0.5):
        self.stdfactor = stdfactor
        self.mu = mu
    
    def context(self, model):
        """Apply hooks with input layer detection."""
        # Find the first layer that should get SIGNmu rule
        first_meaningful_layer = None
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                first_meaningful_layer = module
                break
                
        # Apply hooks
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                is_input_layer = (module == first_meaningful_layer)
                hook = TFExactLRPSignEpsilonStdXMuHook(
                    stdfactor=self.stdfactor,
                    mu=self.mu,
                    is_input_layer=is_input_layer
                )
                handle = module.register_full_backward_hook(hook.backward)
                forward_handle = module.register_forward_hook(hook.forward)
                handles.extend([handle, forward_handle])
                
        return _CompositeContext(model, handles)


class _CompositeContext:
    """Context manager for hook cleanup."""
    
    def __init__(self, model, handles):
        self.model = model
        self.handles = handles
        
    def __enter__(self):
        return self.model
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.handles:
            handle.remove()


def create_tf_exact_lrpsign_epsilon_std_x_mu_composite(stdfactor: float = 0.25, mu: float = 0.5) -> TFExactLRPSignEpsilonStdXMuComposite:
    """
    Create a composite for TF-exact LRP SIGN Epsilon StdX with mu analysis.
    
    Args:
        stdfactor: Standard deviation factor for epsilon calculation
        mu: Mu parameter for SIGNmu rule
        
    Returns:
        Composite that applies TF-exact LRP SIGN Epsilon StdX with mu rules
    """
    return TFExactLRPSignEpsilonStdXMuComposite(stdfactor=stdfactor, mu=mu)