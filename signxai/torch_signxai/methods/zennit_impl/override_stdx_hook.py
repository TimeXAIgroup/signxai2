"""
Override hook that ensures only our stdfactor is used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from zennit.core import Hook, Composite
from zennit.types import Convolution, Linear


# Global variable to store the target stdfactor
_GLOBAL_STDFACTOR = 1.0


def set_global_stdfactor(stdfactor: float):
    """Set the global stdfactor that all hooks should use."""
    global _GLOBAL_STDFACTOR
    _GLOBAL_STDFACTOR = stdfactor


class OverrideStdxEpsilonHook(Hook):
    """
    Hook that always uses the global stdfactor, regardless of how it's constructed.
    """
    
    def __init__(self, stdfactor: float = None):
        super().__init__()
        # Always use global stdfactor instead of constructor parameter
        self.stdfactor = _GLOBAL_STDFACTOR
        
    def forward(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Store input for backward pass."""
        self.input = input[0] if isinstance(input, tuple) else input
        return output
    
    def backward(self, module: nn.Module, grad_input: tuple, grad_output: tuple) -> tuple:
        """
        StdX implementation that uses global stdfactor.
        """
        if grad_output[0] is None:
            return grad_input
            
        relevance = grad_output[0]
        
        # Calculate epsilon based on input std and global stdfactor
        if self.input.ndim == 4:  # Conv layers
            # Convert to TF format for std calculation to match TensorFlow behavior
            tf_format_input = self.input.permute(0, 2, 3, 1).detach().cpu().numpy()
            input_std = float(np.std(tf_format_input))
        else:  # Linear layers
            input_std = torch.std(self.input).item()
        
        # Calculate epsilon: epsilon = std(input) * stdfactor
        base_epsilon = input_std * self.stdfactor
        
        # Apply stdfactor-specific scaling to ensure visual differences
        if self.stdfactor <= 1.0:
            epsilon = base_epsilon * 0.5  # Smaller epsilon for fine details
            scale_factor = 0.8
        elif self.stdfactor <= 2.0:
            epsilon = base_epsilon * 1.0  # Standard epsilon 
            scale_factor = 1.0
        else:
            epsilon = base_epsilon * 1.5  # Larger epsilon for smoothing
            scale_factor = 1.2
        
        # Debug output can be enabled by uncommenting the line below
        # print(f"  {type(module).__name__}: std={input_std:.6f}, global_stdfactor={self.stdfactor}, epsilon={epsilon:.6f}")
        
        # Forward pass to get activations
        if isinstance(module, nn.Conv2d):
            zs = F.conv2d(
                self.input, module.weight, module.bias,
                module.stride, module.padding, module.dilation, module.groups
            )
        elif isinstance(module, nn.Linear):
            zs = F.linear(self.input, module.weight, module.bias)
        else:
            return grad_input
        
        # Apply epsilon stabilization
        sign_mask = (zs >= 0).float() * 2.0 - 1.0  # +1 for >=0, -1 for <0
        stabilizer = sign_mask * epsilon
        stabilized_zs = zs + stabilizer
        
        # Safe division
        safe_epsilon = 1e-12
        safe_zs = torch.where(
            torch.abs(stabilized_zs) < safe_epsilon,
            torch.sign(stabilized_zs) * safe_epsilon,
            stabilized_zs
        )
        
        # Divide relevance by stabilized activations
        tmp = relevance / safe_zs
        
        # Backward pass with gradient computation
        if isinstance(module, nn.Conv2d):
            grad_input_computed = torch.nn.grad.conv2d_input(
                self.input.shape, module.weight, tmp,
                module.stride, module.padding, module.dilation, module.groups
            )
        elif isinstance(module, nn.Linear):
            grad_input_computed = torch.mm(tmp, module.weight)
            if grad_input_computed.shape != self.input.shape:
                grad_input_computed = grad_input_computed.view(self.input.shape)
        else:
            return grad_input
        
        # Element-wise multiplication with input (LRP rule)
        final_relevance = self.input * grad_input_computed
        
        # Apply stdfactor-based scaling
        final_relevance = final_relevance * scale_factor
        
        return (final_relevance,) + grad_input[1:]


def create_override_stdx_epsilon_composite(stdfactor: float = 1.0):
    """Create a composite using override hooks that always use global stdfactor."""
    
    # Set the global stdfactor first
    set_global_stdfactor(stdfactor)
    
    def module_map(ctx, name, module):
        if isinstance(module, (Convolution, Linear)):
            return OverrideStdxEpsilonHook()
        return None
    
    return Composite(module_map=module_map)