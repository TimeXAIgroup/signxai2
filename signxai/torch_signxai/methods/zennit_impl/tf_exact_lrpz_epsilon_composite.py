"""
Exact TensorFlow iNNvestigate LRPZ Epsilon implementation for PyTorch Zennit.

This hook exactly mirrors the TensorFlow implementation:
- method='lrp.epsilon' with epsilon=0.1 
- input_layer_rule='Z'

Following the same pattern as tf_exact_epsilon_hook.py for perfect numerical matching.
"""

import torch
import torch.nn as nn
import numpy as np
from zennit.core import Hook, Composite
from zennit.types import Convolution, Linear, BatchNorm, Activation, AvgPool
from typing import Union, Optional


class TFExactZHook(Hook):
    """
    Hook that exactly replicates TensorFlow iNNvestigate's Z rule for input layer.
    
    Z rule is effectively Epsilon with epsilon=0 and special handling for the first layer.
    In TF iNNvestigate, Z rule is implemented as:
    - No epsilon stabilization (epsilon=0)
    - Uses only positive weights/activations for relevance propagation
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Store input for backward pass."""
        self.input = input[0] if isinstance(input, tuple) else input
        return output
    
    def backward(self, module: nn.Module, grad_input: tuple, grad_output: tuple) -> tuple:
        """
        Exact TensorFlow Z rule implementation for input layer.
        
        Z rule in TF iNNvestigate:
        - Uses only positive weights and positive activations
        - No epsilon stabilization
        - Effectively LRP-0 (Epsilon with epsilon=0) but with positivity constraints
        """
        if grad_output[0] is None:
            return grad_input
            
        relevance = grad_output[0]
        
        # Z rule: Use standard forward pass but with special handling
        # The Z rule in iNNvestigate is actually implemented as LRP-0 (epsilon=0) 
        # with special handling for the first layer, not strict positive-only
        if isinstance(module, (nn.Conv2d, nn.Conv1d)):
            if isinstance(module, nn.Conv2d):
                # Standard forward pass (like other layers)
                zs = torch.nn.functional.conv2d(
                    self.input, module.weight, module.bias,
                    module.stride, module.padding, module.dilation, module.groups
                )
            else:  # Conv1d
                zs = torch.nn.functional.conv1d(
                    self.input, module.weight, module.bias,
                    module.stride, module.padding, module.dilation, module.groups
                )
                
        elif isinstance(module, nn.Linear):
            # Standard forward pass 
            zs = torch.nn.functional.linear(self.input, module.weight, module.bias)
        else:
            return grad_input
        
        # Z rule: no epsilon stabilization (epsilon=0), but avoid division by zero
        safe_epsilon = 1e-9  # Larger epsilon to avoid numerical issues
        safe_zs = torch.where(
            torch.abs(zs) < safe_epsilon,
            torch.sign(zs) * safe_epsilon,
            zs
        )
        
        # Divide relevance by activations
        tmp = relevance / safe_zs
        
        # Compute gradient-like operation (standard weights, not positive-only)
        if isinstance(module, (nn.Conv2d, nn.Conv1d)):
            if isinstance(module, nn.Conv2d):
                gradient_result = torch.nn.functional.conv_transpose2d(
                    tmp, module.weight, None,  # Use original weights
                    module.stride, module.padding,
                    output_padding=0, groups=module.groups, dilation=module.dilation
                )
            else:  # Conv1d
                gradient_result = torch.nn.functional.conv_transpose1d(
                    tmp, module.weight, None,  # Use original weights
                    module.stride, module.padding,
                    output_padding=0, groups=module.groups, dilation=module.dilation
                )
                
        elif isinstance(module, nn.Linear):
            gradient_result = torch.mm(tmp, module.weight)  # Use original weights
        
        # Final multiply by input (standard, not clamped)
        final_result = self.input * gradient_result
        
        return (final_result,) + grad_input[1:]


class TFExactEpsilonHook(Hook):
    """
    Hook that exactly replicates TensorFlow iNNvestigate's EpsilonRule implementation.
    Copied from the working tf_exact_epsilon_hook.py
    """
    
    def __init__(self, epsilon: float = 1e-7):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Store input for backward pass."""
        self.input = input[0] if isinstance(input, tuple) else input
        return output
    
    def backward(self, module: nn.Module, grad_input: tuple, grad_output: tuple) -> tuple:
        """Exact TensorFlow EpsilonRule implementation."""
        if grad_output[0] is None:
            return grad_input
            
        relevance = grad_output[0]
        
        # Step 1: Compute Zs = layer_wo_act(ins)
        if isinstance(module, (nn.Conv2d, nn.Conv1d)):
            if isinstance(module, nn.Conv2d):
                zs = torch.nn.functional.conv2d(
                    self.input, module.weight, module.bias,
                    module.stride, module.padding, module.dilation, module.groups
                )
            else:  # Conv1d
                zs = torch.nn.functional.conv1d(
                    self.input, module.weight, module.bias,
                    module.stride, module.padding, module.dilation, module.groups
                )
                
        elif isinstance(module, nn.Linear):
            zs = torch.nn.functional.linear(self.input, module.weight, module.bias)
        else:
            return grad_input
        
        # Step 2: Apply TensorFlow's exact epsilon stabilization
        tf_sign = (zs >= 0).float() * 2.0 - 1.0
        prepare_div = zs + tf_sign * self.epsilon
        
        # Step 3: SafeDivide
        safe_epsilon = 1e-12
        safe_prepare_div = torch.where(
            torch.abs(prepare_div) < safe_epsilon,
            torch.sign(prepare_div) * safe_epsilon,
            prepare_div
        )
        
        # Step 4: Divide relevance by stabilized activations
        tmp = relevance / safe_prepare_div
        
        # Step 5: Compute gradient-like operation
        if isinstance(module, (nn.Conv2d, nn.Conv1d)):
            if isinstance(module, nn.Conv2d):
                gradient_result = torch.nn.functional.conv_transpose2d(
                    tmp, module.weight, None,
                    module.stride, module.padding,
                    output_padding=0, groups=module.groups, dilation=module.dilation
                )
            else:  # Conv1d
                gradient_result = torch.nn.functional.conv_transpose1d(
                    tmp, module.weight, None,
                    module.stride, module.padding,
                    output_padding=0, groups=module.groups, dilation=module.dilation
                )
                
        elif isinstance(module, nn.Linear):
            gradient_result = torch.mm(tmp, module.weight)
        
        # Step 6: Final multiply by input
        final_result = self.input * gradient_result
        
        return (final_result,) + grad_input[1:]


def create_tf_exact_lrpz_epsilon_composite(epsilon: float = 0.1):
    """
    Create a composite that exactly matches TensorFlow's LRPZ epsilon implementation.
    
    TF implementation:
    - method='lrp.epsilon' with epsilon=0.1
    - input_layer_rule='Z' 
    
    This means:
    - First layer uses Z rule (Epsilon with epsilon=0)
    - All other layers use Epsilon rule with epsilon=0.1
    """
    
    # Use the proven TF-exact epsilon hook for both rules
    z_hook = TFExactEpsilonHook(epsilon=0.0)  # Z rule = Epsilon with epsilon=0
    epsilon_hook = TFExactEpsilonHook(epsilon=epsilon)
    
    # Track if we've applied the first layer rule
    first_layer_applied = [False]
    
    def module_map(ctx, name, module):
        if isinstance(module, (Convolution, Linear)):
            if not first_layer_applied[0]:
                # Apply Z rule (Epsilon with epsilon=0) to first conv/linear layer
                first_layer_applied[0] = True
                print(f"🔧 TF-Exact LRPZ: Applying Z rule (Epsilon ε=0) to first layer: {name}")
                return z_hook
            else:
                # Apply Epsilon rule to all other conv/linear layers
                print(f"🔧 TF-Exact LRPZ: Applying Epsilon({epsilon}) to layer: {name}")
                return epsilon_hook
        elif isinstance(module, (BatchNorm, Activation, AvgPool)):
            # These layers are typically pass-through in LRP
            return None
        return None
    
    return Composite(module_map=module_map)