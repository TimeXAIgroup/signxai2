"""
Exact TensorFlow iNNvestigate EpsilonRule implementation for PyTorch Zennit.

This hook exactly mirrors the TensorFlow implementation from 
signxai/tf_signxai/methods/innvestigate/analyzer/relevance_based/relevance_rule_base.py
to achieve perfect numerical matching.
"""

import torch
import torch.nn as nn
import numpy as np
from zennit.core import Hook
from typing import Union, Optional


class TFExactEpsilonHook(Hook):
    """
    Hook that exactly replicates TensorFlow iNNvestigate's EpsilonRule implementation.
    
    This follows the exact same mathematical pattern as TF:
    1. Zs = layer_wo_act(ins) 
    2. prepare_div = Zs + (cast(greater_equal(Zs, 0), float) * 2 - 1) * epsilon
    3. tmp = SafeDivide([reversed_outs, prepare_div])
    4. gradient(Zs, ins, output_gradients=tmp)
    5. Multiply([ins, gradient_result])
    """
    
    def __init__(self, epsilon: float = 1e-7):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Store input for backward pass."""
        self.input = input[0] if isinstance(input, tuple) else input
        return output
    
    def backward(self, module: nn.Module, grad_input: tuple, grad_output: tuple) -> tuple:
        """
        Exact TensorFlow EpsilonRule implementation.
        
        TF code equivalent:
        prepare_div = keras_layers.Lambda(lambda x: x + (K.cast(K.greater_equal(x, 0), K.floatx()) * 2 - 1) * self._epsilon)
        tmp = ilayers.SafeDivide()([reversed_outs, prepare_div(Zs)])
        tmp2 = tape.gradient(Zs, ins, output_gradients=tmp)
        ret = keras_layers.Multiply()([ins, tmp2])
        """
        if grad_output[0] is None:
            return grad_input
            
        relevance = grad_output[0]
        
        # Step 1: Compute Zs = layer_wo_act(ins) - forward pass without bias for Conv/Linear
        if isinstance(module, (nn.Conv2d, nn.Conv1d)):
            if isinstance(module, nn.Conv2d):
                # Forward pass to get activations (Zs in TF)
                zs = torch.nn.functional.conv2d(
                    self.input, module.weight, module.bias,  # Include bias like TF
                    module.stride, module.padding, module.dilation, module.groups
                )
            else:  # Conv1d
                zs = torch.nn.functional.conv1d(
                    self.input, module.weight, module.bias,  # Include bias like TF
                    module.stride, module.padding, module.dilation, module.groups
                )
                
        elif isinstance(module, nn.Linear):
            zs = torch.nn.functional.linear(self.input, module.weight, module.bias)  # Include bias like TF
        else:
            return grad_input
        
        # Step 2: Apply TensorFlow's exact epsilon stabilization
        # TF: (K.cast(K.greater_equal(x, 0), K.floatx()) * 2 - 1) * self._epsilon
        tf_sign = (zs >= 0).float() * 2.0 - 1.0  # +1 for >=0, -1 for <0 (TF uses >= for 0)
        prepare_div = zs + tf_sign * self.epsilon
        
        # Step 3: SafeDivide - handle division by zero like TF
        # TF SafeDivide uses a small epsilon to avoid division by zero
        safe_epsilon = 1e-12
        safe_prepare_div = torch.where(
            torch.abs(prepare_div) < safe_epsilon,
            torch.sign(prepare_div) * safe_epsilon,
            prepare_div
        )
        
        # Step 4: Divide relevance by stabilized activations
        tmp = relevance / safe_prepare_div
        
        # Step 5: Compute gradient-like operation
        # In TF: tape.gradient(Zs, ins, output_gradients=tmp)
        # This is equivalent to: tmp * (gradient of Zs w.r.t. input)
        
        if isinstance(module, (nn.Conv2d, nn.Conv1d)):
            if isinstance(module, nn.Conv2d):
                # The gradient of conv2d w.r.t. input is conv_transpose2d with the same weights
                gradient_result = torch.nn.functional.conv_transpose2d(
                    tmp, module.weight, None,  # No bias in gradient
                    module.stride, module.padding,
                    output_padding=0, groups=module.groups, dilation=module.dilation
                )
            else:  # Conv1d
                gradient_result = torch.nn.functional.conv_transpose1d(
                    tmp, module.weight, None,  # No bias in gradient
                    module.stride, module.padding,
                    output_padding=0, groups=module.groups, dilation=module.dilation
                )
                
        elif isinstance(module, nn.Linear):
            # For linear layer: gradient w.r.t. input is weight.T @ output_gradient
            gradient_result = torch.mm(tmp, module.weight)
        
        # Step 6: Final multiply by input (like TF)
        # TF: keras_layers.Multiply()([ins, tmp2])
        final_result = self.input * gradient_result
        
        return (final_result,) + grad_input[1:]


def create_tf_exact_epsilon_composite(epsilon: float = 1e-7):
    """Create a composite using TFExactEpsilonHook that exactly matches TensorFlow."""
    from zennit.core import Composite
    from zennit.types import Convolution, Linear, BatchNorm, Activation, AvgPool
    
    epsilon_hook = TFExactEpsilonHook(epsilon=epsilon)
    
    def module_map(ctx, name, module):
        if isinstance(module, (Convolution, Linear)):
            return epsilon_hook
        elif isinstance(module, (BatchNorm, Activation, AvgPool)):
            return None
        return None
    
    return Composite(module_map=module_map)


def create_tf_exact_w2lrp_epsilon_composite(epsilon: float = 0.1):
    """
    Create a composite for TF-exact W2LRP + Epsilon analysis.
    
    This exactly replicates TensorFlow iNNvestigate's behavior for methods like
    w2lrp_epsilon_0_1 which use:
    - WSquare rule for the first layer (input layer) 
    - Epsilon rule for all other layers
    
    Args:
        epsilon: Epsilon value for stabilization (default: 0.1)
        
    Returns:
        Composite that applies WSquare rule to first layer and Epsilon to others
    """
    from zennit.core import Composite
    from zennit.types import Convolution, Linear
    from zennit.rules import WSquare
    
    # Track if we've applied the first layer rule
    first_layer_applied = [False]
    
    def module_map(ctx, name, module):
        if isinstance(module, (Convolution, Linear)):
            if not first_layer_applied[0]:
                first_layer_applied[0] = True
                print(f" TF-Exact W2LRP+Epsilon: Applying WSquare rule to first layer: {name}")
                # Create separate instance for WSquare rule
                return WSquare()
            else:
                print(f" TF-Exact W2LRP+Epsilon: Applying Epsilon(ε={epsilon}) to layer: {name}")
                # Create separate instance for each layer to avoid sharing state
                return TFExactEpsilonHook(epsilon=epsilon)
        return None
    
    return Composite(module_map=module_map)