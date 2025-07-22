"""
Exact TensorFlow iNNvestigate StdxEpsilonRule implementation for PyTorch Zennit.

This hook exactly mirrors the TensorFlow StdxEpsilonRule implementation to 
achieve perfect numerical matching for std_x epsilon methods.
"""

import torch
import torch.nn as nn
import numpy as np
from zennit.core import Hook
from typing import Union, Optional


class TFExactStdxEpsilonHook(Hook):
    """
    Hook that exactly replicates TensorFlow iNNvestigate's StdxEpsilonRule implementation.
    
    Key TensorFlow behavior:
    1. eps = np.std(ins) * self._stdfactor  (computed per layer from layer input)
    2. prepare_div = Zs + (cast(greater_equal(Zs, 0), float) * 2 - 1) * eps
    3. Same gradient computation as EpsilonRule
    
    Default stdfactor in TF: 0.25
    """
    
    def __init__(self, stdfactor: float = 0.25):
        super().__init__()
        self.stdfactor = stdfactor
    
    def copy(self):
        """Return a copy of this hook with the same stdfactor parameter."""
        return self.__class__(stdfactor=self.stdfactor)
    
    def forward(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Store input for backward pass."""
        self.input = input[0] if isinstance(input, tuple) else input
        return output
    
    def backward(self, module: nn.Module, grad_input: tuple, grad_output: tuple) -> tuple:
        """
        Exact TensorFlow StdxEpsilonRule implementation.
        
        TF code equivalent:
        eps = np.std(ins) * self._stdfactor
        prepare_div = keras_layers.Lambda(lambda x: x + (K.cast(K.greater_equal(x, 0), K.floatx()) * 2 - 1) * eps)
        tmp = ilayers.SafeDivide()([reversed_outs, prepare_div(Zs)])
        tmp2 = tape.gradient(Zs, ins, output_gradients=tmp)
        ret = keras_layers.Multiply()([ins, tmp2])
        """
        if grad_output[0] is None:
            return grad_input
            
        relevance = grad_output[0]
        
        # Step 1: Calculate eps = np.std(ins) * self._stdfactor (per-layer, from layer input)
        # TF uses np.std(ins) where ins is the current layer's input in (B, H, W, C) format
        # Convert PyTorch (B, C, H, W) format to TensorFlow (B, H, W, C) format for std calculation
        if self.input.ndim == 4:  # Image: (B, C, H, W)
            # Convert to TF format (B, H, W, C) for std calculation to match TensorFlow exactly
            tf_format_input = self.input.permute(0, 2, 3, 1).detach().cpu().numpy()
            eps = float(np.std(tf_format_input)) * self.stdfactor
        else:  # Other formats (e.g., 1D or fully connected)
            eps = torch.std(self.input).item() * self.stdfactor
        
        # Step 2: Compute Zs = layer_wo_act(ins) - forward pass without bias for Conv/Linear
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
        
        # Step 3: Apply TensorFlow's exact stdx epsilon stabilization
        # TF: (K.cast(K.greater_equal(x, 0), K.floatx()) * 2 - 1) * eps
        tf_sign = (zs >= 0).float() * 2.0 - 1.0  # +1 for >=0, -1 for <0 (TF uses >= for 0)
        prepare_div = zs + tf_sign * eps
        
        # Step 4: SafeDivide - handle division by zero like TF
        safe_epsilon = 1e-12
        safe_prepare_div = torch.where(
            torch.abs(prepare_div) < safe_epsilon,
            torch.sign(prepare_div) * safe_epsilon,
            prepare_div
        )
        
        # Step 5: Divide relevance by stabilized activations
        tmp = relevance / safe_prepare_div
        
        # Step 6: Compute gradient-like operation (same as epsilon rule)
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
            # tmp has shape [batch_size, output_features], weight has shape [output_features, input_features]
            # Need to get [batch_size, input_features]
            gradient_result = torch.mm(tmp, module.weight)
        
        # Step 7: Final multiply by input (like TF)
        # TF: keras_layers.Multiply()([ins, tmp2])
        final_result = self.input * gradient_result
        
        return (final_result,) + grad_input[1:]


def create_tf_exact_stdx_epsilon_composite(stdfactor: float = 0.25):
    """Create a composite using TFExactStdxEpsilonHook that exactly matches TensorFlow."""
    from zennit.core import Composite
    from zennit.types import Convolution, Linear, BatchNorm, Activation, AvgPool
    
    def module_map(ctx, name, module):
        if isinstance(module, (Convolution, Linear)):
            # Create a separate hook instance for each layer to avoid sharing state
            return TFExactStdxEpsilonHook(stdfactor=stdfactor)
        elif isinstance(module, (BatchNorm, Activation, AvgPool)):
            return None
        return None
    
    return Composite(module_map=module_map)


def create_tf_exact_lrpz_stdx_epsilon_composite(stdfactor: float = 0.1):
    """
    Create a composite for TF-exact LRP Z + StdxEpsilon analysis.
    
    This exactly replicates TensorFlow iNNvestigate's behavior for methods like
    lrpz_epsilon_0_1_std_x which use:
    - Z rule for the first layer (input layer) 
    - StdxEpsilon rule for all other layers
    
    Args:
        stdfactor: Standard deviation factor for epsilon calculation (default: 0.1)
        
    Returns:
        Composite that applies TF-exact Z rule to first layer and StdxEpsilon to others
    """
    from zennit.core import Composite
    from zennit.types import Convolution, Linear
    from .tf_exact_epsilon_hook import TFExactEpsilonHook
    
    # Track if we've applied the first layer rule
    first_layer_applied = [False]
    
    def module_map(ctx, name, module):
        if isinstance(module, (Convolution, Linear)):
            if not first_layer_applied[0]:
                first_layer_applied[0] = True
                print(f"🔧 TF-Exact LRPZ+StdxEpsilon: Applying Z rule (Epsilon ε=0) to first layer: {name}")
                # Create separate instance for Z rule (Epsilon with ε=0)
                return TFExactEpsilonHook(epsilon=0.0)
            else:
                print(f"🔧 TF-Exact LRPZ+StdxEpsilon: Applying StdxEpsilon(stdfactor={stdfactor}) to layer: {name}")
                # Create separate instance for each layer to avoid sharing state
                return TFExactStdxEpsilonHook(stdfactor=stdfactor)
        return None
    
    return Composite(module_map=module_map)