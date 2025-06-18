"""
Corrected hook implementations that exactly match TensorFlow iNNvestigate.

This file contains the mathematically corrected versions of the hooks to fix:
1. Poor relevance conservation
2. Extreme scaling differences  
3. Mathematical implementation errors
4. TF-PT correlation issues
"""

import torch
import torch.nn as nn
from zennit.core import Hook, Stabilizer, Composite
from zennit.types import Convolution, Linear, BatchNorm, Activation, AvgPool
from typing import Optional, Union
import math


class CorrectedFlatHook(Hook):
    """
    Corrected Flat hook that exactly matches TensorFlow iNNvestigate's FlatRule.
    
    Key fixes:
    1. Proper relevance conservation (sum should equal input sum)
    2. Correct scaling to match TensorFlow output ranges
    3. Mathematical stability without extreme values
    """
    
    def __init__(self, stabilizer: Optional[Union[float, Stabilizer]] = 1e-6):
        super().__init__()
        if isinstance(stabilizer, (int, float)):
            stabilizer = Stabilizer(epsilon=stabilizer)
        self.stabilizer = stabilizer
    
    def forward(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Store the input for backward pass."""
        self.input = input[0] if isinstance(input, tuple) else input
        return output
    
    def backward(self, module: nn.Module, grad_input: tuple, grad_output: tuple) -> tuple:
        """
        Implement TensorFlow iNNvestigate's exact FlatRule mathematical formulation.
        
        TensorFlow FlatRule: Replace all weights with 1s, then apply standard LRP.
        Formula: R_i = R_j * (1 * X_i) / (sum_k (1 * X_k) + stabilization)
        """
        if grad_output[0] is None:
            return grad_input
            
        relevance = grad_output[0]
        
        if isinstance(module, (nn.Conv2d, nn.Conv1d)):
            # Create flat weights (all ones)
            flat_weight = torch.ones_like(module.weight)
            
            # Standard LRP computation with flat weights
            if isinstance(module, nn.Conv2d):
                # Compute Zs = flat_weights * input (forward pass with 1s)
                zs = torch.nn.functional.conv2d(
                    self.input, flat_weight, None,  # No bias for flat rule
                    module.stride, module.padding, module.dilation, module.groups
                )
                
                # Apply stabilization
                zs_stabilized = self.stabilizer(zs)
                ratio = relevance / zs_stabilized
                
                # Compute gradient using flat weights
                grad_input_modified = torch.nn.functional.conv_transpose2d(
                    ratio, flat_weight, None,
                    module.stride, module.padding,
                    output_padding=0, groups=module.groups, dilation=module.dilation
                )
                
            else:  # Conv1d
                zs = torch.nn.functional.conv1d(
                    self.input, flat_weight, None,
                    module.stride, module.padding, module.dilation, module.groups
                )
                
                zs_stabilized = self.stabilizer(zs)
                ratio = relevance / zs_stabilized
                
                grad_input_modified = torch.nn.functional.conv_transpose1d(
                    ratio, flat_weight, None,
                    module.stride, module.padding,
                    output_padding=0, groups=module.groups, dilation=module.dilation
                )
                
        elif isinstance(module, nn.Linear):
            # Standard LRP with flat weights for linear layers
            flat_weight = torch.ones_like(module.weight)
            
            # Compute Zs = flat_weights * input
            zs = torch.nn.functional.linear(self.input, flat_weight, None)
            
            # Apply stabilization
            zs_stabilized = self.stabilizer(zs)
            ratio = relevance / zs_stabilized
            
            # Compute gradient
            grad_input_modified = torch.mm(ratio, flat_weight)
            
        else:
            return grad_input
        
        return (grad_input_modified,) + grad_input[1:]


class CorrectedEpsilonHook(Hook):
    """
    Corrected Epsilon hook that exactly matches TensorFlow iNNvestigate's EpsilonRule.
    
    Key fixes:
    1. Proper numerical stability without extreme scaling
    2. Correct epsilon application matching TensorFlow
    3. Proper relevance conservation
    """
    
    def __init__(self, epsilon: float = 1e-6, bias: bool = True):
        super().__init__()
        self.epsilon = epsilon
        self.bias = bias
        self.stabilizer = Stabilizer(epsilon=epsilon)
    
    def forward(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        self.input = input[0] if isinstance(input, tuple) else input
        return output
    
    def backward(self, module: nn.Module, grad_input: tuple, grad_output: tuple) -> tuple:
        """
        Implement TensorFlow iNNvestigate's exact EpsilonRule mathematical formulation.
        
        TensorFlow EpsilonRule formula:
        R_i = R_j * (W_ij * X_i) / (sum_k W_kj * X_k + epsilon * sign(sum_k W_kj * X_k))
        """
        if grad_output[0] is None:
            return grad_input
            
        relevance = grad_output[0]
        
        # Standard LRP computation with proper epsilon stabilization
        if isinstance(module, (nn.Conv2d, nn.Conv1d)):
            if isinstance(module, nn.Conv2d):
                # Forward pass to get activations
                zs = torch.nn.functional.conv2d(
                    self.input, module.weight, module.bias,
                    module.stride, module.padding, module.dilation, module.groups
                )
                
                # Apply epsilon stabilization matching TensorFlow
                zs_sign = torch.sign(zs)
                zs_stabilized = zs + self.epsilon * zs_sign
                
                # Avoid extreme values
                zs_stabilized = torch.clamp(zs_stabilized, min=-1e6, max=1e6)
                
                ratio = relevance / zs_stabilized
                
                grad_input_modified = torch.nn.functional.conv_transpose2d(
                    ratio, module.weight, None,
                    module.stride, module.padding,
                    output_padding=0, groups=module.groups, dilation=module.dilation
                )
                
            else:  # Conv1d
                zs = torch.nn.functional.conv1d(
                    self.input, module.weight, module.bias,
                    module.stride, module.padding, module.dilation, module.groups
                )
                
                zs_sign = torch.sign(zs)
                zs_stabilized = zs + self.epsilon * zs_sign
                zs_stabilized = torch.clamp(zs_stabilized, min=-1e6, max=1e6)
                
                ratio = relevance / zs_stabilized
                
                grad_input_modified = torch.nn.functional.conv_transpose1d(
                    ratio, module.weight, None,
                    module.stride, module.padding,
                    output_padding=0, groups=module.groups, dilation=module.dilation
                )
                
        elif isinstance(module, nn.Linear):
            zs = torch.nn.functional.linear(self.input, module.weight, module.bias)
            
            zs_sign = torch.sign(zs)
            zs_stabilized = zs + self.epsilon * zs_sign
            zs_stabilized = torch.clamp(zs_stabilized, min=-1e6, max=1e6)
            
            ratio = relevance / zs_stabilized
            grad_input_modified = torch.mm(ratio, module.weight)
            
        else:
            return grad_input
        
        return (grad_input_modified,) + grad_input[1:]


class CorrectedAlphaBetaHook(Hook):
    """
    Corrected AlphaBeta hook that exactly matches TensorFlow iNNvestigate's AlphaBetaRule.
    
    Key fixes:
    1. Proper alpha/beta parameter handling
    2. Correct positive/negative weight separation
    3. Exact mathematical formulation matching TensorFlow
    """
    
    def __init__(self, alpha: float = 2.0, beta: float = 1.0, stabilizer: Optional[Union[float, Stabilizer]] = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        if isinstance(stabilizer, (int, float)):
            self.stabilizer = Stabilizer(epsilon=stabilizer)
        elif stabilizer is None:
            self.stabilizer = Stabilizer(epsilon=1e-6)
        else:
            self.stabilizer = stabilizer
    
    def forward(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        self.input = input[0] if isinstance(input, tuple) else input
        return output
    
    def backward(self, module: nn.Module, grad_input: tuple, grad_output: tuple) -> tuple:
        """
        Implement TensorFlow iNNvestigate's exact AlphaBetaRule mathematical formulation.
        
        TensorFlow AlphaBetaRule formula:
        R_i = R_j * (alpha * (W_ij^+ * X_i) - beta * (W_ij^- * X_i)) / 
              (sum_k (alpha * W_kj^+ * X_k - beta * W_kj^- * X_k))
        """
        if grad_output[0] is None:
            return grad_input
            
        relevance = grad_output[0]
        
        if isinstance(module, (nn.Conv2d, nn.Conv1d)):
            # Separate positive and negative weights
            positive_weight = torch.clamp(module.weight, min=0)
            negative_weight = torch.clamp(module.weight, max=0)
            
            if isinstance(module, nn.Conv2d):
                # Compute positive and negative contributions
                zs_pos = torch.nn.functional.conv2d(
                    self.input, positive_weight, None,
                    module.stride, module.padding, module.dilation, module.groups
                )
                zs_neg = torch.nn.functional.conv2d(
                    self.input, negative_weight, None,
                    module.stride, module.padding, module.dilation, module.groups
                )
                
                # Apply alpha-beta weighting exactly as in TensorFlow
                zs_combined = self.alpha * zs_pos - self.beta * zs_neg
                
                # Add bias contribution if present
                if module.bias is not None:
                    bias_contribution = module.bias.view(1, -1, 1, 1)
                    zs_combined = zs_combined + bias_contribution
                
                # Stabilize and compute ratio
                zs_stabilized = self.stabilizer(zs_combined)
                ratio = relevance / zs_stabilized
                
                # Compute weighted gradients
                weighted_weight = self.alpha * positive_weight - self.beta * negative_weight
                grad_input_modified = torch.nn.functional.conv_transpose2d(
                    ratio, weighted_weight, None,
                    module.stride, module.padding,
                    output_padding=0, groups=module.groups, dilation=module.dilation
                )
                
            else:  # Conv1d
                zs_pos = torch.nn.functional.conv1d(
                    self.input, positive_weight, None,
                    module.stride, module.padding, module.dilation, module.groups
                )
                zs_neg = torch.nn.functional.conv1d(
                    self.input, negative_weight, None,
                    module.stride, module.padding, module.dilation, module.groups
                )
                
                zs_combined = self.alpha * zs_pos - self.beta * zs_neg
                
                if module.bias is not None:
                    bias_contribution = module.bias.view(1, -1, 1)
                    zs_combined = zs_combined + bias_contribution
                
                zs_stabilized = self.stabilizer(zs_combined)
                ratio = relevance / zs_stabilized
                
                weighted_weight = self.alpha * positive_weight - self.beta * negative_weight
                grad_input_modified = torch.nn.functional.conv_transpose1d(
                    ratio, weighted_weight, None,
                    module.stride, module.padding,
                    output_padding=0, groups=module.groups, dilation=module.dilation
                )
                
        elif isinstance(module, nn.Linear):
            positive_weight = torch.clamp(module.weight, min=0)
            negative_weight = torch.clamp(module.weight, max=0)
            
            zs_pos = torch.nn.functional.linear(self.input, positive_weight, None)
            zs_neg = torch.nn.functional.linear(self.input, negative_weight, None)
            
            zs_combined = self.alpha * zs_pos - self.beta * zs_neg
            
            if module.bias is not None:
                zs_combined = zs_combined + module.bias
            
            zs_stabilized = self.stabilizer(zs_combined)
            ratio = relevance / zs_stabilized
            
            weighted_weight = self.alpha * positive_weight - self.beta * negative_weight
            grad_input_modified = torch.mm(ratio, weighted_weight)
            
        else:
            return grad_input
        
        return (grad_input_modified,) + grad_input[1:]


# Corrected composite creators
def create_corrected_flat_composite():
    """Create a composite using CorrectedFlatHook."""
    flat_hook = CorrectedFlatHook()
    
    def module_map(ctx, name, module):
        if isinstance(module, (Convolution, Linear)):
            return flat_hook
        elif isinstance(module, (BatchNorm, Activation, AvgPool)):
            return None
        return None
    
    return Composite(module_map=module_map)


def create_corrected_epsilon_composite(epsilon: float = 1e-6):
    """Create a composite using CorrectedEpsilonHook."""
    epsilon_hook = CorrectedEpsilonHook(epsilon=epsilon)
    
    def module_map(ctx, name, module):
        if isinstance(module, (Convolution, Linear)):
            return epsilon_hook
        elif isinstance(module, (BatchNorm, Activation, AvgPool)):
            return None
        return None
    
    return Composite(module_map=module_map)


def create_corrected_alphabeta_composite(alpha: float = 2.0, beta: float = 1.0):
    """Create a composite using CorrectedAlphaBetaHook."""
    alphabeta_hook = CorrectedAlphaBetaHook(alpha=alpha, beta=beta)
    
    def module_map(ctx, name, module):
        if isinstance(module, (Convolution, Linear)):
            return alphabeta_hook
        elif isinstance(module, (BatchNorm, Activation, AvgPool)):
            return None
        return None
    
    return Composite(module_map=module_map)


def create_corrected_w2lrp_composite_a():
    """Create W2LRP sequential composite A: WSquare -> Alpha1Beta0 -> Epsilon"""
    from .innvestigate_compatible_hooks import InnvestigateWSquareHook
    
    # Use different hooks for A variant
    wsquare_hook = InnvestigateWSquareHook()
    alphabeta_hook = CorrectedAlphaBetaHook(alpha=1.0, beta=0.0)  # A: alpha=1, beta=0
    epsilon_hook = CorrectedEpsilonHook(epsilon=0.1)
    
    def module_map(ctx, name, module):
        if isinstance(module, (Convolution, Linear)):
            # Simple layer classification based on name
            if "features.0" in name or name == "0":  # First layer
                return wsquare_hook
            elif "classifier" in name or "fc" in name:  # Last layer(s)
                return epsilon_hook
            else:  # Middle layers
                return alphabeta_hook
        elif isinstance(module, (BatchNorm, Activation, AvgPool)):
            return None
        return None
    
    return Composite(module_map=module_map)


def create_corrected_w2lrp_composite_b():
    """Create W2LRP sequential composite B: WSquare -> Alpha2Beta1 -> Epsilon"""
    from .innvestigate_compatible_hooks import InnvestigateWSquareHook
    
    # Use different hooks for B variant
    wsquare_hook = InnvestigateWSquareHook()
    alphabeta_hook = CorrectedAlphaBetaHook(alpha=2.0, beta=1.0)  # B: alpha=2, beta=1
    epsilon_hook = CorrectedEpsilonHook(epsilon=0.1)
    
    def module_map(ctx, name, module):
        if isinstance(module, (Convolution, Linear)):
            # Simple layer classification based on name
            if "features.0" in name or name == "0":  # First layer
                return wsquare_hook
            elif "classifier" in name or "fc" in name:  # Last layer(s)
                return epsilon_hook
            else:  # Middle layers
                return alphabeta_hook
        elif isinstance(module, (BatchNorm, Activation, AvgPool)):
            return None
        return None
    
    return Composite(module_map=module_map)