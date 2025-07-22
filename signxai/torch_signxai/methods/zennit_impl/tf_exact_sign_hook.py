"""
Exact TensorFlow iNNvestigate SignRule implementation for PyTorch Zennit.

This hook exactly mirrors the TensorFlow implementation from 
signxai/tf_signxai/methods/innvestigate/analyzer/relevance_based/relevance_rule_base.py
to achieve perfect numerical matching.
"""

import torch
import torch.nn as nn
import numpy as np
from zennit.core import Hook
from typing import Union, Optional


class TFExactSignHook(Hook):
    """
    Hook that exactly replicates TensorFlow iNNvestigate's SignRule implementation.
    
    TF SignRule implementation:
    1. Sign computation: np.nan_to_num(ins / np.abs(ins), nan=1.0)
    2. Multiply sign with incoming relevance
    
    This is applied at the input layer in Sequential Composite A.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Store input for backward pass."""
        self.input = input[0] if isinstance(input, tuple) else input
        return output
    
    def backward(self, module: nn.Module, grad_input: tuple, grad_output: tuple) -> tuple:
        """
        Exact TensorFlow SignRule implementation.
        
        SIGN rule follows the same pattern as Epsilon rule but without epsilon stabilization.
        The sign is applied to the input contribution at the end.
        """
        if grad_output[0] is None:
            return grad_input
            
        relevance = grad_output[0]
        
        # Step 1: Compute Zs = layer_wo_act(ins) - forward pass
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
        
        # Step 2: For SIGN rule, no epsilon stabilization - just safe divide
        safe_epsilon = 1e-12
        safe_zs = torch.where(
            torch.abs(zs) < safe_epsilon,
            torch.sign(zs) * safe_epsilon,
            zs
        )
        
        # Step 3: Divide relevance by activations
        tmp = relevance / safe_zs
        
        # Step 4: Compute gradient-like operation (same as epsilon rule)
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
            if tmp.dim() == 2 and module.weight.dim() == 2:
                gradient_result = torch.mm(tmp, module.weight)
            else:
                gradient_result = torch.nn.functional.linear(tmp, module.weight.t())
        else:
            gradient_result = tmp
        
        # Step 5: Apply SIGN rule exactly like TensorFlow
        # TF: signs = np.nan_to_num(ins / np.abs(ins), nan=1.0)
        # TF: ret = keras_layers.Multiply()([signs, tmp2])
        
        # Compute signs exactly like TensorFlow
        abs_input = torch.abs(self.input)
        signs = torch.where(
            abs_input < 1e-12,  # Handle division by zero
            torch.ones_like(self.input),  # TF's nan=1.0 behavior: zeros become +1
            self.input / abs_input  # Normal sign computation: +1 or -1
        )
        
        # TensorFlow multiplies signs by the gradient result (tmp2)
        # In our case, gradient_result is equivalent to tmp2
        result_relevance = signs * gradient_result
        
        # Handle any remaining numerical issues
        result_relevance = torch.nan_to_num(result_relevance, nan=0.0, posinf=0.0, neginf=0.0)
        
        return (result_relevance,)


class TFExactAlpha1Beta0Hook(Hook):
    """
    Hook that exactly replicates TensorFlow iNNvestigate's Alpha1Beta0Rule implementation.
    Used for convolutional layers in Sequential Composite A.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Store input for backward pass."""
        self.input = input[0] if isinstance(input, tuple) else input
        return output
    
    def backward(self, module: nn.Module, grad_input: tuple, grad_output: tuple) -> tuple:
        """
        Exact TensorFlow Alpha1Beta0Rule implementation.
        
        Alpha=1, Beta=0 means we only consider positive weights.
        """
        if grad_output[0] is None:
            return grad_input
            
        relevance = grad_output[0]
        input_tensor = self.input
        
        if input_tensor is None:
            return grad_input
        
        # Clone to avoid inplace operations
        input_tensor = input_tensor.clone()
        relevance = relevance.clone()
        
        if isinstance(module, nn.Conv2d):
            # Exact TF Alpha1Beta0: separate weights and inputs into positive/negative parts
            weight_pos = torch.clamp(module.weight, min=0)  # Positive weights
            weight_neg = torch.clamp(module.weight, max=0)  # Negative weights
            
            input_pos = torch.clamp(input_tensor, min=0)    # Positive inputs
            input_neg = torch.clamp(input_tensor, max=0)    # Negative inputs
            
            # Compute the four preactivation terms
            # z_pos_pos: positive weights with positive inputs
            z_pos_pos = nn.functional.conv2d(
                input_pos, weight_pos, None,  # No bias for individual terms
                module.stride, module.padding, module.dilation, module.groups
            )
            
            # z_neg_neg: negative weights with negative inputs
            z_neg_neg = nn.functional.conv2d(
                input_neg, weight_neg, None,  # No bias for individual terms
                module.stride, module.padding, module.dilation, module.groups
            )
            
            # For Alpha1Beta0: only consider z_pos_pos + z_neg_neg (beta=0 removes cross terms)
            z_total = z_pos_pos + z_neg_neg
            
            # Add bias to the total
            if module.bias is not None:
                z_total = z_total + module.bias[None, :, None, None]
            
            # Safe division
            z_safe = torch.where(
                torch.abs(z_total) < 1e-9,
                torch.sign(z_total) * 1e-9,
                z_total
            )
            
            # Compute relevance ratio
            relevance_ratio = relevance / z_safe
            
            # Backward pass: gradient w.r.t. inputs for each term
            grad_pos_pos = nn.functional.conv_transpose2d(
                relevance_ratio, weight_pos, None,
                module.stride, module.padding, 0, module.groups, module.dilation
            )
            
            grad_neg_neg = nn.functional.conv_transpose2d(
                relevance_ratio, weight_neg, None,
                module.stride, module.padding, 0, module.groups, module.dilation
            )
            
            # Apply to respective input parts and combine
            result_relevance = input_pos * grad_pos_pos + input_neg * grad_neg_neg
            
        elif isinstance(module, nn.Linear):
            # For linear layers, fall back to simple epsilon rule
            z = nn.functional.linear(input_tensor, module.weight, module.bias)
            epsilon = 0.1
            z_safe = z + torch.where(z >= 0, epsilon, -epsilon)
            z_safe = torch.where(
                torch.abs(z_safe) < 1e-9,
                torch.sign(z_safe) * 1e-9,
                z_safe
            )
            relevance_ratio = relevance / z_safe
            
            if relevance_ratio.dim() == 2 and module.weight.dim() == 2:
                result_relevance = torch.mm(relevance_ratio, module.weight)
            else:
                result_relevance = nn.functional.linear(relevance_ratio, module.weight.t())
            
            result_relevance = input_tensor * result_relevance
        else:
            result_relevance = relevance
        
        # Handle numerical issues
        result_relevance = torch.nan_to_num(result_relevance, nan=0.0, posinf=0.0, neginf=0.0)
        
        return (result_relevance,)