"""
SIGN and SIGNmu rule implementations for Zennit and PyTorch.
These custom rules implement the SIGN and SIGNmu rules from TensorFlow InvestiNNs.
"""
import torch
import numpy as np
from zennit.rules import BasicHook
from zennit.types import Convolution, Linear


class SIGNRule(BasicHook):
    """
    SIGN rule from the TensorFlow implementation.
    This rule uses the sign of the input to propagate relevance.
    
    Args:
        bias (bool, optional): Whether to include bias in the computation.
            Default: True.
    """
    
    def __init__(self, bias=True):
        """
        Initialize SIGN rule.
        
        Args:
            bias (bool, optional): Whether to include bias in the computation.
                Default: True.
        """
        super().__init__()
        self.bias = bias
    
    def forward(self, module, input_tensor, output_tensor):
        """
        Store input and output tensors for the backward pass.
        
        Args:
            module (nn.Module): PyTorch module for which this rule is being applied.
            input_tensor (Tensor): Input tensor to the module.
            output_tensor (Tensor): Output tensor from the module.
            
        Returns:
            Tuple[Tensor, callable]: The output tensor and the backward function.
        """
        # Store for backward pass
        self._module = module
        self._input = input_tensor
        self._output = output_tensor
        
        # Return output unchanged and provide backward function
        return output_tensor, self._backward
    
    def _backward(self, relevance_tensor):
        """
        Backward pass for SIGN rule. 
        Handles the propagation of relevance with sign information.
        
        Args:
            relevance_tensor (Tensor): Relevance tensor from next layer.
            
        Returns:
            Tensor: Propagated relevance for the previous layer.
        """
        # Get the input tensor (handle tuple case)
        if isinstance(self._input, tuple):
            input_tensor = self._input[0].detach().requires_grad_(True)
        else:
            input_tensor = self._input.detach().requires_grad_(True)
        with torch.enable_grad():
            # Create the layer's output without activation
            # We need to handle different module types (Conv, Linear, etc.)
            if isinstance(self._module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
                output = torch.nn.functional.conv2d(
                    input_tensor, 
                    self._module.weight, 
                    self._module.bias if self.bias else None, 
                    self._module.stride, 
                    self._module.padding, 
                    self._module.dilation, 
                    self._module.groups
                )
            elif isinstance(self._module, torch.nn.Linear):
                output = torch.nn.functional.linear(
                    input_tensor, 
                    self._module.weight, 
                    self._module.bias if self.bias else None
                )
            else:
                # For other layer types, pass through
                output = self._module(input_tensor)
            
            # Compute gradients
            output.backward(relevance_tensor, retain_graph=True)
            gradients = input_tensor.grad.clone()
            
            # Get sign of the input
            sign = torch.sign(input_tensor.detach())
            # Handle zeros: sign(0) = 1 in this implementation
            sign[sign == 0] = 1.0
            
            # Apply sign to the gradients
            result = gradients * sign
            
            return result


class SIGNmuRule(BasicHook):
    """
    SIGNmu rule from the TensorFlow implementation.
    This rule uses a threshold mu to determine the sign of the input for relevance propagation.
    
    Args:
        mu (float, optional): Threshold for SIGN function. 
            Values >= mu will get +1, values < mu will get -1.
            Default: 0.0.
        bias (bool, optional): Whether to include bias in the computation.
            Default: True.
    """
    
    def __init__(self, mu=0.0, bias=True):
        """
        Initialize SIGNmu rule.
        
        Args:
            mu (float, optional): Threshold for SIGN function.
                Values >= mu will get +1, values < mu will get -1.
                Default: 0.0.
            bias (bool, optional): Whether to include bias in the computation.
                Default: True.
        """
        super().__init__()
        self.mu = mu
        self.bias = bias
    
    def forward(self, module, input_tensor, output_tensor):
        """
        Store input and output tensors for the backward pass.
        
        Args:
            module (nn.Module): PyTorch module for which this rule is being applied.
            input_tensor (Tensor): Input tensor to the module.
            output_tensor (Tensor): Output tensor from the module.
            
        Returns:
            Tuple[Tensor, callable]: The output tensor and the backward function.
        """
        # Store for backward pass
        self._module = module
        self._input = input_tensor
        self._output = output_tensor
        
        # Return output unchanged and provide backward function
        return output_tensor, self._backward
    
    def _backward(self, relevance_tensor):
        """
        Backward pass for SIGNmu rule.
        Uses a threshold mu to determine the sign for relevance propagation.
        
        Args:
            relevance_tensor (Tensor): Relevance tensor from next layer.
            
        Returns:
            Tensor: Propagated relevance for the previous layer.
        """
        # Get the input tensor (handle tuple case)
        if isinstance(self._input, tuple):
            input_tensor = self._input[0].detach().requires_grad_(True)
        else:
            input_tensor = self._input.detach().requires_grad_(True)
        with torch.enable_grad():
            # Create the layer's output without activation
            # We need to handle different module types (Conv, Linear, etc.)
            if isinstance(self._module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
                output = torch.nn.functional.conv2d(
                    input_tensor, 
                    self._module.weight, 
                    self._module.bias if self.bias else None, 
                    self._module.stride, 
                    self._module.padding, 
                    self._module.dilation, 
                    self._module.groups
                )
            elif isinstance(self._module, torch.nn.Linear):
                output = torch.nn.functional.linear(
                    input_tensor, 
                    self._module.weight, 
                    self._module.bias if self.bias else None
                )
            else:
                # For other layer types, pass through
                output = self._module(input_tensor)
            
            # Compute gradients
            output.backward(relevance_tensor, retain_graph=True)
            gradients = input_tensor.grad.clone()
            
            # Create thresholded sign based on mu
            sign_mu = torch.ones_like(input_tensor.detach())
            sign_mu[input_tensor < self.mu] = -1.0
            
            # Apply SIGNmu to the gradients
            result = gradients * sign_mu
            
            return result