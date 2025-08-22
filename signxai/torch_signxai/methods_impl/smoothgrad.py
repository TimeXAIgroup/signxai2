"""PyTorch implementation of SmoothGrad."""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Union, Optional


class SmoothGrad:
    """SmoothGrad attribution method.
    
    Implements SmoothGrad as described in the original paper:
    "SmoothGrad: removing noise by adding noise"
    https://arxiv.org/abs/1706.03825
    """
    
    def __init__(self, model, num_samples=50, noise_level=0.2):
        """Initialize SmoothGrad.
        
        Args:
            model: PyTorch model
            num_samples: Number of noisy samples to use
            noise_level: Level of noise to add as a fraction of the input range
        """
        self.model = model
        self.num_samples = num_samples
        self.noise_level = noise_level
        
    def attribute(self, inputs, target=None, num_samples=None, noise_level=None):
        """Calculate SmoothGrad attribution.
        
        Args:
            inputs: Input tensor
            target: Target class index (None for argmax)
            num_samples: Override the number of samples (optional)
            noise_level: Override the noise level (optional)
            
        Returns:
            Attribution tensor of the same shape as inputs
        """
        # Get parameters (use instance defaults if not provided)
        num_samples = num_samples if num_samples is not None else self.num_samples
        noise_level = noise_level if noise_level is not None else self.noise_level
        
        # Ensure input is a tensor
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float32)
            
        # Clone inputs to avoid modifying the original
        inputs = inputs.clone().detach()
        
        # Calculate input range for noise scaling
        input_range = inputs.max() - inputs.min()
        noise_std = noise_level * input_range
        
        # Store original model mode
        original_mode = self.model.training
        self.model.eval()
        
        # Accumulate gradients
        accumulated_gradients = torch.zeros_like(inputs)
        
        for i in range(num_samples):
            # Generate noisy input
            noise = torch.normal(0, noise_std.item(), size=inputs.shape, device=inputs.device)
            noisy_input = inputs + noise
            noisy_input.requires_grad_(True)
            
            # Forward pass
            self.model.zero_grad()
            output = self.model(noisy_input)
            
            # Determine target classes
            if target is None:
                target_indices = output.argmax(dim=1)
            elif isinstance(target, int):
                target_indices = torch.full((inputs.shape[0],), target, dtype=torch.long, device=inputs.device)
            elif isinstance(target, torch.Tensor):
                if target.numel() == 1:  # Single class for all examples
                    target_indices = torch.full((inputs.shape[0],), target.item(), dtype=torch.long, device=inputs.device)
                else:  # Different target for each example
                    target_indices = target
            else:
                raise ValueError(f"Unsupported target type: {type(target)}")
                
            # One-hot encoding for target classes
            one_hot = torch.zeros_like(output)
            one_hot.scatter_(1, target_indices.view(-1, 1), 1.0)
            
            # Backward pass
            output.backward(gradient=one_hot)
            
            # Accumulate gradients 
            if noisy_input.grad is not None:
                accumulated_gradients += noisy_input.grad
            
        # Restore model mode
        self.model.train(original_mode)
        
        # Average gradients
        smoothgrad_attribution = accumulated_gradients / num_samples
        
        # Apply small value thresholding for numerical stability
        smoothgrad_attribution[torch.abs(smoothgrad_attribution) < 1e-10] = 0.0
        
        return smoothgrad_attribution


class SmoothGradXInput(SmoothGrad):
    """SmoothGrad × Input attribution method.
    
    Implements SmoothGrad multiplied by the input, which can produce more
    visually appealing attributions by focusing on the important input features.
    """
    
    def attribute(self, inputs, target=None, num_samples=None, noise_level=None):
        """Calculate SmoothGrad × Input attribution.
        
        Args:
            inputs: Input tensor
            target: Target class index (None for argmax)
            num_samples: Override the number of samples (optional)
            noise_level: Override the noise level (optional)
            
        Returns:
            Attribution tensor of the same shape as inputs
        """
        # Get smooth gradients
        smooth_gradients = super().attribute(inputs, target, num_samples, noise_level)
        
        # Ensure input is a tensor
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float32)
            
        # Multiply by the original input (element-wise)
        attribution = smooth_gradients * inputs.clone().detach()
        
        return attribution


class SmoothGradXSign(SmoothGrad):
    """SmoothGrad × Sign attribution method.
    
    Implements SmoothGrad multiplied by the sign of (input - threshold),
    which can emphasize both positive and negative contributions.
    """
    
    def __init__(self, model, num_samples=50, noise_level=0.2, mu=0.0):
        """Initialize SmoothGradXSign.
        
        Args:
            model: PyTorch model
            num_samples: Number of noisy samples to use
            noise_level: Level of noise to add as a fraction of the input range
            mu: Threshold value for the sign function
        """
        super().__init__(model, num_samples, noise_level)
        self.mu = mu
        
    def attribute(self, inputs, target=None, num_samples=None, noise_level=None, mu=None):
        """Calculate SmoothGrad × Sign attribution.
        
        Args:
            inputs: Input tensor
            target: Target class index (None for argmax)
            num_samples: Override the number of samples (optional)
            noise_level: Override the noise level (optional)
            mu: Override the threshold value (optional)
            
        Returns:
            Attribution tensor of the same shape as inputs
        """
        # Get smooth gradients
        smooth_gradients = super().attribute(inputs, target, num_samples, noise_level)
        
        # Ensure input is a tensor
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float32)
            
        # Get threshold value (use instance default if not provided)
        mu_value = mu if mu is not None else self.mu
        
        # Calculate sign of (input - threshold)
        input_sign = torch.sign(inputs.clone().detach() - mu_value)
        
        # Multiply by the sign (element-wise)
        attribution = smooth_gradients * input_sign
        
        return attribution


def smoothgrad(model, inputs, target=None, num_samples=50, noise_level=0.2):
    """Calculate SmoothGrad attribution (functional API).
    
    Args:
        model: PyTorch model
        inputs: Input tensor
        target: Target class index (None for argmax)
        num_samples: Number of noisy samples to use
        noise_level: Level of noise to add as a fraction of the input range
        
    Returns:
        Attribution tensor of the same shape as inputs
    """
    # Create SmoothGrad instance and calculate attribution
    return SmoothGrad(model, num_samples, noise_level).attribute(inputs, target)


def smoothgrad_x_input(model, inputs, target=None, num_samples=50, noise_level=0.2):
    """Calculate SmoothGrad × Input attribution (functional API).
    
    Args:
        model: PyTorch model
        inputs: Input tensor
        target: Target class index (None for argmax)
        num_samples: Number of noisy samples to use
        noise_level: Level of noise to add as a fraction of the input range
        
    Returns:
        Attribution tensor of the same shape as inputs
    """
    # Create SmoothGradXInput instance and calculate attribution
    return SmoothGradXInput(model, num_samples, noise_level).attribute(inputs, target)


def smoothgrad_x_sign(model, inputs, target=None, num_samples=50, noise_level=0.2, mu=0.0):
    """Calculate SmoothGrad × Sign attribution (functional API).
    
    Args:
        model: PyTorch model
        inputs: Input tensor
        target: Target class index (None for argmax)
        num_samples: Number of noisy samples to use
        noise_level: Level of noise to add as a fraction of the input range
        mu: Threshold value for the sign function
        
    Returns:
        Attribution tensor of the same shape as inputs
    """
    # Create SmoothGradXSign instance and calculate attribution
    return SmoothGradXSign(model, num_samples, noise_level, mu).attribute(inputs, target)