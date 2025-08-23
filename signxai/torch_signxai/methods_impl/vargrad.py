"""VarGrad implementation and variants for PyTorch."""

import torch
import numpy as np
from typing import Union, Optional, List
from .base import BaseGradient


class VarGrad(BaseGradient):
    """VarGrad attribution method."""
    
    def __init__(self, model, noise_level: float = 0.2, num_samples: int = 50):
        """Initialize with a PyTorch model.
        
        Args:
            model: PyTorch model for which to calculate gradients
            noise_level: Level of noise to add to inputs (default 0.2)
            num_samples: Number of samples to average (default 50)
        """
        super().__init__(model)
        self.noise_level = noise_level
        self.num_samples = num_samples
    
    def attribute(self, inputs: torch.Tensor, target: Optional[Union[int, torch.Tensor]] = None,
                  noise_level: Optional[float] = None, num_samples: Optional[int] = None) -> torch.Tensor:
        """Calculate VarGrad attribution.
        
        Args:
            inputs: Input tensor
            target: Target class index or tensor (None uses argmax)
            noise_level: Level of noise to add to inputs (if None, use self.noise_level)
            num_samples: Number of samples to average (if None, use self.num_samples)
            
        Returns:
            Attribution tensor of the same shape as inputs
        """
        # Use provided parameters or defaults
        noise_level = noise_level if noise_level is not None else self.noise_level
        num_samples = num_samples if num_samples is not None else self.num_samples
        
        # Calculate noise standard deviation
        input_range = inputs.max() - inputs.min()
        stdev = noise_level * input_range
        
        # Original model mode
        original_mode = self.model.training
        self.model.eval()
        
        # Accumulate gradients from noisy samples
        all_gradients = []
        for _ in range(num_samples):
            # Add noise to inputs
            noise = torch.normal(0.0, stdev.item(), size=inputs.shape, device=inputs.device)
            noisy_input = (inputs + noise).clone().detach().requires_grad_(True)
            
            # Forward pass
            self.model.zero_grad()
            outputs = self.model(noisy_input)
            
            # Handle target
            if target is None:
                # Use argmax
                target_indices = outputs.argmax(dim=1)
            elif isinstance(target, int):
                # Use the same target for all examples in the batch
                target_indices = torch.full((inputs.shape[0],), target, dtype=torch.long, device=inputs.device)
            elif isinstance(target, torch.Tensor):
                if target.numel() == 1:
                    # Single target tensor for all examples
                    target_indices = torch.full((inputs.shape[0],), target.item(), dtype=torch.long, device=inputs.device)
                else:
                    # Target tensor with different targets for each example
                    target_indices = target.to(dtype=torch.long, device=inputs.device)
            else:
                raise ValueError(f"Unsupported target type: {type(target)}")
            
            # Create one-hot encoding
            one_hot = torch.zeros_like(outputs)
            one_hot.scatter_(1, target_indices.unsqueeze(1), 1.0)
            
            # Backward pass
            outputs.backward(gradient=one_hot)
            
            # Store gradients
            if noisy_input.grad is None:
                all_gradients.append(torch.zeros_like(inputs))
            else:
                all_gradients.append(noisy_input.grad.clone())
        
        # Restore model mode
        self.model.train(original_mode)
        
        # Stack and calculate variance of gradients (instead of mean like in SmoothGrad)
        variance_gradients = torch.stack(all_gradients).var(dim=0)
        
        # Threshold small values for numerical stability
        variance_gradients[torch.abs(variance_gradients) < 1e-10] = 0.0
        
        return variance_gradients


class VarGradXInput(VarGrad):
    """VarGrad times Input attribution method."""
    
    def attribute(self, inputs: torch.Tensor, target: Optional[Union[int, torch.Tensor]] = None,
                  noise_level: Optional[float] = None, num_samples: Optional[int] = None) -> torch.Tensor:
        """Calculate VarGrad times input attribution.
        
        Args:
            inputs: Input tensor
            target: Target class index or tensor (None uses argmax)
            noise_level: Level of noise to add to inputs (if None, use self.noise_level)
            num_samples: Number of samples to average (if None, use self.num_samples)
            
        Returns:
            Attribution tensor of the same shape as inputs
        """
        # Get VarGrad attributions
        var_gradients = super().attribute(inputs, target, noise_level, num_samples)
        
        # Multiply by the input
        return inputs * var_gradients


class VarGradXSign(VarGrad):
    """VarGrad times Sign attribution method."""
    
    def __init__(self, model, noise_level: float = 0.2, num_samples: int = 50, mu: float = 0.0):
        """Initialize with a PyTorch model.
        
        Args:
            model: PyTorch model for which to calculate gradients
            noise_level: Level of noise to add to inputs (default 0.2)
            num_samples: Number of samples to average (default 50)
            mu: Threshold for sign determination (default 0.0)
        """
        super().__init__(model, noise_level, num_samples)
        self.mu = mu
    
    def attribute(self, inputs: torch.Tensor, target: Optional[Union[int, torch.Tensor]] = None,
                  noise_level: Optional[float] = None, num_samples: Optional[int] = None) -> torch.Tensor:
        """Calculate VarGrad times sign attribution.
        
        Args:
            inputs: Input tensor
            target: Target class index or tensor (None uses argmax)
            noise_level: Level of noise to add to inputs (if None, use self.noise_level)
            num_samples: Number of samples to average (if None, use self.num_samples)
            
        Returns:
            Attribution tensor of the same shape as inputs
        """
        # Get VarGrad attributions
        var_gradients = super().attribute(inputs, target, noise_level, num_samples)
        
        # Generate sign map based on input values and mu threshold
        sign_map = torch.where(
            inputs < self.mu,
            torch.tensor(-1.0, device=inputs.device),
            torch.tensor(1.0, device=inputs.device)
        )
        
        # Multiply by the sign map
        return var_gradients * sign_map