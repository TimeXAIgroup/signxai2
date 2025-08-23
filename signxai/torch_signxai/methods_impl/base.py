"""Base gradient attribution methods for PyTorch."""
import torch
import numpy as np
from typing import Union, Optional


class BaseGradient:
    """Base gradient attribution method."""
    
    def __init__(self, model):
        """Initialize with a PyTorch model.
        
        Args:
            model: PyTorch model for which to calculate gradients
        """
        self.model = model
    
    def attribute(self, inputs: torch.Tensor, target: Optional[Union[int, torch.Tensor]] = None) -> torch.Tensor:
        """Calculate gradient attribution.
        
        Args:
            inputs: Input tensor
            target: Target class index or tensor (None uses argmax)
            
        Returns:
            Gradient tensor of the same shape as inputs
        """
        # Clone the input and set requires_grad
        inputs_grad = inputs.clone().detach().requires_grad_(True)
        
        # Set model to eval mode
        original_mode = self.model.training
        self.model.eval()
        
        # Forward pass
        self.model.zero_grad()
        outputs = self.model(inputs_grad)
        
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
        
        # Get gradients
        gradients = inputs_grad.grad
        
        # Threshold very small values to zero for numerical stability
        gradients[torch.abs(gradients) < 1e-10] = 0.0
        
        # Restore model mode
        self.model.train(original_mode)
        
        return gradients


class InputXGradient(BaseGradient):
    """Input times gradient attribution method."""
    
    def attribute(self, inputs: torch.Tensor, target: Optional[Union[int, torch.Tensor]] = None) -> torch.Tensor:
        """Calculate input times gradient attribution.
        
        Args:
            inputs: Input tensor
            target: Target class index or tensor (None uses argmax)
            
        Returns:
            Attribution tensor of the same shape as inputs
        """
        # Get gradients using the parent class
        gradients = super().attribute(inputs, target)
        
        # Element-wise multiply with inputs
        return inputs * gradients


class GradientXSign(BaseGradient):
    """Gradient times sign attribution method."""
    
    def __init__(self, model, mu: float = 0.0):
        """Initialize with a PyTorch model and threshold.
        
        Args:
            model: PyTorch model for which to calculate gradients
            mu: Threshold for sign determination (default 0.0)
        """
        super().__init__(model)
        self.mu = mu
    
    def attribute(self, inputs: torch.Tensor, target: Optional[Union[int, torch.Tensor]] = None) -> torch.Tensor:
        """Calculate gradient times sign attribution.
        
        Args:
            inputs: Input tensor
            target: Target class index or tensor (None uses argmax)
            
        Returns:
            Attribution tensor of the same shape as inputs
        """
        # Get gradients using the parent class
        gradients = super().attribute(inputs, target)
        
        # Generate sign map based on input values and mu threshold
        sign_map = torch.sign(inputs - self.mu)
        
        # Element-wise multiply gradients with sign map
        return gradients * sign_map