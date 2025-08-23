"""Integrated Gradients implementation and variants for PyTorch.

Implements the method described in "Axiomatic Attribution for Deep Networks"
(https://arxiv.org/abs/1703.01365).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union, Optional, List
from .base import BaseGradient


class IntegratedGradients(BaseGradient):
    """Integrated Gradients attribution method."""
    
    def __init__(self, model, steps: int = 50, baseline_type: str = "zero"):
        """Initialize with a PyTorch model.
        
        Args:
            model: PyTorch model for which to calculate gradients
            steps: Number of interpolation steps (default 50)
            baseline_type: Type of baseline to use (default "zero")
        """
        super().__init__(model)
        self.steps = steps
        self.baseline_type = baseline_type
    
    def _create_baseline(self, inputs: torch.Tensor) -> torch.Tensor:
        """Create baseline tensor based on baseline_type.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Baseline tensor of the same shape as inputs
        """
        if self.baseline_type == "zero" or self.baseline_type is None:
            return torch.zeros_like(inputs)
        elif self.baseline_type == "black":
            return torch.zeros_like(inputs)
        elif self.baseline_type == "white":
            return torch.ones_like(inputs)
        elif self.baseline_type == "gaussian":
            return torch.randn_like(inputs) * 0.1
        else:
            raise ValueError(f"Unsupported baseline_type: {self.baseline_type}")
    
    def attribute(self, inputs: torch.Tensor, target: Optional[Union[int, torch.Tensor]] = None, 
                  baseline: Optional[torch.Tensor] = None, steps: Optional[int] = None,
                  baselines: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate integrated gradients attribution.
        
        Args:
            inputs: Input tensor
            target: Target class index or tensor (None uses argmax)
            baseline: Baseline tensor (if None, created based on baseline_type)
            baselines: Alternative spelling for baseline (for compatibility)
            steps: Number of interpolation steps (if None, use self.steps)
            
        Returns:
            Attribution tensor of the same shape as inputs
        """
        # Handle both baseline and baselines parameters for compatibility
        if baseline is None and baselines is not None:
            baseline = baselines
            
        # Use provided parameters or defaults
        steps = steps if steps is not None else self.steps
        baseline = baseline if baseline is not None else self._create_baseline(inputs)
        
        # Ensure input is a tensor
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float32)
            
        # Clone inputs and baseline to avoid modifying originals
        inputs = inputs.clone().detach()
        baseline = baseline.clone().detach().to(inputs.device, inputs.dtype)
        
        # Ensure baseline has the same shape as inputs
        if baseline.shape != inputs.shape:
            raise ValueError(f"Baseline shape {baseline.shape} must match inputs shape {inputs.shape}")
        
        # Save original model mode
        original_mode = self.model.training
        self.model.eval()
        
        # Create scaled inputs
        scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(steps + 1)]
        
        # Calculate gradients for each scaled input
        gradients = []
        for scaled_input in scaled_inputs:
            # Clone and set requires_grad
            scaled_input_grad = scaled_input.clone().detach().requires_grad_(True)
            
            # Forward pass
            self.model.zero_grad()
            outputs = self.model(scaled_input_grad)
            
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
            if scaled_input_grad.grad is None:
                gradients.append(torch.zeros_like(inputs))
            else:
                gradients.append(scaled_input_grad.grad.clone())
        
        # Restore model mode
        self.model.train(original_mode)
        
        # Stack gradients
        gradients_tensor = torch.stack(gradients)
        
        # Compute trapezoidal approximation
        avg_gradients = (gradients_tensor[:-1] + gradients_tensor[1:]) / 2.0
        integrated_gradients = torch.mean(avg_gradients, dim=0) * (inputs - baseline)
        
        # Threshold small values for numerical stability
        integrated_gradients[torch.abs(integrated_gradients) < 1e-10] = 0.0
        
        return integrated_gradients


class IntegratedGradientsXInput(IntegratedGradients):
    """Integrated Gradients times Input attribution method."""
    
    def attribute(self, inputs: torch.Tensor, target: Optional[Union[int, torch.Tensor]] = None, 
                 baseline: Optional[torch.Tensor] = None, steps: Optional[int] = None,
                 baselines: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate integrated gradients times input attribution.
        
        Args:
            inputs: Input tensor
            target: Target class index or tensor (None uses argmax)
            baseline: Baseline tensor (if None, created based on baseline_type)
            baselines: Alternative spelling for baseline (for compatibility)
            steps: Number of interpolation steps (if None, use self.steps)
            
        Returns:
            Attribution tensor of the same shape as inputs
        """
        # Get integrated gradients
        integrated_gradients = super().attribute(inputs, target, baseline, steps, baselines)
        
        # Multiply by the input
        return inputs * integrated_gradients


class IntegratedGradientsXSign(IntegratedGradients):
    """Integrated Gradients times Sign attribution method."""
    
    def __init__(self, model, steps: int = 50, baseline_type: str = "zero", mu: float = 0.0):
        """Initialize with a PyTorch model.
        
        Args:
            model: PyTorch model for which to calculate gradients
            steps: Number of interpolation steps (default 50)
            baseline_type: Type of baseline to use (default "zero")
            mu: Threshold for sign determination (default 0.0)
        """
        super().__init__(model, steps, baseline_type)
        self.mu = mu
    
    def attribute(self, inputs: torch.Tensor, target: Optional[Union[int, torch.Tensor]] = None, 
                 baseline: Optional[torch.Tensor] = None, steps: Optional[int] = None,
                 baselines: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate integrated gradients times sign attribution.
        
        Args:
            inputs: Input tensor
            target: Target class index or tensor (None uses argmax)
            baseline: Baseline tensor (if None, created based on baseline_type)
            baselines: Alternative spelling for baseline (for compatibility)
            steps: Number of interpolation steps (if None, use self.steps)
            
        Returns:
            Attribution tensor of the same shape as inputs
        """
        # Get integrated gradients
        integrated_gradients = super().attribute(inputs, target, baseline, steps, baselines)
        
        # Generate sign map based on input values and mu threshold
        sign_map = torch.where(
            inputs < self.mu,
            torch.tensor(-1.0, device=inputs.device),
            torch.tensor(1.0, device=inputs.device)
        )
        
        # Multiply by the sign map
        return integrated_gradients * sign_map


# Functional API for compatibility
def integrated_gradients(model, inputs, target=None, baselines=None, steps=50):
    """Calculate Integrated Gradients attribution (functional API).
    
    Args:
        model: PyTorch model
        inputs: Input tensor
        target: Target class index (None for argmax)
        baselines: Baseline tensor (if None, created with zeros)
        steps: Number of integration steps
        
    Returns:
        Attribution tensor of the same shape as inputs
    """
    # Create IntegratedGradients instance and calculate attribution
    ig = IntegratedGradients(model, steps=steps)
    return ig.attribute(inputs, target=target, baselines=baselines, steps=steps)