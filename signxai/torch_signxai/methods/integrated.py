"""PyTorch implementation of Integrated Gradients."""
import torch
import torch.nn as nn
import numpy as np
from typing import Union, Optional, Tuple, List


class IntegratedGradients:
    """Integrated Gradients attribution method.
    
    Implements the method described in "Axiomatic Attribution for Deep Networks"
    (https://arxiv.org/abs/1703.01365).
    """
    
    def __init__(self, model, steps=50, baseline_type="zero"):
        """Initialize Integrated Gradients.
        
        Args:
            model: PyTorch model
            steps: Number of integration steps (default 50)
            baseline_type: Type of baseline to use ("zero", "black", "white", or "gaussian")
        """
        self.model = model
        self.steps = steps
        self.baseline_type = baseline_type
        
    def _create_baseline(self, inputs):
        """Create baseline tensor based on baseline_type.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Baseline tensor of the same shape as inputs
        """
        if self.baseline_type == "zero" or self.baseline_type is None:
            return torch.zeros_like(inputs)
        elif self.baseline_type == "black":
            # Black image usually has minimal values
            return torch.zeros_like(inputs)
        elif self.baseline_type == "white":
            # White image usually has maximal values
            return torch.ones_like(inputs)
        elif self.baseline_type == "gaussian":
            # Small random noise
            return torch.randn_like(inputs) * 0.1
        else:
            raise ValueError(f"Unsupported baseline_type: {self.baseline_type}")
            
    def attribute(self, inputs, target=None, baselines=None, steps=None):
        """Calculate attribution using Integrated Gradients.
        
        Args:
            inputs: Input tensor
            target: Target class index (None for argmax)
            baselines: Baseline tensor (if None, created based on baseline_type)
            steps: Number of integration steps (if None, use self.steps)
            
        Returns:
            Attribution tensor of the same shape as inputs
        """
        # Use provided parameters or defaults
        steps = steps if steps is not None else self.steps
        baseline = baselines if baselines is not None else self._create_baseline(inputs)
        
        # Ensure input is a tensor
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float32)
            
        # Clone inputs and baseline to avoid modifying originals
        inputs = inputs.clone().detach()
        baseline = baseline.clone().detach().to(inputs.device, inputs.dtype)
        
        # Ensure baseline has the same shape as inputs
        if baseline.shape != inputs.shape:
            raise ValueError(f"Baseline shape {baseline.shape} must match inputs shape {inputs.shape}")
            
        # Original model mode
        original_mode = self.model.training
        self.model.eval()
        
        # Generate scaled inputs for each step
        scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(steps + 1)]
        
        # Accumulate gradients
        gradients = []
        
        for scaled_input in scaled_inputs:
            # Enable gradient tracking
            scaled_input_grad = scaled_input.clone().detach().requires_grad_(True)
            
            # Forward pass
            self.model.zero_grad()
            output = self.model(scaled_input_grad)
            
            # Determine target class
            if target is None:
                # Use argmax of the output
                target_indices = output.argmax(dim=1)
            elif isinstance(target, int):
                # Use the same target for all examples in the batch
                target_indices = torch.full((inputs.shape[0],), target, 
                                          dtype=torch.long, device=inputs.device)
            elif isinstance(target, torch.Tensor):
                if target.numel() == 1:
                    # Single scalar tensor target
                    target_indices = torch.full((inputs.shape[0],), target.item(), 
                                              dtype=torch.long, device=inputs.device)
                else:
                    # Tensor with multiple targets (one per batch item)
                    target_indices = target
            else:
                raise ValueError(f"Unsupported target type: {type(target)}")
                
            # Create one-hot encoding for target class(es)
            one_hot = torch.zeros_like(output)
            for i, idx in enumerate(target_indices):
                one_hot[i, idx] = 1.0
                
            # Backward pass
            output.backward(gradient=one_hot)
            
            # Store gradients
            if scaled_input_grad.grad is None:
                gradients.append(torch.zeros_like(inputs))
            else:
                gradients.append(scaled_input_grad.grad.clone())
                
        # Restore model mode
        self.model.train(original_mode)
        
        # Convert gradients to tensor
        gradients = torch.stack(gradients)
        
        # Compute trapezoidal approximation
        avg_gradients = (gradients[:-1] + gradients[1:]) / 2.0
        integrated_gradients = torch.mean(avg_gradients, dim=0) * (inputs - baseline)
        
        # Apply small value thresholding for numerical stability
        integrated_gradients[torch.abs(integrated_gradients) < 1e-10] = 0.0
        
        return integrated_gradients


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