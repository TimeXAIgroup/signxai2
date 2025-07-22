#!/usr/bin/env python3
"""TF-exact VarGrad implementation to match TensorFlow iNNvestigate behavior exactly."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union


class TFExactVarGradAnalyzer:
    """TF-exact VarGrad analyzer that matches TensorFlow iNNvestigate implementation exactly."""
    
    def __init__(self, model: nn.Module, noise_scale: float = 0.2, augment_by_n: int = 50):
        """Initialize TF-exact VarGrad analyzer.
        
        Args:
            model: PyTorch model
            noise_scale: Standard deviation of noise to add (TF default: 0.2)
            augment_by_n: Number of noisy samples to generate (TF default: 50)
        """
        self.model = model
        self.noise_scale = noise_scale
        self.augment_by_n = augment_by_n
    
    def analyze(self, input_tensor: torch.Tensor, target_class: Optional[Union[int, torch.Tensor]] = None, **kwargs) -> np.ndarray:
        """Analyze input using TF-exact VarGrad algorithm.
        
        Args:
            input_tensor: Input tensor to analyze
            target_class: Target class index
            **kwargs: Additional arguments (ignored)
            
        Returns:
            VarGrad attribution as numpy array
        """
        # Override parameters from kwargs if provided
        # Handle both TF parameter names and PT comparison script parameter names
        noise_scale = kwargs.get('noise_scale', kwargs.get('noise_level', self.noise_scale))
        augment_by_n = kwargs.get('augment_by_n', kwargs.get('num_samples', self.augment_by_n))
        
        # Ensure model is in eval mode
        original_mode = self.model.training
        self.model.eval()
        
        # Convert input to tensor if needed
        if not isinstance(input_tensor, torch.Tensor):
            input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
            
        # Add batch dimension if needed
        needs_batch_dim = input_tensor.ndim == 3
        if needs_batch_dim:
            input_tensor = input_tensor.unsqueeze(0)
        
        # Get target class if not provided
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()
        elif isinstance(target_class, torch.Tensor):
            target_class = target_class.item()
        
        # Create multiple noisy samples and compute gradients
        all_gradients = []
        
        for _ in range(augment_by_n):
            # Create noisy input exactly as TensorFlow does:
            # TF: noise = np.random.normal(0, self._noise_scale, np.shape(x))
            # where noise_scale = 0.2 by default and is absolute (not relative to input range)
            noise = torch.normal(mean=0.0, std=noise_scale, size=input_tensor.shape, device=input_tensor.device, dtype=input_tensor.dtype)
            noisy_input = input_tensor + noise
            noisy_input = noisy_input.clone().detach().requires_grad_(True)
            
            # Compute gradient for this noisy sample
            self.model.zero_grad()
            output = self.model(noisy_input)
            
            # Create target tensor for backpropagation
            if isinstance(target_class, int):
                target_tensor = torch.zeros_like(output)
                target_tensor[0, target_class] = 1.0
            else:
                target_tensor = target_class
            
            # Compute gradient
            output.backward(gradient=target_tensor)
            
            if noisy_input.grad is not None:
                all_gradients.append(noisy_input.grad.clone().detach())
            else:
                print("Warning: VarGrad - gradient is None for one sample")
                all_gradients.append(torch.zeros_like(noisy_input))
        
        # Restore original training mode
        self.model.train(original_mode)
        
        if not all_gradients:
            print("Warning: VarGrad - no gradients collected, returning zeros")
            result = torch.zeros_like(input_tensor)
            if needs_batch_dim:
                result = result.squeeze(0)
            return result.cpu().numpy()
        
        # Stack gradients: shape (num_samples, batch, channels, height, width)
        grad_stack = torch.stack(all_gradients, dim=0)
        
        # Compute variance exactly as TensorFlow VariationalAugmentReduceBase does:
        # Code from TF: 
        # gk = X[key]  # shape: (num_samples, ...)
        # mn_gk = np.mean(gk, axis=0)  # mean across samples
        # inner = (gk - mn_gk) ** 2   # squared differences
        # means[key] = np.mean(inner, axis=0)  # mean of squared differences
        
        # 1. Compute mean across samples (axis=0 in TF)
        mean_grad = torch.mean(grad_stack, dim=0)  # Remove samples dimension
        
        # 2. Compute squared differences from mean
        variance_terms = (grad_stack - mean_grad.unsqueeze(0)) ** 2
        
        # 3. Take mean of squared differences across samples (axis=0 in TF)
        variance_grad = torch.mean(variance_terms, dim=0)
        
        # Remove batch dimension if it was added
        if needs_batch_dim:
            variance_grad = variance_grad.squeeze(0)
        
        # Convert to numpy
        result = variance_grad.cpu().numpy()
        
        return result


def create_tf_exact_vargrad_analyzer(model: nn.Module, **kwargs):
    """Create TF-exact VarGrad analyzer with TensorFlow-compatible parameters.
    
    Args:
        model: PyTorch model
        **kwargs: Additional arguments
        
    Returns:
        TFExactVarGradAnalyzer instance
    """
    # Use TensorFlow defaults
    noise_scale = kwargs.get('noise_scale', 0.2)  # TF default
    augment_by_n = kwargs.get('augment_by_n', 50)  # TF default
    
    return TFExactVarGradAnalyzer(
        model=model,
        noise_scale=noise_scale,
        augment_by_n=augment_by_n
    )