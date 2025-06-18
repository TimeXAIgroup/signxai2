"""Tests for TensorFlow compatibility of gradient-based methods."""

import torch
import numpy as np
import pytest

from signxai.torch_signxai.methods.base import BaseGradient, InputXGradient, GradientXSign


class SimpleModel(torch.nn.Module):
    """Simple model for testing purposes."""
    
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(16 * 4 * 4, 10)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def test_base_gradient_clones_input():
    """Test that BaseGradient clones input tensor."""
    # Create model and inputs
    model = SimpleModel()
    inputs = torch.randn(2, 3, 8, 8, requires_grad=False)
    original_inputs = inputs.clone()
    
    # Compute gradients
    grad = BaseGradient(model).attribute(inputs)
    
    # Check that original input tensor is unchanged
    assert torch.all(torch.eq(inputs, original_inputs))
    
    # Check that thresholding was applied
    assert torch.sum(torch.abs(grad) < 1e-10) == 0


def test_input_x_gradient_uses_original_input():
    """Test that InputXGradient uses the original input values."""
    # Create model and inputs
    model = SimpleModel()
    inputs = torch.randn(2, 3, 8, 8, requires_grad=False)
    original_inputs = inputs.clone()
    
    # Compute attribution
    attribution = InputXGradient(model).attribute(inputs)
    
    # Check that original input tensor is unchanged
    assert torch.all(torch.eq(inputs, original_inputs))
    
    # Check that gradients have small values thresholded
    gradients = attribution / (inputs + 1e-10)  # Extract gradients
    assert torch.sum(torch.abs(gradients) < 1e-10) == 0


def test_gradient_x_sign_uses_original_input():
    """Test that GradientXSign uses the original input values for sign computation."""
    # Create model and inputs
    model = SimpleModel()
    inputs = torch.randn(2, 3, 8, 8, requires_grad=False)
    original_inputs = inputs.clone()
    
    # Set mu threshold in middle of input range
    mu = inputs.mean().item()
    
    # Compute attribution
    attribution = GradientXSign(model, mu=mu).attribute(inputs)
    
    # Check that original input tensor is unchanged
    assert torch.all(torch.eq(inputs, original_inputs))
    
    # Verify sign map was correctly applied
    expected_sign = torch.where(inputs < mu, torch.tensor(-1.0), torch.tensor(1.0))
    sign_used = torch.sign(attribution / (BaseGradient(model).attribute(inputs) + 1e-10))
    
    # Check if the signs match (allowing for some numerical differences at the threshold)
    sign_match_ratio = (sign_used == expected_sign).float().mean().item()
    assert sign_match_ratio > 0.95  # At least 95% of signs should match


def test_gradient_with_target_handling():
    """Test that target handling matches TensorFlow's behavior."""
    # Create model and inputs
    model = SimpleModel()
    inputs = torch.randn(2, 3, 8, 8, requires_grad=False)
    
    # Test scalar target
    scalar_target = torch.tensor(3)
    grad1 = BaseGradient(model).attribute(inputs, target=scalar_target)
    
    # Test int target
    int_target = 3
    grad2 = BaseGradient(model).attribute(inputs, target=int_target)
    
    # Both should give identical results
    assert torch.allclose(grad1, grad2)
    
    # Test different types of tensor targets
    tensor_target = torch.tensor([3, 2])
    grad3 = BaseGradient(model).attribute(inputs, target=tensor_target)
    
    # Should not raise an error and produce valid gradients
    assert grad3 is not None
    assert not torch.isnan(grad3).any()