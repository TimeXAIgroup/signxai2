"""Tests for verifying gradient method compatibility with TensorFlow."""

import torch
import numpy as np
import pytest

from signxai.torch_signxai.methods.base import BaseGradient, InputXGradient, GradientXSign


class SimpleModel(torch.nn.Module):
    """Simple model for testing gradient calculations."""
    
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


def test_gradient_tensor_handling():
    """Test that gradients are calculated correctly and input tensors are preserved."""
    # Create a simple model and input
    model = SimpleModel()
    inputs = torch.randn(2, 3, 8, 8)
    original_inputs = inputs.clone()
    
    # Calculate gradients
    gradient_calc = BaseGradient(model)
    grads = gradient_calc.attribute(inputs)
    
    # Verify input tensor wasn't modified (matches TensorFlow's behavior)
    assert torch.all(torch.eq(inputs, original_inputs))
    
    # Verify gradients were calculated and thresholded
    assert grads is not None
    assert not torch.isnan(grads).any()
    # Check that very small gradients were thresholded to zero
    assert torch.sum(torch.abs(grads) < 1e-10) == 0


def test_input_gradient_multiplication():
    """Test that InputXGradient correctly uses original inputs."""
    # Create a simple model and input
    model = SimpleModel()
    inputs = torch.randn(2, 3, 8, 8)
    original_inputs = inputs.clone()
    
    # Calculate InputXGradient attribution
    ig_calc = InputXGradient(model)
    attribution = ig_calc.attribute(inputs)
    
    # Verify input tensor wasn't modified
    assert torch.all(torch.eq(inputs, original_inputs))
    
    # Calculate regular gradients for comparison
    gradients = BaseGradient(model).attribute(inputs)
    
    # Verify attribution equals input * gradient (within numerical precision)
    expected = original_inputs * gradients
    assert torch.allclose(attribution, expected, rtol=1e-5, atol=1e-5)


def test_gradient_sign_thresholding():
    """Test that GradientXSign correctly applies thresholding on original inputs."""
    # Create a simple model and input
    model = SimpleModel()
    inputs = torch.randn(2, 3, 8, 8)
    original_inputs = inputs.clone()
    
    # Set mu threshold to the mean value of inputs
    mu = inputs.mean().item()
    
    # Calculate GradientXSign attribution
    sign_calc = GradientXSign(model, mu=mu)
    attribution = sign_calc.attribute(inputs)
    
    # Verify input tensor wasn't modified
    assert torch.all(torch.eq(inputs, original_inputs))
    
    # Calculate regular gradients
    gradients = BaseGradient(model).attribute(inputs)
    
    # Calculate expected sign map based on original inputs
    expected_sign = torch.where(
        original_inputs < mu,
        torch.tensor(-1.0, device=inputs.device),
        torch.tensor(1.0, device=inputs.device)
    )
    
    # Verify attribution equals gradient * sign_map
    expected = gradients * expected_sign
    assert torch.allclose(attribution, expected, rtol=1e-5, atol=1e-5)


def test_target_handling():
    """Test that target handling matches TensorFlow's behavior."""
    # Create a simple model and input
    model = SimpleModel()
    inputs = torch.randn(2, 3, 8, 8)
    
    # Test different ways to specify targets
    targets = [
        None,  # Uses argmax
        5,  # Integer target
        torch.tensor(5),  # Scalar tensor
        torch.tensor([5, 3]),  # Tensor with batch size
    ]
    
    # All these should calculate gradients without error
    for target in targets:
        gradients = BaseGradient(model).attribute(inputs, target=target)
        assert gradients is not None
        assert gradients.shape == inputs.shape
        
    # Test special case of broadcasting scalar tensor to batch
    scalar_target = torch.tensor(5)
    batch_target = torch.full((inputs.shape[0],), 5, dtype=torch.long)
    
    grad1 = BaseGradient(model).attribute(inputs, target=scalar_target)
    grad2 = BaseGradient(model).attribute(inputs, target=batch_target)
    
    # Both should give the same result (broadcasting works correctly)
    assert torch.allclose(grad1, grad2, rtol=1e-5, atol=1e-5)


def test_tensorflow_gradient_behavior():
    """Test specific behaviors that should match TensorFlow's gradient calculation."""
    # Create a simple model and input
    model = SimpleModel()
    inputs = torch.randn(2, 3, 8, 8)
    
    # 1. Test backward pass with retain_graph
    grad_calc = BaseGradient(model)
    grads1 = grad_calc.attribute(inputs, target=5)
    
    # Should be able to calculate gradients again without error
    # This matches TensorFlow's ability to reuse GradientTape
    grads2 = grad_calc.attribute(inputs, target=3)
    
    # Gradients should be different for different targets
    assert not torch.allclose(grads1, grads2)
    
    # 2. Test one-hot encoding matches TensorFlow's tf.one_hot approach
    batch_size = inputs.shape[0]
    target = torch.tensor([5, 3])
    
    # Manual approach with one_hot
    outputs = model(inputs)
    one_hot = torch.zeros_like(outputs)
    one_hot[torch.arange(batch_size), target] = 1.0
    
    # Set up for manual gradient calculation
    inputs_clone = inputs.clone().detach().requires_grad_(True)
    outputs_clone = model(inputs_clone)
    
    # Backward with one-hot
    outputs_clone.backward(gradient=one_hot)
    manual_grads = inputs_clone.grad
    manual_grads[torch.abs(manual_grads) < 1e-10] = 0.0
    
    # Compare with our implementation
    auto_grads = grad_calc.attribute(inputs, target=target)
    
    # Should match closely
    assert torch.allclose(manual_grads, auto_grads, rtol=1e-5, atol=1e-5)