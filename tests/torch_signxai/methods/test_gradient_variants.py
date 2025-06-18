"""Tests for the gradient-based methods and their variants."""

import torch
import numpy as np
import pytest

from signxai.torch_signxai.methods import (
    BaseGradient,
    InputXGradient,
    GradientXSign,
    SmoothGrad,
    SmoothGradXInput,
    SmoothGradXSign,
    IntegratedGradients,
    IntegratedGradientsXInput,
    IntegratedGradientsXSign,
    VarGrad,
    VarGradXInput,
    VarGradXSign,
)


class SimpleModel(torch.nn.Module):
    """Simple model for testing purposes."""
    
    def __init__(self, input_dim=2, hidden_dim=10, output_dim=3):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class SimpleConvModel(torch.nn.Module):
    """Simple convolutional model for testing."""
    
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


@pytest.fixture
def simple_model():
    """Create a simple model instance."""
    return SimpleModel()


@pytest.fixture
def conv_model():
    """Create a simple convolutional model instance."""
    return SimpleConvModel()


@pytest.fixture
def simple_input():
    """Create a simple input tensor."""
    return torch.randn(4, 2)


@pytest.fixture
def image_input():
    """Create a simple image input tensor."""
    return torch.randn(2, 3, 8, 8)


def test_base_gradient(simple_model, simple_input):
    """Test BaseGradient attribution."""
    gradient = BaseGradient(simple_model)
    attribution = gradient.attribute(simple_input)
    
    # Check shape
    assert attribution.shape == simple_input.shape
    
    # Check that gradients are computed
    assert not torch.allclose(attribution, torch.zeros_like(attribution))


def test_input_times_gradient(simple_model, simple_input):
    """Test InputXGradient attribution."""
    input_grad = InputXGradient(simple_model)
    attribution = input_grad.attribute(simple_input)
    
    # Check shape
    assert attribution.shape == simple_input.shape
    
    # Check that attribution equals input * base_gradient
    base_grad = BaseGradient(simple_model).attribute(simple_input)
    expected = simple_input * base_grad
    
    assert torch.allclose(attribution, expected)


def test_gradient_times_sign(simple_model, simple_input):
    """Test GradientXSign attribution."""
    # Test with default mu = 0.0
    grad_sign = GradientXSign(simple_model)
    attribution = grad_sign.attribute(simple_input)
    
    # Check shape
    assert attribution.shape == simple_input.shape
    
    # Check that attribution equals base_gradient * sign(input)
    base_grad = BaseGradient(simple_model).attribute(simple_input)
    expected_sign = torch.sign(simple_input)
    expected = base_grad * expected_sign
    
    assert torch.allclose(attribution, expected)


def test_gradient_times_sign_with_mu(simple_model, simple_input):
    """Test GradientXSign attribution with custom mu value."""
    mu = 0.5
    grad_sign = GradientXSign(simple_model, mu=mu)
    attribution = grad_sign.attribute(simple_input)
    
    # Check shape
    assert attribution.shape == simple_input.shape
    
    # Check that attribution equals base_gradient * sign(input with threshold)
    base_grad = BaseGradient(simple_model).attribute(simple_input)
    expected_sign = torch.where(
        simple_input < mu,
        torch.tensor(-1.0),
        torch.tensor(1.0)
    )
    expected = base_grad * expected_sign
    
    assert torch.allclose(attribution, expected)


def test_smoothgrad(simple_model, simple_input):
    """Test SmoothGrad attribution."""
    # Use small number of samples for testing speed
    smooth_grad = SmoothGrad(simple_model, num_samples=3)
    attribution = smooth_grad.attribute(simple_input)
    
    # Check shape
    assert attribution.shape == simple_input.shape
    
    # Check that attributions are not all zero
    assert not torch.allclose(attribution, torch.zeros_like(attribution))


def test_smoothgrad_times_input(simple_model, simple_input):
    """Test SmoothGradXInput attribution."""
    # Use small number of samples for testing speed
    smooth_grad_input = SmoothGradXInput(simple_model, num_samples=3)
    attribution = smooth_grad_input.attribute(simple_input)
    
    # Check shape
    assert attribution.shape == simple_input.shape
    
    # Check that attributions are not all zero
    assert not torch.allclose(attribution, torch.zeros_like(attribution))
    
    # Verify that attribution is smoothgrad * input
    smoothgrad_attr = SmoothGrad(simple_model, num_samples=3).attribute(simple_input)
    expected = simple_input * smoothgrad_attr
    
    assert torch.allclose(attribution, expected)


def test_smoothgrad_times_sign(simple_model, simple_input):
    """Test SmoothGradXSign attribution."""
    # Use small number of samples for testing speed
    mu = 0.0
    smooth_grad_sign = SmoothGradXSign(simple_model, num_samples=3, mu=mu)
    attribution = smooth_grad_sign.attribute(simple_input)
    
    # Check shape
    assert attribution.shape == simple_input.shape
    
    # Check that attributions are not all zero
    assert not torch.allclose(attribution, torch.zeros_like(attribution))
    
    # Verify that attribution is smoothgrad * sign(input)
    smoothgrad_attr = SmoothGrad(simple_model, num_samples=3).attribute(simple_input)
    expected_sign = torch.sign(simple_input)
    expected = smoothgrad_attr * expected_sign
    
    assert torch.allclose(attribution, expected)


def test_integrated_gradients(simple_model, simple_input):
    """Test IntegratedGradients attribution."""
    # Use small number of steps for testing speed
    integrated_grad = IntegratedGradients(simple_model, steps=5)
    attribution = integrated_grad.attribute(simple_input)
    
    # Check shape
    assert attribution.shape == simple_input.shape
    
    # Check that attributions are not all zero
    assert not torch.allclose(attribution, torch.zeros_like(attribution))


def test_integrated_gradients_times_input(simple_model, simple_input):
    """Test IntegratedGradientsXInput attribution."""
    # Use small number of steps for testing speed
    integrated_grad_input = IntegratedGradientsXInput(simple_model, steps=5)
    attribution = integrated_grad_input.attribute(simple_input)
    
    # Check shape
    assert attribution.shape == simple_input.shape
    
    # Verify that attribution is integrated_gradients * input
    ig_attr = IntegratedGradients(simple_model, steps=5).attribute(simple_input)
    expected = simple_input * ig_attr
    
    assert torch.allclose(attribution, expected)


def test_integrated_gradients_times_sign(simple_model, simple_input):
    """Test IntegratedGradientsXSign attribution."""
    # Use small number of steps for testing speed
    mu = 0.0
    integrated_grad_sign = IntegratedGradientsXSign(simple_model, steps=5, mu=mu)
    attribution = integrated_grad_sign.attribute(simple_input)
    
    # Check shape
    assert attribution.shape == simple_input.shape
    
    # Verify that attribution is integrated_gradients * sign(input)
    ig_attr = IntegratedGradients(simple_model, steps=5).attribute(simple_input)
    expected_sign = torch.sign(simple_input)
    expected = ig_attr * expected_sign
    
    assert torch.allclose(attribution, expected)


def test_vargrad(simple_model, simple_input):
    """Test VarGrad attribution."""
    # Use small number of samples for testing speed
    var_grad = VarGrad(simple_model, num_samples=3)
    attribution = var_grad.attribute(simple_input)
    
    # Check shape
    assert attribution.shape == simple_input.shape
    
    # Check that attributions are not all zero and are non-negative (variance is always >= 0)
    assert not torch.allclose(attribution, torch.zeros_like(attribution))
    assert torch.all(attribution >= 0)


def test_vargrad_times_input(simple_model, simple_input):
    """Test VarGradXInput attribution."""
    # Use small number of samples for testing speed
    var_grad_input = VarGradXInput(simple_model, num_samples=3)
    attribution = var_grad_input.attribute(simple_input)
    
    # Check shape
    assert attribution.shape == simple_input.shape
    
    # Verify that attribution is vargrad * input
    vargrad_attr = VarGrad(simple_model, num_samples=3).attribute(simple_input)
    expected = simple_input * vargrad_attr
    
    assert torch.allclose(attribution, expected)


def test_vargrad_times_sign(simple_model, simple_input):
    """Test VarGradXSign attribution."""
    # Use small number of samples for testing speed
    mu = 0.0
    var_grad_sign = VarGradXSign(simple_model, num_samples=3, mu=mu)
    attribution = var_grad_sign.attribute(simple_input)
    
    # Check shape
    assert attribution.shape == simple_input.shape
    
    # Verify that attribution is vargrad * sign(input)
    vargrad_attr = VarGrad(simple_model, num_samples=3).attribute(simple_input)
    expected_sign = torch.sign(simple_input)
    expected = vargrad_attr * expected_sign
    
    assert torch.allclose(attribution, expected)


def test_convolutional_model_compatibility(conv_model, image_input):
    """Test compatibility with convolutional models."""
    # Test all methods with a convolutional model
    methods = [
        BaseGradient(conv_model),
        InputXGradient(conv_model),
        GradientXSign(conv_model, mu=0.5),
        SmoothGrad(conv_model, num_samples=2),
        SmoothGradXInput(conv_model, num_samples=2),
        SmoothGradXSign(conv_model, num_samples=2, mu=0.5),
        IntegratedGradients(conv_model, steps=2),
        IntegratedGradientsXInput(conv_model, steps=2),
        IntegratedGradientsXSign(conv_model, steps=2, mu=0.5),
        VarGrad(conv_model, num_samples=2),
        VarGradXInput(conv_model, num_samples=2),
        VarGradXSign(conv_model, num_samples=2, mu=0.5),
    ]
    
    for method in methods:
        attribution = method.attribute(image_input)
        # Check shape
        assert attribution.shape == image_input.shape
        # Check that attributions are computed
        assert not torch.allclose(attribution, torch.zeros_like(attribution))


def test_target_specification(conv_model, image_input):
    """Test different ways to specify target classes."""
    # Different ways to specify targets
    targets = [
        None,  # Use argmax
        0,  # Integer target
        torch.tensor(0),  # Scalar tensor
        torch.tensor([0, 1]),  # Tensor with batch size
    ]
    
    methods = [
        BaseGradient(conv_model),
        InputXGradient(conv_model),
        GradientXSign(conv_model),
        SmoothGrad(conv_model, num_samples=2),
        IntegratedGradients(conv_model, steps=2),
        VarGrad(conv_model, num_samples=2),
    ]
    
    for method in methods:
        for target in targets:
            # Should not raise errors
            attribution = method.attribute(image_input, target=target)
            assert attribution.shape == image_input.shape