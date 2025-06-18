"""Tests for Integrated Gradients implementation."""

import pytest
import torch
import torch.nn as nn

from signxai.torch_signxai.methods.integrated import IntegratedGradients


class SimpleNet(nn.Module):
    """Simple network for testing."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


@pytest.fixture
def model():
    """Create a simple model for testing."""
    return SimpleNet()


@pytest.fixture
def input_tensor():
    """Create a test input tensor."""
    return torch.randn(1, 3, 32, 32)


def test_integrated_gradients_initialization(model):
    """Test integrated gradients initialization."""
    ig = IntegratedGradients(model)
    assert isinstance(ig.model, nn.Module)
    assert ig.steps == 50
    assert ig.baseline is None


def test_integrated_gradients_attribution(model, input_tensor):
    """Test basic integrated gradients attribution."""
    ig = IntegratedGradients(model)
    attribution = ig.attribute(input_tensor)

    # Check shape matches input
    assert attribution.shape == input_tensor.shape

    # Check attributions are not all zero
    assert not torch.allclose(attribution, torch.zeros_like(attribution))


def test_integrated_gradients_with_target(model, input_tensor):
    """Test integrated gradients with specific target."""
    ig = IntegratedGradients(model)
    target = torch.tensor([5])  # Choose class 5
    attribution = ig.attribute(input_tensor, target=target)

    assert attribution.shape == input_tensor.shape
    assert not torch.allclose(attribution, torch.zeros_like(attribution))


def test_integrated_gradients_with_baseline(model, input_tensor):
    """Test integrated gradients with custom baseline."""
    baseline = torch.ones_like(input_tensor) * 0.5
    ig = IntegratedGradients(model, baseline=baseline)
    attribution = ig.attribute(input_tensor)

    assert attribution.shape == input_tensor.shape
    assert not torch.allclose(attribution, torch.zeros_like(attribution))


def test_integrated_gradients_steps(model, input_tensor):
    """Test different numbers of integration steps."""
    steps = [10, 50]
    attributions = []

    for n_steps in steps:
        ig = IntegratedGradients(model, steps=n_steps)
        attribution = ig.attribute(input_tensor)
        attributions.append(attribution)

    # Check that more steps give different (presumably more precise) results
    assert not torch.allclose(attributions[0], attributions[1])


def test_integrated_gradients_batch_processing(model):
    """Test integrated gradients with batch input."""
    batch_size = 4
    inputs = torch.randn(batch_size, 3, 32, 32)
    ig = IntegratedGradients(model)
    attribution = ig.attribute(inputs)

    assert attribution.shape == inputs.shape
    assert not torch.allclose(attribution, torch.zeros_like(attribution))


def test_numerical_stability(model):
    """Test numerical stability with edge cases."""
    small_input = torch.randn(1, 3, 32, 32) * 1e-5
    ig = IntegratedGradients(model)
    small_attr = ig.attribute(small_input)
    assert not torch.isnan(small_attr).any()


def test_error_handling(model, input_tensor):
    """Test error handling with invalid inputs."""
    ig = IntegratedGradients(model)

    # Test with invalid target class
    with pytest.raises(Exception):
        ig.attribute(input_tensor, target=torch.tensor([100]))  # Invalid class

    # Test with invalid target shape
    with pytest.raises(Exception):
        ig.attribute(input_tensor, target=torch.tensor([[1, 2]]))  # Wrong shape

    # Test with incompatible baseline shape
    with pytest.raises(Exception):
        ig = IntegratedGradients(model, baseline=torch.zeros(2, 3, 32, 32))
        ig.attribute(input_tensor)


def test_calculate_relevancemap_integrated_gradients(model, input_tensor):
    """Test calculate_relevancemap with integrated gradients."""
    from signxai.torch_signxai.methods import calculate_relevancemap
    
    result = calculate_relevancemap(
        "integrated_gradients",
        input_tensor,
        model,
        steps=10,
        baseline=torch.zeros_like(input_tensor),
        debug=False
    )
    assert isinstance(result, torch.Tensor)
    assert result.shape == input_tensor.shape
    
    # Test without specifying baseline
    result = calculate_relevancemap(
        "integrated_gradients",
        input_tensor,
        model,
        steps=10,
        debug=False
    )
    assert isinstance(result, torch.Tensor)
    assert result.shape == input_tensor.shape