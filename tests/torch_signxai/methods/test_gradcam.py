"""Tests for GradCAM implementation."""

import pytest
import torch
import torch.nn as nn

from signxai.torch_signxai.methods.gradcam import GradCAM

class SimpleConvNet(nn.Module):
    """Simple CNN for testing."""
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
        features = x
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

@pytest.fixture
def model():
    """Create a simple model for testing."""
    return SimpleConvNet()

@pytest.fixture
def input_tensor():
    """Create a test input tensor."""
    return torch.randn(1, 3, 32, 32)

def test_gradcam_initialization(model):
    """Test GradCAM initialization."""
    # Test automatic target layer detection
    gradcam = GradCAM(model)
    assert isinstance(gradcam.target_layer, nn.Conv2d)

    # Test manual target layer specification
    gradcam = GradCAM(model, target_layer=model.conv1)
    assert gradcam.target_layer == model.conv1

def test_gradcam_attribution(model, input_tensor):
    """Test basic GradCAM attribution."""
    gradcam = GradCAM(model)
    attribution = gradcam.attribute(input_tensor)

    # Check shape
    assert attribution.shape == (1, input_tensor.shape[2], input_tensor.shape[3])

    # Check range [0, 1]
    assert torch.all(attribution >= 0)
    assert torch.all(attribution <= 1)

def test_gradcam_with_target(model, input_tensor):
    """Test GradCAM with specific target."""
    gradcam = GradCAM(model)
    target = torch.tensor([5])  # Choose class 5
    attribution = gradcam.attribute(input_tensor, target=target)

    assert attribution.shape == (1, input_tensor.shape[2], input_tensor.shape[3])
    assert not torch.allclose(attribution, torch.zeros_like(attribution))

def test_gradcam_batch_processing(model):
    """Test GradCAM with batch input."""
    batch_size = 4
    inputs = torch.randn(batch_size, 3, 32, 32)

    gradcam = GradCAM(model)
    attribution = gradcam.attribute(inputs)

    assert attribution.shape == (batch_size, inputs.shape[2], inputs.shape[3])

def test_gradcam_interpolation(model, input_tensor):
    """Test GradCAM with size interpolation."""
    gradcam = GradCAM(model)
    target_size = (64, 64)
    attribution = gradcam.attribute(input_tensor, interpolate_size=target_size)

    assert attribution.shape == (1, *target_size)

def test_gradcam_integrated(model, input_tensor):
    """Test integrated GradCAM."""
    gradcam = GradCAM(model)
    attribution = gradcam.attribute_with_integrated_gradients(
        input_tensor,
        steps=10
    )

    assert attribution.shape == (1, input_tensor.shape[2], input_tensor.shape[3])
    assert not torch.allclose(attribution, torch.zeros_like(attribution))

def test_invalid_target_layer():
    """Test handling of invalid target layer."""
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 2)
    )

    with pytest.raises(ValueError):
        GradCAM(model)


def test_calculate_relevancemap_gradcam(model, input_tensor):
    """Test GradCAM with the calculate_relevancemap interface."""
    from signxai.torch_signxai.methods import calculate_relevancemap
    
    # Test with explicit layer specification
    result = calculate_relevancemap("grad_cam", input_tensor, model, last_conv_layer=model.conv2, debug=False)
    assert isinstance(result, torch.Tensor)
    assert result.shape == input_tensor.shape
    
    # Test with string layer specification
    result = calculate_relevancemap("grad_cam", input_tensor, model, last_conv="conv2", debug=False)
    assert isinstance(result, torch.Tensor)
    assert result.shape == input_tensor.shape
    
    # Test without layer specification should raise error
    with pytest.raises(ValueError):
        calculate_relevancemap("grad_cam", input_tensor, model, debug=False)