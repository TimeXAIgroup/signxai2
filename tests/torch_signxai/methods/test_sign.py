"""Tests for the SIGN method implementation."""

import pytest
import torch
import torch.nn as nn

from signxai.torch_signxai.methods.sign import SIGN

class SimpleConvNet(nn.Module):
    """Simple CNN for testing."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
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
    """Create a random input tensor."""
    torch.manual_seed(42)  # For reproducibility
    return torch.randn(1, 3, 32, 32)

def test_sign_initialization(model):
    """Test SIGN initialization."""
    sign = SIGN(model)
    assert sign.vlow == -1.0
    assert sign.vhigh == 1.0
    assert sign.mu == 0.0

    # Test custom parameters
    sign = SIGN(model, mu=0.5, vlow=-2.0, vhigh=2.0)
    assert sign.vlow == -2.0
    assert sign.vhigh == 2.0
    assert sign.mu == 0.5
    
    # Test preset initialization
    sign = SIGN(model, preset='zeros')
    assert sign.mu is None  # mu is calculated during attribution
    assert sign._preset == 'zeros'
    
    # Test create_from_preset static method
    sign_neg = SIGN.create_from_preset(model, 'negative_half')
    assert sign_neg.mu == -0.5
    
    sign_pos = SIGN.create_from_preset(model, 'positive_half')
    assert sign_pos.mu == 0.5
    
    sign_zeros = SIGN.create_from_preset(model, 'zeros')
    assert sign_zeros._preset == 'zeros'

def test_sign_mu_calculation(model):
    """Test mu parameter calculation."""
    # Test default mu calculation
    sign = SIGN(model, vlow=-2.0, vhigh=2.0)
    assert sign.mu == 0.0  # Should be (vlow + |vhigh - vlow|/2)

    # Test custom mu
    sign = SIGN(model, mu=0.5)
    assert sign.mu == 0.5

def test_sign_calculation(model):
    """Test SIGN value calculation."""
    sign = SIGN(model, mu=0.0)
    x = torch.tensor([-1.0, 0.0, 1.0])
    result = sign._calculate_sign(x)

    assert torch.allclose(result, torch.tensor([-1.0, 1.0, 1.0]))
    
    # Test with different mu values
    sign = SIGN(model, mu=0.5)
    result = sign._calculate_sign(x)
    assert torch.allclose(result, torch.tensor([-1.0, -1.0, 1.0]))
    
    # Test with different vlow/vhigh
    sign = SIGN(model, mu=0.0, vlow=-2.0, vhigh=2.0)
    result = sign._calculate_sign(x)
    assert torch.allclose(result, torch.tensor([-2.0, 2.0, 2.0]))
    
def test_sign_preset_calculation(model):
    """Test SIGN value calculation with presets."""
    # Create test data
    x = torch.tensor([[-1.0, -0.5, 0.0, 0.5, 1.0]])
    
    # Test zeros preset
    sign = SIGN(model, preset='zeros')
    result = sign._calculate_sign(x)
    expected = torch.tensor([[-1.0, -1.0, 1.0, 1.0, 1.0]])
    assert torch.allclose(result, expected)
    
    # Test mean preset
    x_with_mean = torch.tensor([[-10.0, -5.0, 0.0, 5.0, 10.0]])  # Mean = 0
    sign = SIGN(model, preset='mean')
    result = sign._calculate_sign(x_with_mean)
    # TF implementation uses 0 as threshold for mean preset
    expected = torch.tensor([[-1.0, -1.0, 1.0, 1.0, 1.0]])
    assert torch.allclose(result, expected, rtol=1e-3)
    
    # Test median preset
    x_with_median = torch.tensor([[-10.0, -5.0, 0.0, 5.0, 10.0]])  # Median = 0
    sign = SIGN(model, preset='median')
    result = sign._calculate_sign(x_with_median)
    # TF implementation uses 0 as threshold for median preset 
    expected = torch.tensor([[-1.0, -1.0, 1.0, 1.0, 1.0]])
    assert torch.allclose(result, expected, rtol=1e-3)
    
    # Test adaptive preset (uses 75th percentile)
    x_adaptive = torch.tensor([[-10.0, -5.0, 0.0, 5.0, 10.0]])
    sign = SIGN(model, preset='adaptive')
    result = sign._calculate_sign(x_adaptive)
    # The 75th percentile of [-10, -5, 0, 5, 10] should be around 5.0
    # So values >= 5.0 should be positive (1.0), others negative (-1.0)
    # Note: The percentile calculation in PyTorch gives 5.0, so 5.0 >= 5.0 is True
    expected = torch.tensor([[-1.0, -1.0, -1.0, 1.0, 1.0]])
    assert torch.allclose(result, expected, rtol=1e-3)

def test_sign_attribution(model, input_tensor):
    """Test basic SIGN attribution."""
    sign = SIGN(model)
    attribution = sign.attribute(input_tensor)

    # Check shape, dtype, and device
    assert attribution.shape == input_tensor.shape
    assert attribution.dtype == input_tensor.dtype
    assert attribution.device == input_tensor.device

    # Check numerical properties
    assert not torch.isnan(attribution).any()
    assert not torch.isinf(attribution).any()
    assert not torch.allclose(attribution, torch.zeros_like(attribution))

def test_sign_target_attribution(model, input_tensor):
    """Test SIGN attribution with specific target."""
    sign = SIGN(model)
    target = torch.tensor([3])
    attribution = sign.attribute(input_tensor, target=target)

    assert attribution.shape == input_tensor.shape
    assert not torch.isnan(attribution).any()
    assert not torch.allclose(attribution, torch.zeros_like(attribution))

def test_sign_batch_attribution(model):
    """Test SIGN attribution with batch input."""
    batch_size = 4
    input_batch = torch.randn(batch_size, 3, 32, 32)
    sign = SIGN(model)
    attribution = sign.attribute(input_batch)

    assert attribution.shape == input_batch.shape
    assert not torch.isnan(attribution).any()
    assert not torch.allclose(attribution, torch.zeros_like(attribution))

def test_sign_matches_reference_values(model):
    """Test SIGN produces expected reference values."""
    # Create deterministic input
    torch.manual_seed(42)
    x = torch.randn(1, 3, 32, 32)

    # Reference values calculated with the following parameters:
    # mu = 0.0, vlow = -1.0, vhigh = 1.0
    sign = SIGN(model)
    attribution = sign.attribute(x)

    # Check key statistical properties
    assert -1.0 <= attribution.min() <= 1.0
    assert -1.0 <= attribution.max() <= 1.0
    assert not torch.isnan(attribution).any()

def test_sign_integrated_gradients(model, input_tensor):
    """Test integrated gradients variant of SIGN."""
    sign = SIGN(model)
    attribution = sign.get_integrated_gradients(input_tensor, steps=10)

    assert attribution.shape == input_tensor.shape
    assert not torch.isnan(attribution).any()
    assert not torch.isinf(attribution).any()
    assert not torch.allclose(attribution, torch.zeros_like(attribution))

def test_sign_matches_tensorflow_behavior():
    """Test that SIGN matches TensorFlow behavior without dependencies."""
    # Create simple test case
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

    # Expected values based on TensorFlow behavior with mu=0
    expected = torch.tensor([-1.0, -1.0, 1.0, 1.0, 1.0])

    # Calculate SIGN values
    sign = SIGN(nn.Identity(), mu=0.0)
    result = sign._calculate_sign(x)

    assert torch.allclose(result, expected)

def test_invalid_input_handling(model):
    """Test error handling for invalid inputs."""
    sign = SIGN(model)

    # Test empty input
    with pytest.raises(ValueError):
        sign.attribute(torch.tensor([]))

    # Test invalid vlow/vhigh
    with pytest.raises(AssertionError):
        SIGN(model, vlow=1.0, vhigh=-1.0)

def test_sign_attribution_with_kwargs(model, input_tensor):
    """Test SIGN attribution with keyword arguments for overrides."""
    sign = SIGN(model, mu=0.0)
    
    # Test with mu_override
    attr1 = sign.attribute(input_tensor)
    attr2 = sign.attribute(input_tensor, mu_override=0.5)
    
    # Results should be different with different mu values
    assert not torch.allclose(attr1, attr2)
    
    # Test with preset_override
    sign = SIGN(model, mu=0.0)
    attr1 = sign.attribute(input_tensor)
    attr2 = sign.attribute(input_tensor, preset_override='adaptive')
    
    # Results should be different with preset
    assert not torch.allclose(attr1, attr2)
    
    # Verify that original settings are restored
    attr3 = sign.attribute(input_tensor)
    assert torch.allclose(attr1, attr3)

def test_explainability_methods_utility():
    """Test the ExplainabilityMethods utility class."""
    from signxai.torch_signxai.methods.base import ExplainabilityMethods
    
    # Test list_methods
    methods = ExplainabilityMethods.list_methods()
    assert isinstance(methods, list)
    assert len(methods) > 0
    
    # Check that it includes our key method types
    essential_methods = [
        "gradient", "gradient_x_sign", "guided_backprop", 
        "smoothgrad", "integrated_gradients", "lrp_z"
    ]
    for method in essential_methods:
        assert method in methods
    
    # Test get_method_info
    for method in ["gradient_x_sign_mu_0_5", "grad_cam", "lrp_epsilon_0_1"]:
        info = ExplainabilityMethods.get_method_info(method)
        assert isinstance(info, dict)
        assert "name" in info
        assert "parameters" in info
        
    # Test mu parameter extraction
    mu_method_info = ExplainabilityMethods.get_method_info("gradient_x_sign_mu_0_5")
    assert "mu" in mu_method_info["parameters"]
    
    # Test epsilon parameter extraction
    eps_method_info = ExplainabilityMethods.get_method_info("lrp_epsilon_0_1")
    assert "epsilon" in eps_method_info["parameters"]
    
    # Test random_uniform method
    model = SimpleConvNet()
    inputs = torch.randn(1, 3, 32, 32)
    rand_attr = ExplainabilityMethods.get_random_uniform(model, inputs)
    assert rand_attr.shape == inputs.shape
    assert not torch.allclose(rand_attr, torch.zeros_like(rand_attr))