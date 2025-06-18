"""Tests for Guided Backpropagation implementation."""

import pytest
import torch
import torch.nn as nn
import copy
import matplotlib.pyplot as plt


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


from signxai.torch_signxai.methods.guided import GuidedBackprop, DeconvNet


@pytest.fixture
def model():
    """Create a simple model for testing."""
    return SimpleNet()


@pytest.fixture
def input_tensor():
    """Create a test input tensor."""
    return torch.randn(1, 3, 32, 32, requires_grad=True)


def test_guided_backprop_initialization(model):
    """Test Guided Backprop initialization."""
    guided = GuidedBackprop(model)
    assert len(guided._hooks) > 0, "Should have hooks for ReLU layers"


def test_guided_backprop_attribution(model, input_tensor):
    """Test basic Guided Backprop attribution."""
    guided = GuidedBackprop(model)
    attribution = guided.attribute(input_tensor)

    # Check shape matches input
    assert attribution.shape == input_tensor.shape, "Attribution shape should match input"

    # Force small values to exactly zero
    attribution[torch.abs(attribution) < 1e-6] = 0.0

    # Check that gradients are sparse (some should be exactly zero)
    assert torch.any(attribution == 0), "Should have some exact zero values"

    # Check that there are both positive and negative gradients
    assert torch.any(attribution > 0), "Should have positive gradients"
    assert torch.any(attribution < 0), "Should have negative gradients"


def test_guided_backprop_with_target(model, input_tensor):
    """Test Guided Backprop with specific target."""
    guided = GuidedBackprop(model)
    target = torch.tensor([5])  # Choose class 5
    attribution = guided.attribute(input_tensor, target=target)

    assert attribution.shape == input_tensor.shape, "Attribution shape should match input"
    assert not torch.allclose(attribution, torch.zeros_like(attribution)), "Attribution should not be all zeros"


def test_guided_backprop_batch_processing(model):
    """Test Guided Backprop with batch input."""
    batch_size = 4
    inputs = torch.randn(batch_size, 3, 32, 32)

    guided = GuidedBackprop(model)
    attribution = guided.attribute(inputs)

    assert attribution.shape == inputs.shape, "Attribution shape should match batch input"


def test_deconvnet_attribution(model, input_tensor):
    """Test DeconvNet attribution."""
    deconv = DeconvNet(model)
    attribution = deconv.attribute(input_tensor)

    assert attribution.shape == input_tensor.shape, "Attribution shape should match input"
    assert not torch.allclose(attribution, torch.zeros_like(attribution)), "Attribution should not be all zeros"


def test_hook_cleanup(model):
    """Test that hooks are properly cleaned up."""
    guided = GuidedBackprop(model)
    initial_hooks = len(guided._hooks)

    # Delete the object
    del guided

    # Create a new one
    new_guided = GuidedBackprop(model)
    assert len(new_guided._hooks) == initial_hooks, "Should have same number of hooks"


def test_guided_backprop_gradients(model, input_tensor):
    """Test specific gradient properties of Guided Backprop."""
    guided = GuidedBackprop(model)
    attribution = guided.attribute(input_tensor)

    # Get regular gradients for comparison
    input_tensor.requires_grad_(True)
    output = model(input_tensor)
    target = output.argmax(dim=1)
    output[0, target].backward()
    regular_grads = input_tensor.grad.clone()

    # Force small values to exactly zero for both
    attribution[torch.abs(attribution) < 1e-6] = 0.0
    regular_grads[torch.abs(regular_grads) < 1e-6] = 0.0

    # Guided Backprop should be more sparse
    assert (attribution == 0).sum() >= (regular_grads == 0).sum(), "Guided Backprop should be more sparse"


def test_deconvnet_vs_guided(model, input_tensor):
    """Compare DeconvNet and Guided Backprop results."""
    # Create separate copies of the model for each attribution method
    model_guided = copy.deepcopy(model)
    model_deconv = copy.deepcopy(model)

    # Initialize attribution methods with separate models
    guided = GuidedBackprop(model_guided)
    deconv = DeconvNet(model_deconv)

    # Compute attributions
    guided_attr = guided.attribute(input_tensor)
    deconv_attr = deconv.attribute(input_tensor)

    # Results should be different
    are_close = torch.allclose(guided_attr, deconv_attr, atol=1e-4)
    assert not are_close, "DeconvNet and GuidedBackprop should produce different attributions"

    # Check shapes match
    assert guided_attr.shape == deconv_attr.shape, "Attribution shapes should match"

    # Force small values to exactly zero for both
    guided_attr[torch.abs(guided_attr) < 1e-6] = 0.0
    deconv_attr[torch.abs(deconv_attr) < 1e-6] = 0.0

    # Check sparsity patterns
    guided_sparsity = (guided_attr == 0).float().mean().item()
    deconv_sparsity = (deconv_attr == 0).float().mean().item()
    print(f"GuidedBackprop Sparsity: {guided_sparsity}")
    print(f"DeconvNet Sparsity: {deconv_sparsity}")

    # Allow some tolerance in sparsity comparison
    assert abs(guided_sparsity - deconv_sparsity) < 0.1, "Sparsity patterns should be similar"

    # Optional: Visualize attributions
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(guided_attr.squeeze().cpu().detach().numpy(), cmap='viridis')
    axs[0].set_title('Guided Backpropagation')
    axs[1].imshow(deconv_attr.squeeze().cpu().detach().numpy(), cmap='viridis')
    axs[1].set_title('DeconvNet')
    plt.show()
    """
    plt.close('all')


def test_calculate_relevancemap_guided_methods(model, input_tensor):
    """Test guided methods using calculate_relevancemap."""
    from signxai.torch_signxai.methods import calculate_relevancemap
    
    methods = ["guided_backprop", "deconvnet"]
    for method in methods:
        result = calculate_relevancemap(method, input_tensor, model, debug=False)
        assert isinstance(result, torch.Tensor)
        assert result.shape == input_tensor.shape
        assert not torch.allclose(result, torch.zeros_like(result))
        
    # Test with sign variants
    sign_methods = ["guided_backprop_x_sign", "deconvnet_x_sign"]
    for method in sign_methods:
        result = calculate_relevancemap(method, input_tensor, model, debug=False)
        assert isinstance(result, torch.Tensor)
        assert result.shape == input_tensor.shape
        assert not torch.allclose(result, torch.zeros_like(result))