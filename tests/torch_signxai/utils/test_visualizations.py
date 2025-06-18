# tests/test_utils/test_visualizations.py

"""Tests for visualization utilities."""

from typing import Union, Optional, Tuple, Dict

import pytest
import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt

from signxai.torch_signxai.utils.visualizations import (
    normalize_attribution,
    visualize_attribution,
    visualize_multiple_attributions
)


@pytest.fixture
def sample_image():
    """Create a sample image."""
    return np.random.rand(3, 32, 32)


@pytest.fixture
def sample_attribution():
    """Create a sample attribution map."""
    return np.random.randn(32, 32)


def test_normalize_attribution(sample_attribution):
    """Test attribution normalization."""
    # Test numpy input
    normalized = normalize_attribution(sample_attribution)
    assert isinstance(normalized, np.ndarray), "Normalized attribution should be a numpy array."
    assert normalized.min() >= 0, "Normalized attribution should have min >= 0."
    assert normalized.max() <= 1, "Normalized attribution should have max <= 1."

    # Test torch tensor input
    tensor_attr = torch.from_numpy(sample_attribution)
    normalized = normalize_attribution(tensor_attr)
    assert isinstance(normalized, np.ndarray), "Normalized attribution should be a numpy array when input is a torch tensor."
    assert normalized.min() >= 0, "Normalized attribution should have min >= 0."
    assert normalized.max() <= 1, "Normalized attribution should have max <= 1."

    # Test percentile clipping
    percentile = 90
    normalized = normalize_attribution(sample_attribution, percentile=percentile)
    assert isinstance(normalized, np.ndarray), "Normalized attribution should be a numpy array."
    assert normalized.min() >= 0, "Normalized attribution should have min >= 0."
    assert normalized.max() <= 1, "Normalized attribution should have max <= 1."


def test_visualize_attribution(sample_image, sample_attribution):
    """Test attribution visualization."""
    # Basic visualization
    fig = visualize_attribution(sample_image, sample_attribution)
    assert isinstance(fig, plt.Figure), "Visualization should return a matplotlib Figure."
    plt.close(fig)

    # Test with different cmaps
    fig = visualize_attribution(sample_image, sample_attribution, cmap='viridis')
    assert isinstance(fig, plt.Figure), "Visualization with cmap='viridis' should return a matplotlib Figure."
    plt.close(fig)

    # Test with different alpha
    fig = visualize_attribution(sample_image, sample_attribution, alpha=0.8)
    assert isinstance(fig, plt.Figure), "Visualization with alpha=0.8 should return a matplotlib Figure."
    plt.close(fig)

    # Test without colorbar
    fig = visualize_attribution(
        sample_image,
        sample_attribution,
        show_colorbar=False
    )
    assert isinstance(fig, plt.Figure), "Visualization without colorbar should return a matplotlib Figure."
    plt.close(fig)

    # Test with percentile clipping
    fig = visualize_attribution(
        sample_image,
        sample_attribution,
        percentile=90
    )
    assert isinstance(fig, plt.Figure), "Visualization with percentile=90 should return a matplotlib Figure."
    plt.close(fig)


def test_visualize_multiple_attributions(sample_image):
    """Test multiple attribution visualization."""
    # Create multiple attributions
    attributions = {
        'Method 1': np.random.randn(32, 32),
        'Method 2': np.random.randn(32, 32),
        'Method 3': np.random.randn(32, 32)
    }

    # Basic visualization
    fig = visualize_multiple_attributions(image=sample_image, attributions=attributions)
    assert isinstance(fig, plt.Figure), "Multiple attribution visualization should return a matplotlib Figure."
    plt.close(fig)

    # Test with custom figsize
    fig = visualize_multiple_attributions(
        image=sample_image,
        attributions=attributions,
        figsize=(12, 4)
    )
    assert isinstance(fig, plt.Figure), "Visualization with figsize=(12, 4) should return a matplotlib Figure."
    plt.close(fig)

    # Test with single attribution
    single_attribution = {'Method 1': np.random.randn(32, 32)}
    fig = visualize_multiple_attributions(image=sample_image, attributions=single_attribution)
    assert isinstance(fig, plt.Figure), "Visualization with single attribution should return a matplotlib Figure."
    plt.close(fig)


def test_different_input_formats():
    """Test handling of different input formats."""
    # Test different image formats
    # Channel-first RGB
    img_chw = torch.rand(3, 32, 32)
    attr = torch.rand(32, 32)
    fig = visualize_attribution(img_chw, attr)
    assert isinstance(fig, plt.Figure), "Visualization with channel-first RGB should return a matplotlib Figure."
    plt.close(fig)

    # Channel-last RGB
    img_hwc = torch.rand(32, 32, 3)
    fig = visualize_attribution(img_hwc, attr)
    assert isinstance(fig, plt.Figure), "Visualization with channel-last RGB should return a matplotlib Figure."
    plt.close(fig)

    # Grayscale
    img_gray = torch.rand(32, 32)
    fig = visualize_attribution(img_gray, attr)
    assert isinstance(fig, plt.Figure), "Visualization with grayscale image should return a matplotlib Figure."
    plt.close(fig)

    # Single channel with dimension
    img_gray_dim = torch.rand(1, 32, 32)
    fig = visualize_attribution(img_gray_dim, attr)
    assert isinstance(fig, plt.Figure), "Visualization with single channel (1, 32, 32) should return a matplotlib Figure."
    plt.close(fig)


def test_error_handling():
    """Test error handling in visualization functions."""
    # Mismatched shapes
    img = torch.rand(3, 32, 32)
    attr = torch.rand(16, 16)  # Wrong size

    with pytest.raises(ValueError):
        visualize_attribution(img, attr)

    # Empty attribution dict
    with pytest.raises(ValueError):
        visualize_multiple_attributions(image=np.random.rand(3, 32, 32), attributions={})

    # Mismatched shapes in multiple attributions
    wrong_attributions = {
        'Method 1': np.random.randn(32, 32),
        'Method 2': np.random.randn(16, 16)  # Wrong size
    }

    with pytest.raises(ValueError):
        visualize_multiple_attributions(image=np.random.rand(3, 32, 32), attributions=wrong_attributions)


def test_normalization_edge_cases():
    """Test normalization edge cases."""
    # All zeros
    zero_attr = np.zeros((32, 32))
    norm = normalize_attribution(zero_attr)
    assert np.allclose(norm, 0), "Normalization of all zeros should return all zeros."

    # All ones
    ones_attr = np.ones((32, 32))
    norm = normalize_attribution(ones_attr)
    assert np.allclose(norm, 1), "Normalization of all ones should return all ones."

    # Single value
    single_val = np.array([[1.0]])
    norm = normalize_attribution(single_val)
    assert not np.isnan(norm).any(), "Normalization should not produce NaNs for single value input."
    assert norm.shape == single_val.shape, "Normalized output should have the same shape as input."

    # Mixed positive and negative
    mixed_attr = np.array([[-1, 0], [1, 2]])
    norm = normalize_attribution(mixed_attr)
    assert norm.min() >= 0, "Normalized mixed attribution should have min >= 0."
    assert norm.max() <= 1, "Normalized mixed attribution should have max <= 1."


def test_colormap_options():
    """Test different colormap options."""
    img = torch.rand(3, 32, 32)
    attr = torch.rand(32, 32)

    # Test different built-in colormaps
    colormaps = ['RdBu_r', 'viridis', 'plasma', 'seismic']
    for cmap in colormaps:
        fig = visualize_attribution(img, attr, cmap=cmap)
        assert isinstance(fig, plt.Figure), f"Visualization with cmap='{cmap}' should return a matplotlib Figure."
        plt.close(fig)

    # Test with multiple attributions
    attributions = {
        'Method 1': np.random.randn(32, 32),
        'Method 2': np.random.randn(32, 32)
    }

    for cmap in colormaps:
        fig = visualize_multiple_attributions(image=np.random.rand(3, 32, 32), attributions=attributions, cmap=cmap)
        assert isinstance(fig, plt.Figure), f"Multiple attribution visualization with cmap='{cmap}' should return a matplotlib Figure."
        plt.close(fig)


def test_device_handling():
    """Test handling of tensors on different devices."""
    # CPU tensors
    cpu_img = torch.rand(3, 32, 32)
    cpu_attr = torch.rand(32, 32)
    fig = visualize_attribution(cpu_img, cpu_attr)
    assert isinstance(fig, plt.Figure), "Visualization with CPU tensors should return a matplotlib Figure."
    plt.close(fig)

    if torch.cuda.is_available():
        # Test with CUDA tensors
        cuda_img = torch.rand(3, 32, 32).cuda()
        cuda_attr = torch.rand(32, 32).cuda()

        fig = visualize_attribution(cuda_img, cuda_attr)
        assert isinstance(fig, plt.Figure), "Visualization with CUDA tensors should return a matplotlib Figure."
        plt.close(fig)

        # Test with mixed devices
        cpu_img = torch.rand(3, 32, 32)
        fig = visualize_attribution(cpu_img, cuda_attr)
        assert isinstance(fig, plt.Figure), "Visualization with mixed device tensors should return a matplotlib Figure."
        plt.close(fig)


def test_visualization_consistency():
    """Additional test to ensure visualization consistency."""
    # Create deterministic data
    np.random.seed(0)
    torch.manual_seed(0)

    sample_img = np.ones((3, 32, 32))
    sample_attr = np.zeros((32, 32))

    # Add a single positive value
    sample_attr[16, 16] = 1.0

    # Normalize
    normalized = normalize_attribution(sample_attr)
    assert normalized.max() == 1.0, "Normalization should scale the single positive value to 1."

    # Visualize
    fig = visualize_attribution(sample_img, sample_attr)
    assert isinstance(fig, plt.Figure), "Visualization should return a matplotlib Figure."
    plt.close(fig)
