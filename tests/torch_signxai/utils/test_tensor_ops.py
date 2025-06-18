"""Tests for tensor operation utilities with TensorFlow compatibility."""

import pytest
import torch
import tensorflow as tf
import numpy as np
from PIL import Image

from signxai.torch_signxai.utils.tensor_ops import (
    preprocess_vgg16,
    standardize_tensor,
    normalize_tensor,
    verify_tensor_range,
    IMAGENET_MEAN_BGR
)

@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    # Create red image to match TensorFlow preprocessing
    return Image.new('RGB', (224, 224), color='red')

def test_preprocess_vgg16_matches_tensorflow(sample_image):
    """Test that preprocessing exactly matches TensorFlow VGG16."""
    # TensorFlow preprocessing
    tf_img = np.array(sample_image)
    tf_processed = tf.keras.applications.vgg16.preprocess_input(tf_img.copy())

    # Our preprocessing
    _, pt_processed = preprocess_vgg16(sample_image)
    pt_processed = pt_processed.squeeze(0).permute(1, 2, 0).numpy()

    # Compare results
    np.testing.assert_allclose(tf_processed, pt_processed, rtol=1e-5, atol=1e-5)

    # Additional checks for each channel
    for i in range(3):
        assert abs(tf_processed[..., i].mean() - pt_processed[..., i].mean()) < 1e-5

def test_preprocess_vgg16_format(sample_image):
    """Test preprocessing format and value ranges."""
    # Check output format
    _, tensor = preprocess_vgg16(sample_image)
    assert tensor.dim() == 4
    assert tensor.shape[0] == 1  # batch size
    assert tensor.shape[1] == 3  # channels
    assert tensor.shape[2:] == (224, 224)  # spatial dims

    # For a red image (255, 0, 0), after BGR conversion and mean subtraction:
    # B channel should be -mean[0]
    # G channel should be -mean[1]
    # R channel should be 255-mean[2]
    img_bgr = tensor.squeeze(0).numpy()

    # B channel (should be close to -mean[0])
    assert abs(img_bgr[0].mean() + IMAGENET_MEAN_BGR[0]) < 1.0

    # G channel (should be close to -mean[1])
    assert abs(img_bgr[1].mean() + IMAGENET_MEAN_BGR[1]) < 1.0

    # R channel (should be close to 255-mean[2])
    assert abs(img_bgr[2].mean() - (255 - IMAGENET_MEAN_BGR[2])) < 1.0

def test_edge_cases():
    """Test edge cases and error handling."""
    # Empty tensor
    with pytest.raises(RuntimeError):
        standardize_tensor(torch.tensor([]))

    # Invalid image type
    with pytest.raises(TypeError):
        preprocess_vgg16([1, 2, 3])

    # Single value tensor
    single_value = torch.tensor([1.0])
    normalized = normalize_tensor(single_value)
    assert not torch.isnan(normalized).any()

def test_numerical_stability():
    """Test numerical stability of operations."""
    # Test with small values
    small_tensor = torch.rand(10, 10) * 1e-10
    result = standardize_tensor(small_tensor)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()

    # Test with large values
    large_tensor = torch.rand(10, 10) * 1e10
    result = normalize_tensor(large_tensor)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()

def test_device_handling():
    """Test device handling."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        sample_image = Image.new('RGB', (224, 224))
        _, tensor = preprocess_vgg16(sample_image, device=device)
        assert tensor.device == device