"""
Tests for GradCAM-based methods
"""
import pytest
import numpy as np
from signxai.tf_signxai.methods.wrappers import (
    grad_cam,
    grad_cam_VGG16ILSVRC,
    guided_grad_cam_VGG16ILSVRC,
    calculate_relevancemap
)

def test_grad_cam(dummy_vgg16_model, dummy_input):
    """Test that GradCAM method returns correct shape"""
    last_conv = 'block5_conv3'  # Last convolutional layer in VGG16
    result = grad_cam(dummy_vgg16_model, dummy_input, last_conv=last_conv)
    
    # GradCAM should return a heatmap with same HxW as input
    # The result shape depends on whether resize=True (default) is used
    assert result.shape[0] == dummy_input.shape[0]
    assert result.shape[1] == dummy_input.shape[1]
    assert not np.isnan(result).any()

def test_grad_cam_vgg16(dummy_vgg16_model, dummy_input):
    """Test that GradCAM for VGG16 returns correct shape"""
    result = grad_cam_VGG16ILSVRC(dummy_vgg16_model, dummy_input)
    
    # GradCAM should return a heatmap with same HxW as input
    assert result.shape[0] == dummy_input.shape[0]
    assert result.shape[1] == dummy_input.shape[1]
    assert not np.isnan(result).any()

def test_guided_grad_cam_vgg16(dummy_vgg16_model, dummy_input):
    """Test that Guided GradCAM returns correct shape"""
    # This may fail if guided backprop implementation is incomplete
    # as it combines two methods
    result = guided_grad_cam_VGG16ILSVRC(dummy_vgg16_model, dummy_input)
    
    # Guided GradCAM should have same shape as input
    assert result.shape == dummy_input.shape
    assert not np.isnan(result).any()

def test_calculate_relevancemap_gradcam(dummy_vgg16_model, dummy_input):
    """Test that calculate_relevancemap with GradCAM method works correctly"""
    result = calculate_relevancemap('grad_cam_VGG16ILSVRC', dummy_input, dummy_vgg16_model)
    
    # GradCAM should return a heatmap with same HxW as input
    assert result.shape[0] == dummy_input.shape[0]
    assert result.shape[1] == dummy_input.shape[1]
    assert not np.isnan(result).any()

    # Should be the same as calling grad_cam_VGG16ILSVRC directly
    direct_result = grad_cam_VGG16ILSVRC(dummy_vgg16_model, dummy_input)
    assert np.array_equal(result, direct_result)