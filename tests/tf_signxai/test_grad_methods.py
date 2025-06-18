"""
Tests for gradient-based methods
"""
import pytest
import numpy as np
from signxai.tf_signxai.methods.wrappers import (
    gradient, 
    gradient_x_input, 
    gradient_x_sign, 
    gradient_x_sign_mu_0,
    gradient_x_sign_mu_0_5,
    gradient_x_sign_mu_neg_0_5,
    calculate_relevancemap
)

def test_gradient(dummy_vgg16_model, dummy_input):
    """Test that gradient method returns correct shape"""
    result = gradient(dummy_vgg16_model, dummy_input)
    assert result.shape == dummy_input.shape
    assert not np.isnan(result).any()

def test_gradient_x_input(dummy_vgg16_model, dummy_input):
    """Test that gradient_x_input method returns correct shape"""
    result = gradient_x_input(dummy_vgg16_model, dummy_input)
    assert result.shape == dummy_input.shape
    assert not np.isnan(result).any()

def test_gradient_x_sign(dummy_vgg16_model, dummy_input):
    """Test that gradient_x_sign method returns correct shape"""
    result = gradient_x_sign(dummy_vgg16_model, dummy_input)
    assert result.shape == dummy_input.shape
    assert not np.isnan(result).any()

def test_gradient_sign_mu_variants(dummy_vgg16_model, dummy_input):
    """Test that different mu values for gradient_x_sign_mu produce different results"""
    result_mu0 = gradient_x_sign_mu_0(dummy_vgg16_model, dummy_input)
    result_mu05 = gradient_x_sign_mu_0_5(dummy_vgg16_model, dummy_input)
    result_mu_neg05 = gradient_x_sign_mu_neg_0_5(dummy_vgg16_model, dummy_input)
    
    assert result_mu0.shape == dummy_input.shape
    assert result_mu05.shape == dummy_input.shape
    assert result_mu_neg05.shape == dummy_input.shape
    
    # Different mu values should produce different results
    assert not np.array_equal(result_mu0, result_mu05)
    assert not np.array_equal(result_mu0, result_mu_neg05)
    assert not np.array_equal(result_mu05, result_mu_neg05)

def test_calculate_relevancemap_gradient(dummy_vgg16_model, dummy_input):
    """Test that calculate_relevancemap with gradient method works correctly"""
    result = calculate_relevancemap('gradient', dummy_input, dummy_vgg16_model)
    assert result.shape == dummy_input.shape
    assert not np.isnan(result).any()

    # Should be the same as calling gradient directly
    direct_result = gradient(dummy_vgg16_model, dummy_input)
    assert np.array_equal(result, direct_result)