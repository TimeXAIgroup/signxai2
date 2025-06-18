"""
Tests for LRP-based methods
"""
import pytest
import numpy as np
import tensorflow as tf
from signxai.tf_signxai.methods.wrappers import (
    lrp_z, 
    lrpsign_z,
    lrp_epsilon_0_1,
    lrpsign_epsilon_0_1,
    lrpz_epsilon_0_1,
    calculate_relevancemap
)
from signxai.tf_signxai.methods.innvestigate.analyzer import create_analyzer

def test_lrp_z_direct(dummy_vgg16_model, dummy_input):
    """Test direct LRP-Z using the innvestigate analyzer"""
    # Use the direct approach for LRP
    analyzer = create_analyzer('lrp.z', dummy_vgg16_model, input_layer_rule='Z')
    analysis = analyzer.analyze(X=[dummy_input], neuron_selection='max_activation')
    
    # Extract the result
    if isinstance(analysis, dict):
        result = analysis[list(analysis.keys())[0]][0]
    else:
        result = analysis[0]
    
    # Check the shape and values
    assert result.shape == dummy_input.shape
    assert not np.isnan(result).any()

def test_lrpsign_z_direct(dummy_vgg16_model, dummy_input):
    """Test direct LRP-SIGN using the innvestigate analyzer"""
    # Use the direct approach for LRP
    analyzer = create_analyzer('lrp.z', dummy_vgg16_model, input_layer_rule='SIGN')
    analysis = analyzer.analyze(X=[dummy_input], neuron_selection='max_activation')
    
    # Extract the result
    if isinstance(analysis, dict):
        result = analysis[list(analysis.keys())[0]][0]
    else:
        result = analysis[0]
    
    # Check the shape and values
    assert result.shape == dummy_input.shape
    assert not np.isnan(result).any()

def test_lrp_epsilon_direct(dummy_vgg16_model, dummy_input):
    """Test direct LRP-Epsilon using the innvestigate analyzer"""
    # Use the direct approach for LRP
    analyzer = create_analyzer('lrp.epsilon', dummy_vgg16_model, epsilon=0.1)
    analysis = analyzer.analyze(X=[dummy_input], neuron_selection='max_activation')
    
    # Extract the result
    if isinstance(analysis, dict):
        result = analysis[list(analysis.keys())[0]][0]
    else:
        result = analysis[0]
    
    # Check the shape and values
    assert result.shape == dummy_input.shape
    assert not np.isnan(result).any()

def test_lrp_variants_direct(dummy_vgg16_model, dummy_input):
    """Test that different LRP variants produce different results using direct analysis"""
    # LRP-Z 
    analyzer1 = create_analyzer('lrp.z', dummy_vgg16_model, input_layer_rule='Z')
    analysis1 = analyzer1.analyze(X=[dummy_input], neuron_selection='max_activation')
    result_lrp_z = analysis1[list(analysis1.keys())[0]][0] if isinstance(analysis1, dict) else analysis1[0]
    
    # LRP-SIGN
    analyzer2 = create_analyzer('lrp.z', dummy_vgg16_model, input_layer_rule='SIGN')
    analysis2 = analyzer2.analyze(X=[dummy_input], neuron_selection='max_activation')
    result_lrpsign_z = analysis2[list(analysis2.keys())[0]][0] if isinstance(analysis2, dict) else analysis2[0]
    
    # LRP-Epsilon
    analyzer3 = create_analyzer('lrp.epsilon', dummy_vgg16_model, epsilon=0.1)
    analysis3 = analyzer3.analyze(X=[dummy_input], neuron_selection='max_activation')
    result_lrp_epsilon = analysis3[list(analysis3.keys())[0]][0] if isinstance(analysis3, dict) else analysis3[0]
    
    # Check shapes
    assert result_lrp_z.shape == dummy_input.shape
    assert result_lrpsign_z.shape == dummy_input.shape
    assert result_lrp_epsilon.shape == dummy_input.shape
    
    # Check for NaNs
    assert not np.isnan(result_lrp_z).any()
    assert not np.isnan(result_lrpsign_z).any()
    assert not np.isnan(result_lrp_epsilon).any()
    
    # Different variants should produce different results
    # Use correlation coefficient instead of exact equality
    def correlation(a, b):
        return np.corrcoef(a.flatten(), b.flatten())[0, 1]
    
    # Expect some differences between methods
    assert correlation(result_lrp_z, result_lrpsign_z) < 0.99
    assert correlation(result_lrp_z, result_lrp_epsilon) < 0.99