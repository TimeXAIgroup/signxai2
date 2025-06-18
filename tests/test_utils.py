"""
Tests for utility functions
"""
import pytest
import numpy as np
from signxai.utils.utils import (
    normalize_heatmap,
    aggregate_and_normalize_relevancemap_rgb
)

def test_normalize_heatmap():
    """Test that normalize_heatmap correctly normalizes to [-1, 1]"""
    # Create a sample heatmap with values from -10 to 10
    heatmap = np.linspace(-10, 10, 100).reshape(10, 10)
    
    # Normalize the heatmap
    normalized = normalize_heatmap(heatmap)
    
    # Check the shape is preserved
    assert normalized.shape == heatmap.shape
    
    # Check the range is [-1, 1]
    assert np.min(normalized) == -1
    assert np.max(normalized) == 1
    
    # Check edge case - constant heatmap
    constant_heatmap = np.ones((10, 10))
    normalized_constant = normalize_heatmap(constant_heatmap)
    assert np.all(normalized_constant == 0)  # Should be all zeros when min=max

def test_aggregate_and_normalize_relevancemap_rgb():
    """Test that aggregate_and_normalize_relevancemap_rgb correctly processes RGB maps"""
    # Create a sample RGB relevance map
    rgb_map = np.random.uniform(-1, 1, (10, 10, 3))
    
    # Aggregate and normalize
    result = aggregate_and_normalize_relevancemap_rgb(rgb_map)
    
    # Check the shape is correct (channels are aggregated)
    assert result.shape == (10, 10)
    
    # Check the range is [-1, 1]
    assert np.min(result) == -1
    assert np.max(result) == 1
    
    # Test with single-channel input (should work the same)
    grayscale_map = np.random.uniform(-1, 1, (10, 10))
    result_grayscale = aggregate_and_normalize_relevancemap_rgb(grayscale_map)
    
    # Check the shape is preserved
    assert result_grayscale.shape == grayscale_map.shape
    
    # Check the range is [-1, 1]
    assert np.min(result_grayscale) == -1
    assert np.max(result_grayscale) == 1