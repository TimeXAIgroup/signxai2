"""Tests for PyTorch implementation of signxai."""
import unittest
import torch
import torch.nn as nn
import numpy as np

# Import the TensorFlow-style API
from signxai.torch_signxai.methods.wrappers import (
    calculate_relevancemap as tf_calculate_relevancemap,
    calculate_relevancemaps as tf_calculate_relevancemaps,
    gradient,
    input_t_gradient,
    gradient_x_input,
    guided_backprop,
    integrated_gradients,
    smoothgrad,
    grad_cam,
    lrp_z,
    lrp_epsilon_0_1,
)

# Import the PyTorch-style API
from signxai.torch_signxai import calculate_relevancemap


class SimpleConvNet(nn.Module):
    """Simple CNN for testing."""
    
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class TestPyTorchAPI(unittest.TestCase):
    """Test PyTorch-style API implementation of signxai."""
    
    def setUp(self):
        """Set up test environment."""
        self.model = SimpleConvNet()
        self.model.eval()
        self.input_tensor = torch.randn(1, 3, 32, 32)
    
    def test_gradient_method(self):
        """Test vanilla gradients method."""
        relevance_map = calculate_relevancemap(self.model, self.input_tensor, method="gradients")
        self.assertEqual(relevance_map.shape, (1, 3, 32, 32))  # B, C, H, W
        self.assertTrue(isinstance(relevance_map, np.ndarray))
    
    def test_integrated_gradients_method(self):
        """Test integrated gradients method."""
        relevance_map = calculate_relevancemap(
            self.model, self.input_tensor, method="integrated_gradients", steps=10
        )
        self.assertEqual(relevance_map.shape, (1, 3, 32, 32))
        self.assertTrue(isinstance(relevance_map, np.ndarray))
    
    def test_guided_backprop_method(self):
        """Test guided backprop method."""
        relevance_map = calculate_relevancemap(self.model, self.input_tensor, method="guided_backprop")
        self.assertEqual(relevance_map.shape, (1, 3, 32, 32))
        self.assertTrue(isinstance(relevance_map, np.ndarray))
    
    def test_grad_cam_method(self):
        """Test Grad-CAM method."""
        relevance_map = calculate_relevancemap(self.model, self.input_tensor, method="grad_cam")
        self.assertEqual(relevance_map.shape, (1, 1, 32, 32))  # B, C, H, W
        self.assertTrue(isinstance(relevance_map, np.ndarray))
    
    def test_lrp_method(self):
        """Test LRP method."""
        relevance_map = calculate_relevancemap(self.model, self.input_tensor, method="lrp_epsilon")
        self.assertEqual(relevance_map.shape, (1, 3, 32, 32))
        self.assertTrue(isinstance(relevance_map, np.ndarray))
    
    def test_sign_transform(self):
        """Test SIGN transform with default mu=0.0."""
        relevance_map = calculate_relevancemap(
            self.model, self.input_tensor, method="gradients", apply_sign=True, sign_mu=0.0
        )
        self.assertEqual(relevance_map.shape, (1, 3, 32, 32))
        self.assertTrue(isinstance(relevance_map, np.ndarray))
        # Check that values are binary (-1, 0, 1)
        unique_values = np.unique(relevance_map)
        self.assertTrue(np.all(np.isin(unique_values, [-1.0, 0.0, 1.0])))
        
    def test_sign_transform_with_threshold(self):
        """Test SIGN transform with non-zero threshold."""
        # Create a tensor with known values
        test_tensor = torch.zeros(1, 3, 32, 32)
        test_tensor[:, 0, :16, :16] = 0.3  # Values between 0 and threshold
        test_tensor[:, 1, 16:, 16:] = 0.6  # Values above threshold
        test_tensor[:, 2, :16, 16:] = -0.3  # Values between -threshold and 0
        test_tensor[:, 2, 16:, :16] = -0.6  # Values below -threshold
        
        # Create a binary sign map directly
        from signxai.torch_signxai.methods.signed import calculate_sign_mu
        sign_map = calculate_sign_mu(test_tensor, mu=0.5)
        
        # Verify that values between -mu and mu become 0
        # Check some sample points from each region
        self.assertEqual(sign_map[0, 0, 8, 8].item(), 0.0)  # Between 0 and threshold
        self.assertEqual(sign_map[0, 1, 24, 24].item(), 1.0)  # Above threshold
        self.assertEqual(sign_map[0, 2, 8, 24].item(), 0.0)  # Between -threshold and 0
        self.assertEqual(sign_map[0, 2, 24, 8].item(), -1.0)  # Below -threshold
    
    def test_advanced_lrp_method(self):
        """Test advanced LRP methods."""
        relevance_map = calculate_relevancemap(
            self.model, self.input_tensor, method="lrp_alphabeta"
        )
        self.assertEqual(relevance_map.shape, (1, 3, 32, 32))
        self.assertTrue(isinstance(relevance_map, np.ndarray))


class TestTensorFlowAPI(unittest.TestCase):
    """Test TensorFlow-style API implementation of signxai."""
    
    def setUp(self):
        """Set up test environment."""
        self.model = SimpleConvNet()
        self.model.eval()
        self.input_tensor = torch.randn(3, 32, 32)  # No batch dimension
    
    def test_gradient_method(self):
        """Test vanilla gradients method."""
        relevance_map = gradient(self.model, self.input_tensor)
        self.assertEqual(relevance_map.shape, (3, 32, 32))  # C, H, W
        self.assertTrue(isinstance(relevance_map, np.ndarray))
    
    def test_input_t_gradient_method(self):
        """Test input * gradient method."""
        relevance_map = input_t_gradient(self.model, self.input_tensor)
        self.assertEqual(relevance_map.shape, (3, 32, 32))
        self.assertTrue(isinstance(relevance_map, np.ndarray))
    
    def test_gradient_x_input_method(self):
        """Test gradient * input method."""
        relevance_map = gradient_x_input(self.model, self.input_tensor)
        self.assertEqual(relevance_map.shape, (3, 32, 32))
        self.assertTrue(isinstance(relevance_map, np.ndarray))
    
    def test_guided_backprop_method(self):
        """Test guided backprop method."""
        relevance_map = guided_backprop(self.model, self.input_tensor)
        self.assertEqual(relevance_map.shape, (3, 32, 32))
        self.assertTrue(isinstance(relevance_map, np.ndarray))
    
    def test_integrated_gradients_method(self):
        """Test integrated gradients method."""
        relevance_map = integrated_gradients(self.model, self.input_tensor, steps=10)
        self.assertEqual(relevance_map.shape, (3, 32, 32))
        self.assertTrue(isinstance(relevance_map, np.ndarray))
    
    def test_smoothgrad_method(self):
        """Test SmoothGrad method."""
        relevance_map = smoothgrad(self.model, self.input_tensor, noise_scale=0.1, augment_by_n=5)
        self.assertEqual(relevance_map.shape, (3, 32, 32))
        self.assertTrue(isinstance(relevance_map, np.ndarray))
    
    def test_grad_cam_method(self):
        """Test Grad-CAM method."""
        relevance_map = grad_cam(self.model, self.input_tensor)
        # Grad-CAM produces a spatial heatmap
        self.assertEqual(relevance_map.shape, (32, 32))
        self.assertTrue(isinstance(relevance_map, np.ndarray))
    
    def test_lrp_methods(self):
        """Test LRP methods."""
        # Test LRP-Z
        relevance_map_z = lrp_z(self.model, self.input_tensor)
        self.assertEqual(relevance_map_z.shape, (3, 32, 32))
        self.assertTrue(isinstance(relevance_map_z, np.ndarray))
        
        # Test LRP-Epsilon
        relevance_map_eps = lrp_epsilon_0_1(self.model, self.input_tensor)
        self.assertEqual(relevance_map_eps.shape, (3, 32, 32))
        self.assertTrue(isinstance(relevance_map_eps, np.ndarray))
    
    def test_tf_calculate_relevancemap(self):
        """Test TensorFlow-style calculate_relevancemap function."""
        relevance_map = tf_calculate_relevancemap("gradient", self.input_tensor, self.model)
        self.assertEqual(relevance_map.shape, (3, 32, 32))
        self.assertTrue(isinstance(relevance_map, np.ndarray))
    
    def test_tf_calculate_relevancemaps_batch(self):
        """Test TensorFlow-style calculate_relevancemaps function with batch input."""
        batch_size = 3
        batch_input = torch.stack([torch.randn(3, 32, 32) for _ in range(batch_size)])
        
        relevance_maps = tf_calculate_relevancemaps("gradient", batch_input, self.model)
        self.assertEqual(relevance_maps.shape, (batch_size, 3, 32, 32))
        self.assertTrue(isinstance(relevance_maps, np.ndarray))


if __name__ == "__main__":
    unittest.main()