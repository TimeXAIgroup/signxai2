"""PyTorch implementation of Grad-CAM."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Union, Optional, Callable, Dict, Any


class GradCAM:
    """Grad-CAM implementation for PyTorch models.
    
    Grad-CAM uses the gradients of a target concept flowing into the final
    convolutional layer to produce a coarse localization map highlighting
    important regions in the image for prediction.
    """
    
    def __init__(self, model, target_layer):
        """Initialize GradCAM.
        
        Args:
            model: PyTorch model
            target_layer: Target layer for Grad-CAM
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Register hooks
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def forward(self, x, target_class=None):
        """Generate Grad-CAM attribution map.
        
        Args:
            x: Input tensor
            target_class: Target class index (None for argmax)
            
        Returns:
            Grad-CAM attribution map
        """
        # Forward pass
        self.model.zero_grad()
        output = self.model(x)
        
        # Select target class
        if target_class is None:
            target_class = output.argmax(dim=1)
        elif isinstance(target_class, int):
            # Convert int to tensor
            target_class = torch.tensor([target_class], device=output.device)
        elif isinstance(target_class, (list, tuple)):
            # Convert list/tuple to tensor
            target_class = torch.tensor(target_class, device=output.device)
        
        # Create one-hot encoding
        if output.dim() == 2:  # Batch output
            one_hot = torch.zeros_like(output)
            if target_class.dim() == 0:  # Single value
                target_class = target_class.unsqueeze(0)
            one_hot.scatter_(1, target_class.unsqueeze(1), 1.0)
        else:  # Single output
            one_hot = torch.zeros_like(output)
            if target_class.dim() > 0:
                target_class = target_class[0]  # Get scalar value
            one_hot[target_class] = 1.0
        
        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Calculate weights
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weight activations by importance
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Normalize
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)
        
        return cam


def find_target_layer(model):
    """Find the last convolutional layer in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Last convolutional layer
    """
    # Check if model has a layer named 'layer4' (ResNet) or 'features' (VGG)
    if hasattr(model, 'layer4'):
        # ResNet-like models
        return model.layer4[-1].conv2
    elif hasattr(model, 'features'):
        # VGG-like models
        for i in range(len(model.features) - 1, -1, -1):
            if isinstance(model.features[i], nn.Conv2d):
                return model.features[i]
    
    # Search recursively for the last conv layer
    last_conv = None
    
    def search_conv(module):
        nonlocal last_conv
        for m in module.children():
            if len(list(m.children())) > 0:
                # Recurse into submodules
                search_conv(m)
            elif isinstance(m, nn.Conv2d):
                last_conv = m
    
    search_conv(model)
    return last_conv


def calculate_grad_cam_relevancemap(model, input_tensor, target_layer=None, target_class=None):
    """Calculate Grad-CAM relevance map for images.
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor
        target_layer: Target layer for Grad-CAM (None to auto-detect)
        target_class: Target class index (None for argmax)
        
    Returns:
        Grad-CAM relevance map
    """
    # Find target layer if not provided
    if target_layer is None:
        target_layer = find_target_layer(model)
        if target_layer is None:
            raise ValueError("Could not find convolutional layer for Grad-CAM")
    
    # Initialize Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # Generate attribution map
    with torch.enable_grad():
        cam = grad_cam.forward(input_tensor, target_class)
    
    # Convert to numpy and return
    if input_tensor.dim() == 4:  # Batch
        return cam.squeeze(1).detach().cpu().numpy()
    else:  # Single image
        return cam.squeeze().detach().cpu().numpy()


def calculate_grad_cam_relevancemap_timeseries(model, input_tensor, target_layer=None, target_class=None):
    """Calculate Grad-CAM relevance map for time series data.
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor (B, C, T)
        target_layer: Target layer for Grad-CAM (None to auto-detect)
        target_class: Target class index (None for argmax)
        
    Returns:
        Grad-CAM relevance map
    """
    # Implementation similar to image case but for 1D time series
    # Find target layer if not provided
    if target_layer is None:
        # Find the last conv1d layer
        for module in reversed(list(model.modules())):
            if isinstance(module, nn.Conv1d):
                target_layer = module
                break
    
    if target_layer is None:
        raise ValueError("Could not find Conv1d layer for time series Grad-CAM")
    
    # Store activations and gradients
    activations = []
    gradients = []
    
    # Register hooks
    def forward_hook(module, input, output):
        activations.append(output)
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    
    # Forward pass
    model.zero_grad()
    output = model(input_tensor)
    
    # Select target class
    if target_class is None:
        target_class = output.argmax(dim=1)
    
    # Create one-hot encoding
    one_hot = torch.zeros_like(output)
    one_hot.scatter_(1, target_class.unsqueeze(1), 1.0)
    
    # Backward pass
    output.backward(gradient=one_hot)
    
    # Clean up hooks
    forward_handle.remove()
    backward_handle.remove()
    
    # Calculate weights (averaging over time dimension)
    weights = torch.mean(gradients[0], dim=2, keepdim=True)
    
    # Weight activations by importance
    cam = torch.sum(weights * activations[0], dim=1, keepdim=True)
    
    # Apply ReLU
    cam = F.relu(cam)
    
    # Resize to input size
    cam = F.interpolate(cam, size=input_tensor.shape[2], mode='linear')
    
    # Normalize
    if torch.max(cam) > 0:
        cam = cam / torch.max(cam)
    
    # Convert to numpy and return
    return cam.squeeze(1).detach().cpu().numpy()