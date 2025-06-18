"""PyTorch implementation of Grad-CAM."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Optional, Tuple, List


class GradCAM:
    """Grad-CAM (Gradient-weighted Class Activation Mapping) implementation for PyTorch.
    
    Implements the method described in "Grad-CAM: Visual Explanations from Deep Networks 
    via Gradient-based Localization" (https://arxiv.org/abs/1610.02391).
    """
    
    def __init__(self, model, target_layer=None):
        """Initialize Grad-CAM with a model and target layer.
        
        Args:
            model: PyTorch model
            target_layer: Layer to use for Grad-CAM (usually the last convolutional layer)
                         If None, will try to automatically find the last convolutional layer
        """
        self.model = model
        
        # If target_layer is not provided, try to find the last convolutional layer
        if target_layer is None:
            self.target_layer = self._find_target_layer(model)
        else:
            self.target_layer = target_layer
            
        # Check if target_layer was found or provided
        if self.target_layer is None:
            raise ValueError("Could not automatically identify a target convolutional layer. "
                            "Please specify one explicitly.")
        
        # Register hooks to get activation and gradient
        self.activations = None
        self.gradients = None
        self.hooks = []
        
    def _find_target_layer(self, model):
        """Find the last convolutional layer in the model."""
        target_layer = None
        
        # Try to find the last convolutional layer
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                target_layer = module
                print(f"Found convolutional layer: {name}")
                break
                
        return target_layer
    
    def _register_hooks(self):
        """Register hooks for activation and gradient capture."""
        # Clear any existing hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # Forward hook to capture activations
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        # Backward hook to capture gradients
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        
        # Register hooks
        forward_handle = self.target_layer.register_forward_hook(forward_hook)
        
        # Use register_full_backward_hook for newer PyTorch, or register_backward_hook for older
        try:
            backward_handle = self.target_layer.register_full_backward_hook(backward_hook)
        except AttributeError:
            # Fallback for older PyTorch versions
            backward_handle = self.target_layer.register_backward_hook(backward_hook)
            
        # Store handles to remove later
        self.hooks.extend([forward_handle, backward_handle])
        
    def attribute(self, inputs, target=None, resize_to_input=True):
        """Generate Grad-CAM heatmap.
        
        Args:
            inputs: Input tensor
            target: Target class index (None for argmax)
            resize_to_input: Whether to resize heatmap to input size
            
        Returns:
            Grad-CAM heatmap (same size as input if resize_to_input=True)
        """
        # Set model to eval mode
        original_mode = self.model.training
        self.model.eval()
        
        # Handle tensor conversion
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float32)
            
        # Clone input to avoid modifying the original
        inputs = inputs.clone().detach().requires_grad_(True)
        
        # Reset stored activations and gradients
        self.activations = None
        self.gradients = None
        
        # Register hooks for this forward/backward pass
        self._register_hooks()
        
        try:
            # Forward pass
            self.model.zero_grad()
            outputs = self.model(inputs)
            
            # Determine target class
            if target is None:
                # Use argmax of the output
                target_indices = outputs.argmax(dim=1)
            elif isinstance(target, int):
                # Use the same target for all examples in the batch
                target_indices = torch.full((inputs.shape[0],), target, 
                                          dtype=torch.long, device=inputs.device)
            elif isinstance(target, torch.Tensor):
                if target.numel() == 1:
                    # Single scalar tensor target
                    target_indices = torch.full((inputs.shape[0],), target.item(), 
                                              dtype=torch.long, device=inputs.device)
                else:
                    # Tensor with multiple targets (one per batch item)
                    target_indices = target
            else:
                raise ValueError(f"Unsupported target type: {type(target)}")
                
            # Create one-hot encoding for target class(es)
            one_hot = torch.zeros_like(outputs)
            for i, idx in enumerate(target_indices):
                one_hot[i, idx] = 1.0
                
            # Backward pass to get gradients
            outputs.backward(gradient=one_hot)
            
            # Ensure we have activations and gradients
            if self.activations is None or self.gradients is None:
                raise ValueError("Could not capture activations or gradients. "
                                "Check that the target layer is correct.")
                
            # Global average pooling of gradients
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
            
            # Weight the activations by the gradients
            cam = (weights * self.activations).sum(dim=1, keepdim=True)
            
            # Apply ReLU to keep only positive influences
            cam = F.relu(cam)
            
            # Normalize CAM to 0-1 range per example
            batch_size = cam.shape[0]
            cams_normalized = []
            for i in range(batch_size):
                cam_i = cam[i]
                min_val = cam_i.min()
                max_val = cam_i.max()
                
                if max_val > min_val:
                    # Normalize to 0-1
                    cam_i = (cam_i - min_val) / (max_val - min_val)
                else:
                    # If constant value, set to 0.5 (neutral)
                    cam_i = torch.ones_like(cam_i) * 0.5
                    
                cams_normalized.append(cam_i)
                
            cam = torch.cat(cams_normalized, dim=0).unsqueeze(1)
            
            # Resize CAM to match input size if requested
            if resize_to_input and inputs.dim() == 4:  # For images
                # Get input spatial dimensions
                input_h, input_w = inputs.shape[2], inputs.shape[3]
                
                # Resize CAM to match input size
                cam = F.interpolate(cam, size=(input_h, input_w), mode='bilinear', align_corners=False)
                
            # Remove hooks
            for hook in self.hooks:
                hook.remove()
            self.hooks = []
                
            # Restore model mode
            self.model.train(original_mode)
            
            return cam
            
        except Exception as e:
            # Clean up hooks even if an error occurs
            for hook in self.hooks:
                hook.remove()
            self.hooks = []
            
            # Restore model mode
            self.model.train(original_mode)
            
            # Re-raise the exception
            raise e


def calculate_grad_cam_relevancemap(model, input_tensor, target_layer=None, target_class=None):
    """Calculate Grad-CAM relevance map.
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor
        target_layer: Target layer for Grad-CAM (usually the last convolutional layer)
        target_class: Target class index (None for argmax)
        
    Returns:
        Grad-CAM heatmap (same size as input)
    """
    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam.attribute(input_tensor, target=target_class)
    
    # Convert to numpy array
    if isinstance(cam, torch.Tensor):
        cam = cam.detach().cpu().numpy()
        
    # Remove batch dimension if single example
    if cam.shape[0] == 1:
        cam = cam[0]
        
    # Remove channel dimension if present
    if cam.ndim == 3 and cam.shape[0] == 1:
        cam = cam[0]
        
    return cam


def calculate_grad_cam_relevancemap_timeseries(model, input_tensor, target_layer=None, target_class=None):
    """Calculate Grad-CAM relevance map for time series data.
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor (batch, channels, time_steps)
        target_layer: Target layer for Grad-CAM (usually the last convolutional layer)
        target_class: Target class index (None for argmax)
        
    Returns:
        Grad-CAM heatmap (batch, time_steps)
    """
    # Special handling for time series data (1D)
    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam.attribute(input_tensor, target=target_class)
    
    # Convert to numpy array
    if isinstance(cam, torch.Tensor):
        cam = cam.detach().cpu().numpy()
        
    # Remove batch dimension if single example
    if cam.shape[0] == 1:
        cam = cam[0]
        
    # Handle dimensionality to make it compatible with 1D time series output
    if cam.ndim == 3:  # (batch, channel, time)
        # Average across channels if multiple channels
        if cam.shape[1] > 1:
            cam = np.mean(cam, axis=1)
        else:
            # Squeeze out the channel dimension if only one channel
            cam = np.squeeze(cam, axis=1)
            
    return cam