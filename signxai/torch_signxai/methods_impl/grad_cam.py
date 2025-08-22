"""Unified PyTorch implementation of Grad-CAM combining the best features from both implementations."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Optional, Tuple, List


class GradCAM:
    """Unified Grad-CAM implementation for PyTorch models.
    
    Combines the automatic layer detection from gradcam.py with the 
    TensorFlow-compatible behavior from grad_cam.py.
    
    Grad-CAM uses the gradients of a target concept flowing into the final
    convolutional layer to produce a coarse localization map highlighting
    important regions in the image for prediction.
    """
    
    def __init__(self, model, target_layer=None):
        """Initialize GradCAM.
        
        Args:
            model: PyTorch model
            target_layer: Target layer for Grad-CAM. If None, will try to 
                         automatically find the last convolutional layer.
        """
        # Wrap model to avoid inplace operations if needed
        self.model = self._wrap_model_no_inplace(model)
        self.original_model = model  # Keep reference to original
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        # If target_layer is not provided, try to find the last convolutional layer
        if self.target_layer is None:
            self.target_layer = self._find_target_layer(self.model)
            
        # Check if target_layer was found or provided
        if self.target_layer is None:
            raise ValueError("Could not automatically identify a target convolutional layer. "
                            "Please specify one explicitly.")
        
        # Register hooks
        self._register_hooks()
    
    def _wrap_model_no_inplace(self, model):
        """Wrap model to replace inplace operations that can cause issues with hooks."""
        import copy
        
        class NoInplaceWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                # Use deepcopy to avoid modifying the original model
                self.model = copy.deepcopy(model)
                self._replace_inplace_relu(self.model)
            
            def _replace_inplace_relu(self, module):
                """Recursively replace inplace ReLUs."""
                for name, child in module.named_children():
                    if isinstance(child, nn.ReLU) and child.inplace:
                        setattr(module, name, nn.ReLU(inplace=False))
                    else:
                        self._replace_inplace_relu(child)
            
            def forward(self, x):
                return self.model(x)
            
            def __getattr__(self, name):
                # Delegate attribute access to the wrapped model
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    return getattr(self.model, name)
        
        # Check if model has inplace operations
        has_inplace = False
        
        def check_inplace(module):
            nonlocal has_inplace
            for child in module.children():
                if isinstance(child, nn.ReLU) and child.inplace:
                    has_inplace = True
                    return
                check_inplace(child)
        
        check_inplace(model)
        
        # Only wrap if there are inplace operations
        if has_inplace:
            return NoInplaceWrapper(model)
        return model
    
    def _find_target_layer(self, model):
        """Find the last convolutional layer in the model.
        
        This method searches for Conv2d (images) and Conv1d (time series) layers.
        """
        target_layer = None
        
        # Special handling for known architectures
        if hasattr(model, 'layer4'):
            # ResNet-like models
            return model.layer4[-1].conv2
        elif hasattr(model, 'features'):
            # VGG-like models
            for i in range(len(model.features) - 1, -1, -1):
                if isinstance(model.features[i], (nn.Conv2d, nn.Conv1d)):
                    return model.features[i]
        
        # Generic search for the last conv layer
        last_conv = None
        
        def search_conv(module):
            nonlocal last_conv
            for m in module.children():
                if len(list(m.children())) > 0:
                    # Recurse into submodules
                    search_conv(m)
                elif isinstance(m, (nn.Conv2d, nn.Conv1d)):
                    last_conv = m
        
        search_conv(model)
        return last_conv
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        # Clear any existing hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        def forward_hook(module, input, output):
            # Clone to avoid inplace modification issues
            self.activations = output.clone().detach()
        
        def backward_hook(module, grad_input, grad_output):
            # Clone to avoid inplace modification issues
            self.gradients = grad_output[0].clone().detach()
        
        # Register hooks
        forward_handle = self.target_layer.register_forward_hook(forward_hook)
        
        # Use register_full_backward_hook for newer PyTorch, fallback for older versions
        try:
            backward_handle = self.target_layer.register_full_backward_hook(backward_hook)
        except AttributeError:
            backward_handle = self.target_layer.register_backward_hook(backward_hook)
            
        self.hooks.extend([forward_handle, backward_handle])
    
    def forward(self, x, target_class=None):
        """Generate Grad-CAM attribution map using the TensorFlow-compatible approach.
        
        Args:
            x: Input tensor
            target_class: Target class index (None for argmax)
            
        Returns:
            Grad-CAM attribution map
        """
        # Set model to eval mode
        original_mode = self.model.training
        self.model.eval()
        
        # Clone input to avoid modifying the original
        x = x.clone().detach().requires_grad_(True)
        
        # Reset stored activations and gradients
        self.activations = None
        self.gradients = None
        
        # Forward pass
        self.model.zero_grad()
        output = self.model(x)
        
        # Select target class
        if target_class is None:
            target_class = output.argmax(dim=1)
        elif isinstance(target_class, (int, np.integer)):
            target_class = torch.tensor([int(target_class)], device=output.device)
        elif isinstance(target_class, (list, tuple)):
            target_class = torch.tensor(target_class, device=output.device)
        elif isinstance(target_class, np.ndarray):
            target_class = torch.from_numpy(target_class).to(output.device)
        
        # Create one-hot encoding
        if output.dim() == 2:  # Batch output
            one_hot = torch.zeros_like(output)
            if target_class.dim() == 0:  # Single value
                target_class = target_class.unsqueeze(0)
            one_hot.scatter_(1, target_class.unsqueeze(1), 1.0)
        else:  # Single output
            one_hot = torch.zeros_like(output)
            if target_class.dim() > 0:
                target_class = target_class[0]
            one_hot[target_class] = 1.0
        
        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Ensure we have activations and gradients
        if self.activations is None or self.gradients is None:
            raise ValueError("Could not capture activations or gradients. "
                            "Check that the target layer is correct.")
        
        # Calculate weights - match TensorFlow's reduce_mean behavior
        if self.gradients.dim() == 4:  # For images (B, C, H, W)
            weights = torch.mean(self.gradients, dim=(0, 2, 3), keepdim=False)
        else:  # For time series (B, C, T)
            weights = torch.mean(self.gradients, dim=(0, 2), keepdim=False)
        
        # Extract first sample's activations (match TensorFlow behavior)
        activations = self.activations[0]  # Remove batch dimension
        
        # Weight activations by importance
        if activations.dim() == 3:  # (C, H, W)
            weighted_output = activations * weights[:, None, None]
        else:  # (C, T)
            weighted_output = activations * weights[:, None]
        
        # Sum across feature map channels
        cam = torch.sum(weighted_output, dim=0, keepdim=False)
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        
        # TensorFlow-style normalization
        epsilon = 1e-7
        cam = cam / (torch.max(cam) + epsilon)
        
        # Don't resize here - we'll resize in calculate_grad_cam_relevancemap
        # This avoids double resizing and keeps the raw CAM output
        pass
        
        # Clean up hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # Restore model mode
        self.model.train(original_mode)
        
        return cam
    
    def attribute(self, inputs, target=None, resize_to_input=True):
        """Generate Grad-CAM heatmap (compatible with gradcam.py interface).
        
        Args:
            inputs: Input tensor
            target: Target class index (None for argmax)
            resize_to_input: Whether to resize heatmap to input size
            
        Returns:
            Grad-CAM heatmap (same size as input if resize_to_input=True)
        """
        # Handle tensor conversion
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float32)
            
        # Use the forward method
        cam = self.forward(inputs, target_class=target)
        
        # Handle batch dimension for compatibility
        if inputs.dim() == 4 and cam.dim() == 2:  # Batch input, single CAM
            cam = cam.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            if inputs.shape[0] > 1:
                # Repeat for batch size
                cam = cam.repeat(inputs.shape[0], 1, 1, 1)
        
        return cam


def calculate_grad_cam_relevancemap(model, input_tensor, target_layer=None, target_class=None, layer_name=None, **kwargs):
    """Calculate Grad-CAM relevance map for images.
    
    This function provides a convenient interface compatible with grad_cam.py.
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor
        target_layer: Target layer for Grad-CAM (None to auto-detect)
        target_class: Target class index (None for argmax)
        layer_name: Alternative name for target_layer (for compatibility)
        **kwargs: Additional parameters (ignored)
        
    Returns:
        Grad-CAM relevance map as numpy array
    """
    # Use layer_name if target_layer not provided (for compatibility)
    if target_layer is None and layer_name is not None:
        # Get the actual layer from the model using the layer name
        # Handle both dot notation and indexed access
        parts = layer_name.split('.')
        target_layer = model
        for part in parts:
            if part.isdigit():
                # Numeric index
                target_layer = target_layer[int(part)]
            else:
                # Attribute access
                target_layer = getattr(target_layer, part)
    
    # Initialize Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # Generate attribution map
    with torch.enable_grad():
        cam = grad_cam.forward(input_tensor, target_class)
    
    # Convert to numpy and handle dimensions
    if isinstance(cam, torch.Tensor):
        cam = cam.detach().cpu().numpy()
        
    # Resize CAM to input size for consistency with TensorFlow
    input_shape = input_tensor.shape
    if input_tensor.dim() == 4:  # Batch input (B, C, H, W)
        target_size = (input_shape[2], input_shape[3])  # (H, W)
    else:  # Single input (C, H, W)
        target_size = (input_shape[1], input_shape[2])  # (H, W)
    
    # Resize if necessary
    if cam.shape[-2:] != target_size:
        try:
            import cv2
            # cv2 expects (W, H) for resize
            cv2_size = (target_size[1], target_size[0])
            if cam.ndim == 2:
                cam = cv2.resize(cam, cv2_size)
            elif cam.ndim == 3:  # Batch of CAMs
                resized_cams = []
                for i in range(cam.shape[0]):
                    resized_cams.append(cv2.resize(cam[i], cv2_size))
                cam = np.stack(resized_cams, axis=0)
        except ImportError:
            # Fallback to scipy if cv2 is not available
            from scipy.ndimage import zoom
            if cam.ndim == 2:
                zoom_factors = (target_size[0] / cam.shape[0], target_size[1] / cam.shape[1])
                cam = zoom(cam, zoom_factors, order=1)
            elif cam.ndim == 3:
                zoom_factors = (1, target_size[0] / cam.shape[1], target_size[1] / cam.shape[2])
                cam = zoom(cam, zoom_factors, order=1)
    
    # Handle batch dimension if present
    if hasattr(input_tensor, 'dim') and input_tensor.dim() == 4:  # Batch
        # Return with batch dimension
        if cam.ndim == 2:  # Single CAM without batch
            cam = np.expand_dims(cam, axis=0)
    else:  # Single input
        # Remove any extra dimensions if input is single
        if cam.ndim == 3 and cam.shape[0] == 1:
            cam = cam[0]
    
    return cam


def calculate_grad_cam_relevancemap_timeseries(model, input_tensor, target_layer=None, target_class=None):
    """Calculate Grad-CAM relevance map for time series data.
    
    This function provides compatibility with grad_cam.py's timeseries function.
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor (B, C, T)
        target_layer: Target layer for Grad-CAM (None to auto-detect)
        target_class: Target class index (None for argmax)
        
    Returns:
        Grad-CAM relevance map as numpy array
    """
    # Find the last conv1d layer if not specified
    if target_layer is None:
        for module in reversed(list(model.modules())):
            if isinstance(module, nn.Conv1d):
                target_layer = module
                break
    
    if target_layer is None:
        raise ValueError("Could not find Conv1d layer for time series Grad-CAM")
    
    # Use the unified implementation
    return calculate_grad_cam_relevancemap(model, input_tensor, target_layer, target_class)


# Aliases for backward compatibility
find_target_layer = lambda model: GradCAM(model)._find_target_layer(model)