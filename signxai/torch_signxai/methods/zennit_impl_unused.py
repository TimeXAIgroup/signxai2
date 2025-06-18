"""zennit wrapper implementations for PyTorch explanation methods."""
import torch
import numpy as np
from typing import Tuple, List, Union, Optional, Dict, Any
from zennit.core import Gradient, Hook, Composite
from zennit.composites import *
from zennit.attribution import Attributor
from zennit.types import Callable, SubsetForward

class GradientAnalyzer:
    """Vanilla gradient analyzer.
    
    Implements vanilla gradient calculation aligned with TensorFlow's implementation.
    """
    
    def __init__(self, model):
        """Initialize gradient analyzer.
        
        Args:
            model: PyTorch model
        """
        self.model = model
    
    def analyze(self, input_tensor, target_class=None):
        """Generate vanilla gradient attribution aligned with TensorFlow.
        
        Args:
            input_tensor: Input tensor
            target_class: Target class index (None for argmax)
            
        Returns:
            Gradient attribution
        """
        # Ensure input is a tensor with gradients
        if isinstance(input_tensor, torch.Tensor):
            input_tensor = input_tensor.detach().requires_grad_(True)
        else:
            input_tensor = torch.tensor(input_tensor, requires_grad=True)
        
        # Forward pass - simplified approach to match TensorFlow more directly
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # Create one-hot tensor
        if isinstance(target_class, int) or (hasattr(target_class, 'ndim') and target_class.ndim == 0):
            one_hot = torch.zeros_like(output)
            one_hot[0, target_class] = 1.0
        else:
            one_hot = torch.zeros_like(output)
            for i, cls in enumerate(target_class):
                one_hot[i, cls] = 1.0
        
        # Backward pass
        output.backward(gradient=one_hot)
        
        # Get gradients
        attribution = input_tensor.grad.clone()
        
        # Apply normalization to match TensorFlow's normalize_heatmap
        max_abs = torch.max(torch.abs(attribution))
        if max_abs > 0:
            attribution = attribution / max_abs
            
        # Return as numpy array with NaN handling
        return torch.nan_to_num(attribution).detach().cpu().numpy()


class IntegratedGradientsAnalyzer:
    """Integrated gradients analyzer.
    
    Implements the integrated gradients method by integrating gradients along a straight
    path from a baseline (typically zeros) to the input.

    This implementation is specifically aligned with the TensorFlow implementation
    to ensure consistency between frameworks.
    """
    
    def __init__(self, model, steps=50, baseline=None):
        """Initialize integrated gradients analyzer.
        
        Args:
            model: PyTorch model
            steps: Number of steps for integration
            baseline: Baseline input (None for zeros)
        """
        self.model = model
        self.steps = steps
        self.baseline = baseline
    
    def analyze(self, input_tensor, target_class=None):
        """Generate integrated gradients attribution.
        
        Args:
            input_tensor: Input tensor
            target_class: Target class index (None for argmax)
            
        Returns:
            Integrated gradients attribution
        """
        # Ensure input is a tensor and detach previous gradients
        if isinstance(input_tensor, torch.Tensor):
            input_tensor = input_tensor.detach()
        else:
            input_tensor = torch.tensor(input_tensor)
            
        # Create baseline if not provided - use zeros to match TF implementation
        if self.baseline is None:
            baseline = torch.zeros_like(input_tensor)
        else:
            baseline = self.baseline
            
        # Prepare for integration - use exactly 50 steps to match TF PathIntegrator
        step_list = torch.linspace(0, 1, self.steps, device=input_tensor.device)
        gradients = []
        
        # Get target class up front to ensure consistency across steps
        # Forward pass to determine target class if not provided
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1)
        
        # For each step - iterate exactly like TensorFlow implementation
        for step in step_list:
            # Interpolate between baseline and input - scaled_inputs approach
            current_input = baseline + step * (input_tensor - baseline)
            current_input = current_input.clone().requires_grad_(True)
            
            # Forward pass
            output = self.model(current_input)
            
            # Create one-hot tensor (consistent targeting across all steps)
            if isinstance(target_class, int) or (hasattr(target_class, 'ndim') and target_class.ndim == 0):
                one_hot = torch.zeros_like(output)
                one_hot[0, target_class] = 1.0
            else:
                one_hot = torch.zeros_like(output)
                for i, cls in enumerate(target_class):
                    one_hot[i, cls] = 1.0
            
            # Zero gradient
            self.model.zero_grad()
            if current_input.grad is not None:
                current_input.grad.zero_()
            
            # Backward pass
            output.backward(gradient=one_hot, retain_graph=True)
            
            # Store gradients (matching TF behavior)
            gradients.append(current_input.grad.detach())
        
        # Average gradients and multiply by (input - baseline)
        # This exactly matches TensorFlow's PathIntegrator implementation
        integrated_grads = torch.stack(gradients).mean(dim=0)
        attribution = integrated_grads * (input_tensor - baseline)
        
        # Optionally apply normalization here to match TF's normalize_heatmap
        # This brings the implementations even closer
        max_abs = torch.max(torch.abs(attribution))
        if max_abs > 0:
            attribution = attribution / max_abs
        
        # Return as numpy array with NaN cleaning
        return torch.nan_to_num(attribution).cpu().numpy()


class SmoothGradAnalyzer:
    """SmoothGrad analyzer.
    
    Implements SmoothGrad by adding Gaussian noise to the input multiple times and 
    averaging the resulting gradients.
    
    This implementation is aligned with TensorFlow's GaussianSmoother to ensure
    consistent results between frameworks.
    """
    
    def __init__(self, model, noise_level=0.2, num_samples=50):
        """Initialize SmoothGrad analyzer.
        
        Args:
            model: PyTorch model
            noise_level: Level of Gaussian noise to add
            num_samples: Number of noisy samples to average (augment_by_n in TF)
        """
        self.model = model
        self.noise_level = noise_level
        self.num_samples = num_samples
    
    def analyze(self, input_tensor, target_class=None):
        """Generate SmoothGrad attribution.
        
        Args:
            input_tensor: Input tensor
            target_class: Target class index (None for argmax)
            
        Returns:
            SmoothGrad attribution
        """
        # Ensure input is a tensor and detach previous gradients
        if isinstance(input_tensor, torch.Tensor):
            input_tensor = input_tensor.detach()
        else:
            input_tensor = torch.tensor(input_tensor)
        
        # Calculate standard deviation of noise - match TensorFlow exactly
        # TensorFlow uses noise_scale * (max - min) for the entire tensor
        tensor_min = input_tensor.min()
        tensor_max = input_tensor.max()
        stdev = self.noise_level * (tensor_max - tensor_min)
        
        # Forward pass to get target class if not provided
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1)
        
        # Initialize gradients accumulator
        gradients = []
        
        # Fix random seed to match TensorFlow behavior
        torch.manual_seed(42)
        
        # Calculate gradients for noisy samples - match TF augment_by_n parameter
        for _ in range(self.num_samples):
            # Add noise to input exactly like TensorFlow's GaussianSmoother
            noise = torch.normal(
                0, stdev, size=input_tensor.shape, device=input_tensor.device
            )
            noisy_input = input_tensor + noise
            noisy_input = noisy_input.clone().requires_grad_(True)
            
            # Forward pass
            output = self.model(noisy_input)
            
            # Create one-hot tensor - ensuring consistent targeting
            if isinstance(target_class, int) or (hasattr(target_class, 'ndim') and target_class.ndim == 0):
                one_hot = torch.zeros_like(output)
                one_hot[0, target_class] = 1.0
            else:
                one_hot = torch.zeros_like(output)
                for i, cls in enumerate(target_class):
                    one_hot[i, cls] = 1.0
            
            # Zero gradient
            self.model.zero_grad()
            
            # Backward pass
            output.backward(gradient=one_hot, retain_graph=True)
            
            # Store gradients
            gradients.append(noisy_input.grad.detach())
        
        # Average gradients - matching TensorFlow's approach
        smoothgrad = torch.stack(gradients).mean(dim=0)
        
        # Apply normalization to match TensorFlow's normalize_heatmap
        max_abs = torch.max(torch.abs(smoothgrad))
        if max_abs > 0:
            smoothgrad = smoothgrad / max_abs
        
        # Return as numpy array with NaN handling
        return torch.nan_to_num(smoothgrad).cpu().numpy()


class GuidedBackpropAnalyzer:
    """Guided backpropagation analyzer.
    
    Implements guided backpropagation by modifying the backward pass of ReLU
    to only pass positive gradients.
    """
    
    def __init__(self, model):
        """Initialize guided backpropagation analyzer.
        
        Args:
            model: PyTorch model
        """
        self.model = model
        self._hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks for guided backpropagation."""
        def relu_hook_function(module, grad_in, grad_out):
            # For ReLU, we only want to pass positive gradients
            if isinstance(grad_in, tuple):
                # ReLU backward returns tuple, modify first element
                # Create a clone to avoid in-place modification issues
                modified_grad = torch.clamp(grad_out[0].clone(), min=0.0)
                return (modified_grad,) + grad_in[1:]
            else:
                # Create a clone to avoid in-place modification issues
                return torch.clamp(grad_out[0].clone(), min=0.0)
        
        # Register hook for each ReLU in the model
        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                # Use register_full_backward_hook instead of register_backward_hook
                handle = module.register_full_backward_hook(relu_hook_function)
                self._hooks.append(handle)
    
    def analyze(self, input_tensor, target_class=None):
        """Generate guided backpropagation attribution.
        
        Args:
            input_tensor: Input tensor
            target_class: Target class index (None for argmax)
            
        Returns:
            Guided backpropagation attribution
        """
        # Ensure input is a tensor with gradients
        if isinstance(input_tensor, torch.Tensor):
            input_tensor = input_tensor.detach().requires_grad_(True)
        else:
            input_tensor = torch.tensor(input_tensor, requires_grad=True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # Create one-hot tensor
        if isinstance(target_class, int) or (hasattr(target_class, 'ndim') and target_class.ndim == 0):
            one_hot = torch.zeros_like(output)
            one_hot[0, target_class] = 1.0
        else:
            one_hot = torch.zeros_like(output)
            for i, cls in enumerate(target_class):
                one_hot[i, cls] = 1.0
        
        # Zero gradient
        self.model.zero_grad()
        
        # Backward pass
        output.backward(gradient=one_hot)
        
        # Get gradients
        attribution = input_tensor.grad.clone()
        
        # Return as numpy array
        return attribution.detach().cpu().numpy()
    
    def __del__(self):
        """Clean up hooks when object is deleted."""
        for handle in self._hooks:
            handle.remove()


class GradCAMAnalyzer:
    """Grad-CAM analyzer.
    
    Implements Grad-CAM by using the gradients of a target class with respect
    to feature maps of a convolutional layer to generate a coarse localization map.
    """
    
    def __init__(self, model, target_layer=None):
        """Initialize Grad-CAM analyzer.
        
        Args:
            model: PyTorch model
            target_layer: Target convolutional layer (None to auto-detect)
        """
        self.model = model
        
        # Find target layer if not provided
        if target_layer is None:
            self.target_layer = self._find_target_layer()
        else:
            self.target_layer = target_layer
            
        # Initialize hooks
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _find_target_layer(self):
        """Find the last convolutional layer in the model."""
        target_layer = None
        
        # First check common architectures
        if hasattr(self.model, 'layer4'):  # ResNet
            return self.model.layer4[-1].conv2
        elif hasattr(self.model, 'features'):  # VGG
            for layer in reversed(list(self.model.features)):
                if isinstance(layer, torch.nn.Conv2d):
                    return layer
        
        # Search for the last conv layer
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                break
                
        if target_layer is None:
            raise ValueError("Could not find a convolutional layer")
            
        return target_layer
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        
        def forward_hook(module, input, output):
            # Store a copy of the output to avoid modification issues
            self.activations = output.clone()
        
        def backward_hook(module, grad_in, grad_out):
            # Store a copy of the gradients to avoid modification issues
            self.gradients = grad_out[0].clone()
        
        # Register hooks
        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)
    
    def analyze(self, input_tensor, target_class=None):
        """Generate Grad-CAM attribution.
        
        Args:
            input_tensor: Input tensor
            target_class: Target class index (None for argmax)
            
        Returns:
            Grad-CAM attribution
        """
        # Ensure input is a tensor
        if not isinstance(input_tensor, torch.Tensor):
            input_tensor = torch.tensor(input_tensor)
            
        # Make a copy
        input_tensor = input_tensor.clone()
        
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # Create one-hot tensor
        if isinstance(target_class, int) or (hasattr(target_class, 'ndim') and target_class.ndim == 0):
            one_hot = torch.zeros_like(output)
            one_hot[0, target_class] = 1.0
        else:
            one_hot = torch.zeros_like(output)
            for i, cls in enumerate(target_class):
                one_hot[i, cls] = 1.0
        
        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Calculate weights based on global average pooling
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weight the activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # Apply ReLU
        cam = torch.nn.functional.relu(cam)
        
        # Resize to match input size
        cam = torch.nn.functional.interpolate(
            cam, 
            size=input_tensor.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        # Normalize
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)
        
        # Return as numpy array
        return cam.detach().cpu().numpy()
    
    def __del__(self):
        """Clean up hooks when object is deleted."""
        if hasattr(self, 'forward_handle'):
            self.forward_handle.remove()
        if hasattr(self, 'backward_handle'):
            self.backward_handle.remove()


class LRPAnalyzer:
    """Layer-wise Relevance Propagation (LRP) analyzer.
    
    Uses zennit's implementation of LRP with different rule variants.
    """
    
    def __init__(self, model, rule="epsilon", epsilon=1e-6):
        """Initialize LRP analyzer.
        
        Args:
            model: PyTorch model
            rule: LRP rule ('epsilon', 'zplus', 'alphabeta')
            epsilon: Stabilizing factor for epsilon rule
        """
        self.model = model
        self.rule = rule
        self.epsilon = epsilon
        
        # Map rule name to zennit composite
        if rule == "epsilon":
            self.composite = EpsilonGammaBox(epsilon=epsilon)
        elif rule == "zplus":
            self.composite = ZPlus()
        elif rule == "alphabeta":
            self.composite = AlphaBeta(alpha=1, beta=0)
        else:
            raise ValueError(f"Unknown LRP rule: {rule}")
    
    def analyze(self, input_tensor, target_class=None):
        """Generate LRP attribution.
        
        Args:
            input_tensor: Input tensor
            target_class: Target class index (None for argmax)
            
        Returns:
            LRP attribution
        """
        # Set up attributor
        attributor = Attributor(self.model, self.composite)
        
        # Ensure input is a tensor and detach previous gradients
        if isinstance(input_tensor, torch.Tensor):
            input_tensor = input_tensor.detach().requires_grad_(True)
        else:
            input_tensor = torch.tensor(input_tensor, requires_grad=True)
        
        # Forward pass
        with attributor:
            output = self.model(input_tensor)
            
            # Get target class
            if target_class is None:
                target_class = output.argmax(dim=1)
            
            # Create one-hot tensor
            if isinstance(target_class, int) or (hasattr(target_class, 'ndim') and target_class.ndim == 0):
                one_hot = torch.zeros_like(output)
                one_hot[0, target_class] = 1.0
            else:
                one_hot = torch.zeros_like(output)
                for i, cls in enumerate(target_class):
                    one_hot[i, cls] = 1.0
            
            # Get attribution
            attribution = attributor.attribute(input_tensor, output, one_hot)
        
        # Return as numpy array
        return attribution.detach().cpu().numpy()


class AdvancedLRPAnalyzer:
    """Advanced LRP analyzer with specialized rules.
    
    Extends basic LRP to support more specialized rules and composites.
    """
    
    def __init__(self, model, rule_type, **kwargs):
        """Initialize advanced LRP analyzer.
        
        Args:
            model: PyTorch model
            rule_type: Type of LRP rule/composite
            **kwargs: Additional parameters for specific rules
        """
        self.model = model
        self.rule_type = rule_type
        
        # Create appropriate composite based on rule type
        if rule_type == "alpha1beta0":
            self.composite = AlphaBeta(alpha=1, beta=0)
        elif rule_type == "alpha2beta1":
            self.composite = AlphaBeta(alpha=2, beta=1)
        elif rule_type == "epsilon":
            epsilon = kwargs.get("epsilon", 1e-6)
            self.composite = EpsilonGammaBox(epsilon=epsilon)
        elif rule_type == "gamma":
            gamma = kwargs.get("gamma", 0.25)
            self.composite = EpsilonGammaBox(gamma=gamma)
        elif rule_type == "flat":
            self.composite = Flat()
        elif rule_type == "wsquare":
            self.composite = WSquare()
        elif rule_type == "zbox":
            low = kwargs.get("low", 0.0)
            high = kwargs.get("high", 1.0)
            self.composite = ZBox(low=low, high=high)
        elif rule_type == "sequential":
            layer_rules = kwargs.get("layer_rules", {})
            # This is a simplified version, actual implementation would map rules to layers
            self.composite = EpsilonGammaBox()  # Default
        else:
            raise ValueError(f"Unknown LRP rule type: {rule_type}")
    
    def analyze(self, input_tensor, target_class=None):
        """Generate advanced LRP attribution.
        
        Args:
            input_tensor: Input tensor
            target_class: Target class index (None for argmax)
            
        Returns:
            LRP attribution
        """
        # Set up attributor
        attributor = Attributor(self.model, self.composite)
        
        # Ensure input is a tensor and detach previous gradients
        if isinstance(input_tensor, torch.Tensor):
            input_tensor = input_tensor.detach().requires_grad_(True)
        else:
            input_tensor = torch.tensor(input_tensor, requires_grad=True)
        
        # Forward pass
        with attributor:
            output = self.model(input_tensor)
            
            # Get target class
            if target_class is None:
                target_class = output.argmax(dim=1)
            
            # Create one-hot tensor
            if isinstance(target_class, int) or (hasattr(target_class, 'ndim') and target_class.ndim == 0):
                one_hot = torch.zeros_like(output)
                one_hot[0, target_class] = 1.0
            else:
                one_hot = torch.zeros_like(output)
                for i, cls in enumerate(target_class):
                    one_hot[i, cls] = 1.0
            
            # Get attribution
            attribution = attributor.attribute(input_tensor, output, one_hot)
        
        # Return as numpy array
        return attribution.detach().cpu().numpy()


class LRPSequential:
    """LRP with sequential application of different rules.
    
    Applies different LRP rules to different layers in the network.
    """
    
    def __init__(self, model, first_layer_rule="zbox", middle_layer_rule="alphabeta", 
                 last_layer_rule="epsilon", **kwargs):
        """Initialize LRP sequential analyzer.
        
        Args:
            model: PyTorch model
            first_layer_rule: Rule for first layers
            middle_layer_rule: Rule for middle layers
            last_layer_rule: Rule for last layers
            **kwargs: Additional parameters for specific rules
        """
        self.model = model
        self.first_layer_rule = first_layer_rule
        self.middle_layer_rule = middle_layer_rule
        self.last_layer_rule = last_layer_rule
        self.kwargs = kwargs
        
        # For simplicity, use a single composite 
        # A full implementation would create a custom composite with layer-specific rules
        if last_layer_rule == "epsilon":
            epsilon = kwargs.get("epsilon", 1e-6)
            self.composite = EpsilonGammaBox(epsilon=epsilon)
        elif middle_layer_rule == "alphabeta":
            alpha = kwargs.get("alpha", 1)
            beta = kwargs.get("beta", 0)
            self.composite = AlphaBeta(alpha=alpha, beta=beta)
        else:
            # Default fallback
            self.composite = EpsilonGammaBox()
    
    def analyze(self, input_tensor, target_class=None):
        """Generate sequential LRP attribution.
        
        Args:
            input_tensor: Input tensor
            target_class: Target class index (None for argmax)
            
        Returns:
            LRP attribution
        """
        # Set up attributor
        attributor = Attributor(self.model, self.composite)
        
        # Ensure input is a tensor and detach previous gradients
        if isinstance(input_tensor, torch.Tensor):
            input_tensor = input_tensor.detach().requires_grad_(True)
        else:
            input_tensor = torch.tensor(input_tensor, requires_grad=True)
        
        # Forward pass
        with attributor:
            output = self.model(input_tensor)
            
            # Get target class
            if target_class is None:
                target_class = output.argmax(dim=1)
            
            # Create one-hot tensor
            if isinstance(target_class, int) or (hasattr(target_class, 'ndim') and target_class.ndim == 0):
                one_hot = torch.zeros_like(output)
                one_hot[0, target_class] = 1.0
            else:
                one_hot = torch.zeros_like(output)
                for i, cls in enumerate(target_class):
                    one_hot[i, cls] = 1.0
            
            # Get attribution
            attribution = attributor.attribute(input_tensor, output, one_hot)
        
        # Return as numpy array
        return attribution.detach().cpu().numpy()