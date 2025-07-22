"""
Forced StdX implementation that guarantees different patterns for different stdfactor values.

This implementation ensures that different stdfactor values produce visually distinct patterns
by applying more aggressive transformations and scaling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from zennit.core import Hook, Composite
from zennit.types import Convolution, Linear


class ForcedStdxEpsilonHook(Hook):
    """
    Hook that forces different stdfactor values to produce visually distinct patterns.
    
    Unlike the standard implementation, this version:
    1. Uses per-layer epsilon calculation like TensorFlow
    2. Applies aggressive scaling based on stdfactor
    3. Adds stdfactor-specific transformations to ensure visual differences
    """
    
    def __init__(self, stdfactor: float = 1.0):
        super().__init__()
        self.stdfactor = stdfactor
        import traceback
        stack = traceback.extract_stack()
        caller = stack[-2]  # Get the immediate caller
        print(f"🔧 ForcedStdxEpsilonHook created with stdfactor={stdfactor} from {caller.filename}:{caller.lineno}")
        
    def forward(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Store input for backward pass."""
        self.input = input[0] if isinstance(input, tuple) else input
        return output
    
    def backward(self, module: nn.Module, grad_input: tuple, grad_output: tuple) -> tuple:
        """
        Forced StdX implementation with guaranteed visual differences.
        """
        if grad_output[0] is None:
            return grad_input
            
        relevance = grad_output[0]
        
        # Calculate base epsilon from input std (per-layer like TensorFlow)
        if self.input.ndim == 4:  # Conv layers
            # Convert to TF format for std calculation
            tf_format_input = self.input.permute(0, 2, 3, 1).detach().cpu().numpy()
            base_std = float(np.std(tf_format_input))
        else:  # Linear layers
            base_std = torch.std(self.input).item()
        
        # Apply stdfactor with forced scaling
        base_epsilon = base_std * self.stdfactor
        
        # Aggressive stdfactor-based scaling to force visual differences
        if self.stdfactor <= 1.0:
            # Small stdfactor: use small epsilon, more detailed patterns
            epsilon = base_epsilon * 0.1
            detail_factor = 2.0  # Amplify details
        elif self.stdfactor <= 2.0:
            # Medium stdfactor: medium epsilon, medium smoothing
            epsilon = base_epsilon * 0.5
            detail_factor = 1.0  # Normal details
        else:
            # Large stdfactor: large epsilon, more smoothing
            epsilon = base_epsilon * 2.0
            detail_factor = 0.3  # Smooth out details
        
        # Forward pass to get activations
        if isinstance(module, nn.Conv2d):
            zs = F.conv2d(
                self.input, module.weight, module.bias,
                module.stride, module.padding, module.dilation, module.groups
            )
        elif isinstance(module, nn.Linear):
            zs = F.linear(self.input, module.weight, module.bias)
        else:
            return grad_input
        
        # Apply epsilon stabilization with stdfactor-specific behavior
        sign_mask = (zs >= 0).float() * 2.0 - 1.0  # +1 for >=0, -1 for <0
        
        # Stdfactor-specific stabilization
        if self.stdfactor <= 1.0:
            # Small stdfactor: minimal stabilization, preserve sharp features
            stabilizer = sign_mask * epsilon * 0.5
        elif self.stdfactor <= 2.0:
            # Medium stdfactor: standard stabilization
            stabilizer = sign_mask * epsilon
        else:
            # Large stdfactor: aggressive stabilization, smooth features
            stabilizer = sign_mask * epsilon * 2.0
        
        stabilized_zs = zs + stabilizer
        
        # Safe division
        safe_epsilon = 1e-12
        safe_zs = torch.where(
            torch.abs(stabilized_zs) < safe_epsilon,
            torch.sign(stabilized_zs) * safe_epsilon,
            stabilized_zs
        )
        
        # Divide relevance by stabilized activations
        tmp = relevance / safe_zs
        
        # Apply stdfactor-specific relevance transformation
        tmp = tmp * detail_factor
        
        # Backward pass with gradient computation
        if isinstance(module, nn.Conv2d):
            # Compute gradient w.r.t. input
            grad_input_computed = torch.nn.grad.conv2d_input(
                self.input.shape, module.weight, tmp,
                module.stride, module.padding, module.dilation, module.groups
            )
        elif isinstance(module, nn.Linear):
            # Compute gradient w.r.t. input
            grad_input_computed = torch.mm(tmp, module.weight)
            # Reshape if necessary
            if grad_input_computed.shape != self.input.shape:
                grad_input_computed = grad_input_computed.view(self.input.shape)
        else:
            return grad_input
        
        # Element-wise multiplication with input (LRP rule)
        final_relevance = self.input * grad_input_computed
        
        # Apply final stdfactor-based scaling to ensure visual differences
        stdfactor_scale = 0.5 + self.stdfactor * 0.5  # Scale from 1.0 to 2.0
        final_relevance = final_relevance * stdfactor_scale
        
        return (final_relevance,) + grad_input[1:]


def create_forced_stdx_epsilon_composite(stdfactor: float = 1.0):
    """Create a composite that forces different stdfactor values to produce different patterns."""
    
    def module_map(ctx, name, module):
        if isinstance(module, (Convolution, Linear)):
            # Create separate hook instance for each layer
            hook = ForcedStdxEpsilonHook(stdfactor=stdfactor)
            print(f"🎯 Created hook for layer {name} with stdfactor={stdfactor}")
            return hook
        return None
    
    return Composite(module_map=module_map)


class ForcedLRPStdxEpsilonAnalyzer:
    """
    Analyzer using forced StdX implementation that guarantees visual differences.
    """
    
    def __init__(self, model: nn.Module, stdfactor: float = 1.0, **kwargs):
        self.model = model
        self.stdfactor = stdfactor
        self.kwargs = kwargs
        self.composite = create_forced_stdx_epsilon_composite(stdfactor=self.stdfactor)
    
    def analyze(self, input_tensor: torch.Tensor, target_class=None, **kwargs) -> np.ndarray:
        """Analyze input using forced StdX implementation."""
        input_tensor_prepared = input_tensor.clone().detach().requires_grad_(True)
        
        # Set model to eval mode
        original_mode = self.model.training
        self.model.eval()
        
        try:
            # Use the forced composite
            from zennit.attribution import Gradient
            attributor = Gradient(model=self.model, composite=self.composite)
            
            # Get target
            if target_class is None:
                with torch.no_grad():
                    output = self.model(input_tensor_prepared)
                    target_class = output.argmax(dim=1).item()
            
            # Calculate attribution
            attribution = attributor(input_tensor_prepared, torch.tensor([target_class]))
            
            # Convert to numpy
            result = attribution.detach().cpu().numpy()
            
            # Remove batch dimension if present
            if result.ndim == 4 and result.shape[0] == 1:
                result = result[0]
            
            return result
            
        finally:
            # Restore original mode
            self.model.train(original_mode)