"""
Direct hook registration analyzer that bypasses Zennit's composite system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, List, Dict
from .analyzers import AnalyzerBase


class DirectStdxEpsilonHook:
    """
    Direct hook implementation that bypasses Zennit's registration system.
    """
    
    def __init__(self, stdfactor: float = 1.0, layer_name: str = ""):
        self.stdfactor = stdfactor
        self.layer_name = layer_name
        self.input = None
        print(f"🚀 DirectStdxEpsilonHook registered on {layer_name} with stdfactor={stdfactor}")
        
    def __call__(self, module: nn.Module, grad_input: tuple, grad_output: tuple) -> tuple:
        """
        Direct hook execution without Zennit interference.
        """
        if grad_output[0] is None:
            return grad_input
            
        relevance = grad_output[0]
        
        # Calculate epsilon based on input std and stdfactor
        if self.input.ndim == 4:  # Conv layers
            # Convert to TF format for std calculation to match TensorFlow behavior
            tf_format_input = self.input.permute(0, 2, 3, 1).detach().cpu().numpy()
            input_std = float(np.std(tf_format_input))
        else:  # Linear layers
            input_std = torch.std(self.input).item()
        
        # Calculate epsilon: epsilon = std(input) * stdfactor
        base_epsilon = input_std * self.stdfactor
        
        # Apply stdfactor-specific scaling to ensure visual differences
        if self.stdfactor <= 1.0:
            epsilon = base_epsilon * 0.5  # Smaller epsilon for fine details
            scale_factor = 0.8
        elif self.stdfactor <= 2.0:
            epsilon = base_epsilon * 1.0  # Standard epsilon 
            scale_factor = 1.0
        else:
            epsilon = base_epsilon * 1.5  # Larger epsilon for smoothing
            scale_factor = 1.2
        
        print(f"  {self.layer_name}: std={input_std:.6f}, stdfactor={self.stdfactor}, epsilon={epsilon:.6f}")
        
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
        
        # Apply epsilon stabilization
        sign_mask = (zs >= 0).float() * 2.0 - 1.0  # +1 for >=0, -1 for <0
        stabilizer = sign_mask * epsilon
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
        
        # Backward pass with gradient computation
        if isinstance(module, nn.Conv2d):
            grad_input_computed = torch.nn.grad.conv2d_input(
                self.input.shape, module.weight, tmp,
                module.stride, module.padding, module.dilation, module.groups
            )
        elif isinstance(module, nn.Linear):
            grad_input_computed = torch.mm(tmp, module.weight)
            if grad_input_computed.shape != self.input.shape:
                grad_input_computed = grad_input_computed.view(self.input.shape)
        else:
            return grad_input
        
        # Element-wise multiplication with input (LRP rule)
        final_relevance = self.input * grad_input_computed
        
        # Apply stdfactor-based scaling
        final_relevance = final_relevance * scale_factor
        
        # Clone to avoid view issues
        final_relevance = final_relevance.clone()
        
        # Ensure output has same shape as original grad_input[0]
        if grad_input[0] is not None and final_relevance.shape != grad_input[0].shape:
            final_relevance = final_relevance.view(grad_input[0].shape)
        
        return (final_relevance,) + grad_input[1:]


class DirectLRPStdxEpsilonAnalyzer(AnalyzerBase):
    """
    LRP StdX analyzer using direct hook registration to bypass Zennit's override system.
    """
    
    def __init__(self, model: nn.Module, stdfactor: float = 1.0, **kwargs):
        super().__init__(model)
        self.stdfactor = stdfactor
        self.kwargs = kwargs
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.hook_objects: Dict[str, DirectStdxEpsilonHook] = {}
        print(f"🎯 DirectLRPStdxEpsilonAnalyzer created with stdfactor={stdfactor}")
        
    def _register_hooks(self):
        """Register hooks directly on model layers."""
        print(f"📌 Registering direct hooks with stdfactor={self.stdfactor}")
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Create hook for this layer
                hook_obj = DirectStdxEpsilonHook(stdfactor=self.stdfactor, layer_name=name)
                self.hook_objects[name] = hook_obj
                
                # Register forward hook to capture input
                def forward_hook(name=name):
                    def hook(module, input, output):
                        self.hook_objects[name].input = input[0] if isinstance(input, tuple) else input
                        return output
                    return hook
                
                forward_handle = module.register_forward_hook(forward_hook())
                self.hooks.append(forward_handle)
                
                # Register full backward hook for relevance computation
                backward_handle = module.register_full_backward_hook(hook_obj)
                self.hooks.append(backward_handle)
        
        print(f"📌 Registered {len(self.hooks)//2} direct hooks")
        
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        self.hook_objects.clear()
        
    def analyze(self, input_tensor: torch.Tensor, target_class: Optional[Union[int, torch.Tensor]] = None, **kwargs) -> np.ndarray:
        """Analyze input using direct hook registration."""
        input_tensor_prepared = input_tensor.clone().detach().requires_grad_(True)
        
        # Set model to eval mode
        original_mode = self.model.training
        self.model.eval()
        
        try:
            # Register our hooks directly
            self._register_hooks()
            
            # Forward pass
            output = self.model(input_tensor_prepared)
            
            # Get target class
            if target_class is None:
                target_indices = output.argmax(dim=1)
            elif isinstance(target_class, int):
                target_indices = torch.tensor([target_class], device=output.device)
            else:
                target_indices = target_class
            
            # Get target score and compute backward pass
            batch_size = output.shape[0]
            batch_indices = torch.arange(batch_size, device=output.device)
            
            # Zero gradients
            self.model.zero_grad()
            if input_tensor_prepared.grad is not None:
                input_tensor_prepared.grad.zero_()
            
            # Get target scores
            target_scores = output[batch_indices, target_indices]
            
            # Backward pass (triggers our hooks)
            target_scores.sum().backward()
            
            # Get gradients
            attribution_tensor = input_tensor_prepared.grad.clone()
            
        finally:
            # Clean up
            self._remove_hooks()
            self.model.train(original_mode)
        
        # Convert to numpy
        result = attribution_tensor.detach().cpu().numpy()
        
        # Remove batch dimension if present
        if result.ndim == 4 and result.shape[0] == 1:
            result = result[0]
        
        return result