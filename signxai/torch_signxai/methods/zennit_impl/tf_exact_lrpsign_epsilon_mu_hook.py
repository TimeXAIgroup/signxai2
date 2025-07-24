"""TensorFlow-exact implementation of LRP-Sign with epsilon and mu parameters for Zennit."""

import torch
from zennit.composites import EpsilonGammaBox
from zennit.rules import Epsilon, ZBox
from zennit.core import Hook
from zennit.layer import Sum
from typing import Optional


class TFExactSignMuInputHook(Hook):
    """Hook to apply SIGN-mu rule to input layer for TensorFlow compatibility."""
    
    def __init__(self, mu: float = 0.0):
        super().__init__()
        self.mu = mu
    
    def forward(self, module, input, output):
        """Store input for backward pass."""
        self.stored_input = input[0].clone()
        return output
    
    def backward(self, module, grad_input, grad_output):
        """Apply SIGN-mu rule during backward pass with proper gradient computation."""
        if not hasattr(self, 'stored_input') or grad_output[0] is None:
            return grad_input
            
        relevance = grad_output[0]
        input_tensor = self.stored_input
        
        # Apply SIGN-mu rule to the input first (this is the key difference from epsilon)
        # SIGN-mu: threshold input at mu, then apply sign
        sign_input = torch.where(input_tensor >= self.mu, 
                               torch.ones_like(input_tensor), 
                               -torch.ones_like(input_tensor))
        
        # Use the sign-modified input for forward computation (this is the SIGN rule core)
        if isinstance(module, torch.nn.Conv2d):
            zs = torch.nn.functional.conv2d(
                sign_input, module.weight, module.bias,
                module.stride, module.padding, module.dilation, module.groups
            )
        elif isinstance(module, torch.nn.Linear):
            zs = torch.nn.functional.linear(sign_input, module.weight, module.bias)
        else:
            # For non-conv/linear layers, just pass through
            return grad_input
        
        # Apply epsilon stabilization (like in the epsilon hook)
        epsilon = 1e-7  # Small epsilon for numerical stability
        tf_sign = (zs >= 0).float() * 2.0 - 1.0
        prepare_div = zs + tf_sign * epsilon
        
        # SafeDivide
        safe_epsilon = 1e-12
        safe_prepare_div = torch.where(
            torch.abs(prepare_div) < safe_epsilon,
            torch.sign(prepare_div) * safe_epsilon,
            prepare_div
        )
        
        # Divide relevance by stabilized activations
        tmp = relevance / safe_prepare_div
        
        # Compute gradient-like operation (backward through the layer)
        if isinstance(module, torch.nn.Conv2d):
            gradient_result = torch.nn.functional.conv_transpose2d(
                tmp, module.weight, None,
                module.stride, module.padding,
                output_padding=0, groups=module.groups, dilation=module.dilation
            )
        elif isinstance(module, torch.nn.Linear):
            gradient_result = torch.mm(tmp, module.weight)
        else:
            gradient_result = tmp
            
        # Final result: sign-modified input times gradient (consistent with epsilon hook pattern)
        final_result = sign_input * gradient_result
        
        return (final_result,) + grad_input[1:]


class TFExactLRPSignEpsilonMuComposite:
    """Composite that applies TF-exact LRP SIGN Epsilon rules with mu parameter."""
    
    def __init__(self, epsilon: float = 100.0, mu: float = 0.0):
        """Initialize the composite with epsilon and mu parameters.
        
        Args:
            epsilon: Epsilon value for stability
            mu: Threshold for SIGN rule (0.0 for pure sign)
        """
        self.epsilon = epsilon
        self.mu = mu
    
    def context(self, model):
        """Apply hooks with input layer detection and mu parameter."""
        # Find the first layer that should get SIGN-mu rule
        first_meaningful_layer = None
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                first_meaningful_layer = module
                break
                
        # Apply hooks
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                is_input_layer = (module == first_meaningful_layer)
                if is_input_layer:
                    # Apply SIGN-mu hook to first layer
                    hook = TFExactSignMuInputHook(mu=self.mu)
                else:
                    # Apply epsilon hook to other layers
                    from .tf_exact_epsilon_hook import TFExactEpsilonHook
                    hook = TFExactEpsilonHook(epsilon=self.epsilon)
                
                # Register backward and forward hooks
                backward_handle = module.register_full_backward_hook(hook.backward)
                forward_handle = module.register_forward_hook(hook.forward)
                handles.extend([backward_handle, forward_handle])
                
        return _CompositeContext(model, handles)


class _CompositeContext:
    """Context manager for hook cleanup."""
    
    def __init__(self, model, handles):
        self.model = model
        self.handles = handles
        
    def __enter__(self):
        return self.model
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.handles:
            handle.remove()


def create_tf_exact_lrpsign_epsilon_mu_composite(epsilon: float = 100.0, mu: float = 0.0):
    """Create a TensorFlow-exact LRP-Sign epsilon-mu composite.
    
    Args:
        epsilon: Epsilon value for LRP
        mu: Threshold for SIGN rule
        
    Returns:
        Composite implementing TF-exact LRP-Sign with given parameters
    """
    return TFExactLRPSignEpsilonMuComposite(epsilon=epsilon, mu=mu)