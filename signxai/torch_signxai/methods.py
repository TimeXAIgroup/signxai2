# signxai/torch_signxai/methods.py
"""
Refactored PyTorch explanation methods with a unified execution entry point.
This module applies DRY principles to eliminate redundant wrapper functions.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List

from signxai.torch_signxai.methods_impl.zennit_impl import (
    GradientAnalyzer, IntegratedGradientsAnalyzer, SmoothGradAnalyzer,
    GuidedBackpropAnalyzer, GradCAMAnalyzer, LRPAnalyzer, AdvancedLRPAnalyzer,
    LRPSequential, BoundedLRPAnalyzer, DeepLiftAnalyzer, LRPStdxEpsilonAnalyzer
)
from signxai.torch_signxai.methods_impl.signed import calculate_sign_mu
from signxai.torch_signxai.methods_impl.grad_cam import calculate_grad_cam_relevancemap, calculate_grad_cam_relevancemap_timeseries
from signxai.common.method_parser import MethodParser
from signxai.common.method_normalizer import MethodNormalizer

# A registry to map base method names to their core implementation functions.
METHOD_IMPLEMENTATIONS = {}


def register_method(name):
    """Decorator to register a method implementation."""

    def decorator(func):
        METHOD_IMPLEMENTATIONS[name] = func
        return func

    return decorator


# --- Core Method Implementations ---

@register_method("gradient")
def _gradient(model, x, **kwargs):
    analyzer = GradientAnalyzer(model)
    return analyzer.analyze(x, kwargs.get('target_class'))


@register_method("smoothgrad")
def _smoothgrad(model, x, **kwargs):
    params = {**MethodNormalizer.METHOD_PRESETS['smoothgrad'], **kwargs}
    analyzer = SmoothGradAnalyzer(model, params['noise_level'], params['num_samples'])
    return analyzer.analyze(x, kwargs.get('target_class'))


@register_method("integrated_gradients")
def _integrated_gradients(model, x, **kwargs):
    params = {**MethodNormalizer.METHOD_PRESETS['integrated_gradients'], **kwargs}
    analyzer = IntegratedGradientsAnalyzer(model, params['steps'], params.get('baseline'))
    return analyzer.analyze(x, kwargs.get('target_class'))


@register_method("guided_backprop")
def _guided_backprop(model, x, **kwargs):
    analyzer = GuidedBackpropAnalyzer(model)
    return analyzer.analyze(x, kwargs.get('target_class'))


@register_method("deconvnet")
def _deconvnet(model, x, **kwargs):
    from signxai.torch_signxai.methods_impl.zennit_impl.analyzers import DeconvNetAnalyzer
    analyzer = DeconvNetAnalyzer(model)
    return analyzer.analyze(x, kwargs.get('target_class'))


@register_method("grad_cam")
def _grad_cam(model, x, **kwargs):
    # Handle both timeseries and image gradcam
    if x.dim() <= 3:  # Assuming timeseries (B, C, T) or (C, T)
        return calculate_grad_cam_relevancemap_timeseries(model, x, **kwargs)
    else:
        return calculate_grad_cam_relevancemap(model, x, **kwargs)


@register_method("lrp")
def _lrp(model, x, **kwargs):
    """
    A unified LRP implementation that handles different rules based on kwargs.
    This replaces dozens of individual LRP wrapper functions.
    """
    # Default to epsilon rule if no specific rule is provided
    rule_name = kwargs.get('rule', 'epsilon')

    # Extract common LRP parameters
    epsilon = kwargs.get('epsilon', 1e-6)
    alpha = kwargs.get('alpha', 1.0)
    beta = kwargs.get('beta', 0.0)

    # More complex rules can be handled here
    if rule_name == 'epsilon':
        analyzer = LRPAnalyzer(model, 'epsilon', epsilon)
    elif rule_name == 'zplus':
        analyzer = LRPAnalyzer(model, 'zplus')
    elif rule_name == 'alphabeta':
        analyzer = LRPAnalyzer(model, 'alphabeta', alpha=alpha, beta=beta)
    # Add other advanced LRP rules from AdvancedLRPAnalyzer as needed
    else:
        # Fallback to AdvancedLRPAnalyzer for more complex, named rules
        analyzer = AdvancedLRPAnalyzer(model, rule_name, **kwargs)

    return analyzer.analyze(x, kwargs.get('target_class'))


# --- Modifier Application ---

def _apply_modifiers(relevance_map: np.ndarray, x: np.ndarray, modifiers: List[str],
                     params: Dict[str, Any]) -> np.ndarray:
    """
    Applies a chain of modifiers to a relevance map.
    """
    if not modifiers:
        return relevance_map

    # Make a copy to avoid in-place modification
    modified_map = relevance_map.copy()

    # Ensure x is a numpy array for calculations
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    for modifier in modifiers:
        if modifier == 'x_input' or modifier == 'input':
            modified_map *= x
        elif modifier == 'x_sign' or modifier == 'sign':
            # Check if there's a mu parameter for sign
            if 'mu' in params:
                mu = params.get('mu', 0.0)
                # Debug output
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Applying sign with mu={mu}")
                modified_map *= calculate_sign_mu(x, mu)
            else:
                s = np.nan_to_num(x / np.abs(x), nan=1.0)
                modified_map *= s
        elif modifier == 'std_x':
            # Standard deviation normalization
            pass  # Implementation needed if used

    return modified_map


# --- Main Execution Function ---

def execute(model: nn.Module, x: torch.Tensor, parsed_method: Dict[str, Any], **kwargs) -> np.ndarray:
    """
    Executes the specified XAI method after parsing and normalization.

    Args:
        model: The PyTorch model.
        x: The input tensor.
        parsed_method: A dictionary from MethodParser.
        **kwargs: Additional runtime keyword arguments.

    Returns:
        The explanation map as a numpy array.
    """
    base_method = MethodNormalizer.normalize(parsed_method['base_method'], 'pytorch')

    # Combine parameters from all sources: parser, kwargs, and presets
    all_params = {
        **MethodNormalizer.METHOD_PRESETS.get(base_method, {}),
        **parsed_method['params'],
        **kwargs
    }

    # Find and execute the core implementation
    if base_method not in METHOD_IMPLEMENTATIONS:
        # Fallback for LRP methods not explicitly registered (e.g., lrp_epsilon)
        if base_method.startswith('lrp'):
            base_method = 'lrp'
            # Pass original name to let the LRP handler decide the rule
            all_params['rule'] = parsed_method['original_name']
        else:
            raise ValueError(f"Method '{base_method}' is not implemented for PyTorch.")

    core_method_func = METHOD_IMPLEMENTATIONS[base_method]

    # Prepare input tensor (add batch dim if necessary)
    input_tensor = x.clone()
    needs_batch_dim = input_tensor.dim() < 4  # Simple heuristic for images/timeseries
    if needs_batch_dim:
        input_tensor = input_tensor.unsqueeze(0)

    model.eval()

    relevance_map_tensor = core_method_func(model, input_tensor, **all_params)

    # Convert to numpy and remove batch dimension if it was added
    if isinstance(relevance_map_tensor, torch.Tensor):
        relevance_map_np = relevance_map_tensor.detach().cpu().numpy()
    else:
        # Already numpy array (from method families)
        relevance_map_np = relevance_map_tensor
        
    if needs_batch_dim and relevance_map_np.shape[0] == 1:
        relevance_map_np = relevance_map_np[0]

    # Apply modifiers
    final_map = _apply_modifiers(relevance_map_np, x, parsed_method['modifiers'], all_params)

    return final_map
