# signxai/tf_signxai/methods.py
"""
Refactored TensorFlow explanation methods with a unified execution entry point.
"""
import numpy as np
import tensorflow as tf
from typing import Dict, Any, List

from signxai.tf_signxai.methods_impl.grad_cam import calculate_grad_cam_relevancemap, calculate_grad_cam_relevancemap_timeseries
from signxai.tf_signxai.methods_impl.guided_backprop import guided_backprop_on_guided_model
from signxai.tf_signxai.methods_impl.signed import calculate_sign_mu
from signxai.utils.utils import calculate_explanation_innvestigate
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
    return calculate_explanation_innvestigate(model, x, method='gradient', **kwargs)


@register_method("smoothgrad")
def _smoothgrad(model, x, **kwargs):
    params = {**MethodNormalizer.METHOD_PRESETS['smoothgrad'], **kwargs}
    return calculate_explanation_innvestigate(model, x, method='smoothgrad', **params)


@register_method("integrated_gradients")
def _integrated_gradients(model, x, **kwargs):
    params = {**MethodNormalizer.METHOD_PRESETS['integrated_gradients'], **kwargs}
    return calculate_explanation_innvestigate(model, x, method='integrated_gradients', **params)


@register_method("guided_backprop")
def _guided_backprop(model, x, **kwargs):
    return calculate_explanation_innvestigate(model, x, method='guided_backprop', **kwargs)


@register_method("deconvnet")
def _deconvnet(model, x, **kwargs):
    return calculate_explanation_innvestigate(model, x, method='deconvnet', **kwargs)


@register_method("grad_cam")
def _grad_cam(model, x, **kwargs):
    if x.ndim <= 3:  # Assuming timeseries
        return calculate_grad_cam_relevancemap_timeseries(x, model, **kwargs)
    else:
        return calculate_grad_cam_relevancemap(x, model, **kwargs)


@register_method("lrp")
def _lrp(model, x, **kwargs):
    """
    Unified LRP implementation for TensorFlow using iNNvestigate.
    """
    # Extract rule parameter and remove it from kwargs to avoid passing it to iNNvestigate
    rule = kwargs.pop('rule', 'epsilon')
    
    # Handle rule names that contain the full method name
    if rule.startswith('lrp'):
        # Extract just the rule part (e.g., 'lrp_epsilon_50_x_sign' -> 'epsilon')
        rule_parts = rule.split('_')
        if len(rule_parts) > 1:
            rule = rule_parts[1]  # Get the actual rule name
    
    # iNNvestigate uses dot notation for LRP methods
    method_name = f"lrp.{rule}"
    return calculate_explanation_innvestigate(model, x, method=method_name, **kwargs)


# --- Modifier Application ---

def _apply_modifiers(relevance_map: np.ndarray, x: np.ndarray, modifiers: List[str],
                     params: Dict[str, Any]) -> np.ndarray:
    """
    Applies a chain of modifiers to a relevance map.
    """
    if not modifiers:
        return relevance_map

    modified_map = relevance_map.copy()

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

def execute(model, x, parsed_method: Dict[str, Any], **kwargs) -> np.ndarray:
    """
    Executes the specified XAI method for TensorFlow.
    """
    base_method = MethodNormalizer.normalize(parsed_method['base_method'], 'tensorflow')

    all_params = {
        **MethodNormalizer.METHOD_PRESETS.get(base_method, {}),
        **parsed_method['params'],
        **kwargs
    }

    if base_method not in METHOD_IMPLEMENTATIONS:
        if base_method.startswith('lrp'):
            base_method = 'lrp'
            all_params['rule'] = parsed_method['original_name'].split('.')[-1]
        else:
            raise ValueError(f"Method '{base_method}' is not implemented for TensorFlow.")

    core_method_func = METHOD_IMPLEMENTATIONS[base_method]
    
    # Prepare input - add batch dimension if necessary
    x_input = x
    # For images: (H, W, C) needs batch -> (1, H, W, C)
    # For time series: (T, F) needs batch -> (1, T, F)
    needs_batch_dim = (x.ndim == 3 and x.shape[-1] <= 4) or x.ndim == 2
    if needs_batch_dim:
        x_input = np.expand_dims(x, axis=0)

    relevance_map_np = core_method_func(model, x_input, **all_params)
    
    # Remove batch dimension if it was added
    if needs_batch_dim and relevance_map_np.ndim == 4 and relevance_map_np.shape[0] == 1:
        relevance_map_np = relevance_map_np[0]

    final_map = _apply_modifiers(relevance_map_np, x, parsed_method['modifiers'], all_params)

    return final_map
