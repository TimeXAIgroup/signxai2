# signxai/tf_signxai/methods.py
"""
Refactored TensorFlow explanation methods with a unified execution entry point.
"""
import numpy as np
import tensorflow as tf
from typing import Dict, Any, List

from .grad_cam import calculate_grad_cam_relevancemap, calculate_grad_cam_relevancemap_timeseries
from .guided_backprop import guided_backprop_on_guided_model
from .signed import calculate_sign_mu
from ...utils.utils import calculate_explanation_innvestigate
from ...common.method_parser import MethodParser
from ...common.method_normalizer import MethodNormalizer

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
    rule = kwargs.get('rule', 'epsilon')
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
        if modifier == 'input':
            modified_map *= x
        elif modifier == 'sign':
            s = np.nan_to_num(x / np.abs(x), nan=1.0)
            modified_map *= s
        elif modifier == 'signmu':
            mu = params.get('mu', 0.0)
            modified_map *= calculate_sign_mu(x, mu)

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

    relevance_map_np = core_method_func(model, x, **all_params)

    final_map = _apply_modifiers(relevance_map_np, x, parsed_method['modifiers'], all_params)

    return final_map
