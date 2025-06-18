"""
Unified API for SignXAI - Cross-framework XAI explanations.

This module provides a unified interface for generating explanations
across TensorFlow and PyTorch frameworks with automatic parameter mapping
and framework detection.
"""

import numpy as np
from typing import Union, Optional, Any, Dict


def explain(
    model,
    x: Union[np.ndarray, "torch.Tensor", "tf.Tensor"],
    method_name: str,
    target_class: Optional[int] = None,
    framework: Optional[str] = None,
    **kwargs
) -> np.ndarray:
    """
    Generate explanations for model predictions using various XAI methods.
    
    This unified API automatically handles framework detection, model preparation,
    and parameter mapping to provide consistent explanations across TensorFlow
    and PyTorch implementations.
    
    Args:
        model: The model to explain (TensorFlow Keras model or PyTorch nn.Module)
        x: Input data as numpy array or framework tensor
        method_name: Name of the XAI method to apply. Supported methods include:
            - Gradient-based: 'gradient', 'smoothgrad', 'integrated_gradients', 'vargrad'
            - Backprop methods: 'guided_backprop', 'deconvnet'
            - Feature methods: 'grad_cam'
            - LRP methods: 'lrp_epsilon', 'lrp_alpha_1_beta_0', 'lrp_alpha_2_beta_1'
            - And many more (see documentation for full list)
        target_class: Target class index for explanation. If None, uses predicted class.
        framework: Framework to use ('tensorflow' or 'pytorch'). If None, auto-detected.
        **kwargs: Method-specific parameters. Common parameters:
            - steps: Number of steps for Integrated Gradients (default: 50)
            - num_samples: Number of samples for SmoothGrad (default: 25)
            - noise_level: Noise level for SmoothGrad (default: 0.1)
            - layer_name: Target layer for Grad-CAM (framework-specific)
            - epsilon: Epsilon value for LRP methods (default: 0.1)
            - alpha, beta: Alpha-beta values for LRP alpha-beta methods
    
    Returns:
        Explanation/relevance map as numpy array with same spatial dimensions as input
        
    Raises:
        ValueError: If framework cannot be detected or is unsupported
        ImportError: If required framework dependencies are not installed
        
    Examples:
        Basic gradient explanation:
        >>> explanation = explain(model, image, 'gradient')
        
        Integrated Gradients with custom steps:
        >>> explanation = explain(model, image, 'integrated_gradients', steps=100)
        
        Grad-CAM on specific layer:
        >>> explanation = explain(model, image, 'grad_cam', layer_name='block5_conv3')
        
        LRP with epsilon rule:
        >>> explanation = explain(model, image, 'lrp_epsilon', epsilon=0.1)
        
        Cross-framework usage (same API for both):
        >>> tf_explanation = explain(tf_model, data, 'smoothgrad', framework='tensorflow')
        >>> pt_explanation = explain(pt_model, data, 'smoothgrad', framework='pytorch')
    """
    # Import here to avoid circular imports
    from . import _detect_framework, _prepare_model, _prepare_input
    from . import _get_predicted_class, _map_parameters
    from . import _call_tensorflow_method, _call_pytorch_method
    from . import _load_tf_signxai, _load_torch_signxai
    
    # Framework detection if not specified
    if framework is None:
        framework = _detect_framework(model)
        if framework is None:
            raise ValueError(
                "Could not detect framework. Please specify framework='tensorflow' or framework='pytorch'"
            )
    
    framework = framework.lower()
    if framework not in ['tensorflow', 'pytorch']:
        raise ValueError("Framework must be 'tensorflow' or 'pytorch'")
    
    # Ensure the framework is available
    if framework == 'tensorflow':
        tf_module = _load_tf_signxai()
        if tf_module is None:
            raise ImportError("TensorFlow not available. Install with: pip install signxai[tensorflow]")
    elif framework == 'pytorch':
        torch_module = _load_torch_signxai()
        if torch_module is None:
            raise ImportError("PyTorch not available. Install with: pip install signxai[pytorch]")
    
    # Prepare model (ensure no softmax for explanations)
    try:
        prepared_model = _prepare_model(model, framework)
    except Exception as e:
        print(f"Warning: Could not remove softmax from model: {e}")
        prepared_model = model
    
    # Prepare input data
    prepared_input = _prepare_input(x, framework)
    
    # Handle target class
    if target_class is None:
        target_class = _get_predicted_class(prepared_model, prepared_input, framework)
    
    # Map common parameters between frameworks
    mapped_kwargs = _map_parameters(method_name, framework, **kwargs)
    
    # Call framework-specific implementation
    if framework == 'tensorflow':
        return _call_tensorflow_method(prepared_model, prepared_input, method_name, target_class, **mapped_kwargs)
    else:  # pytorch
        return _call_pytorch_method(prepared_model, prepared_input, method_name, target_class, **mapped_kwargs)


def list_methods(framework: Optional[str] = None) -> Dict[str, list]:
    """
    List all available XAI methods for the specified framework(s).
    
    Args:
        framework: Framework to list methods for ('tensorflow', 'pytorch', or None for both)
        
    Returns:
        Dictionary with framework names as keys and list of method names as values
    """
    import inspect
    from . import _load_tf_signxai, _load_torch_signxai
    
    methods = {}
    
    if framework is None or framework.lower() == 'tensorflow':
        tf_module = _load_tf_signxai()
        if tf_module is not None:
            try:
                import signxai.tf_signxai.methods.wrappers as tf_wrappers
                tf_methods = [name for name, obj in inspect.getmembers(tf_wrappers)
                             if inspect.isfunction(obj) and not name.startswith('_') and
                             not name.startswith('calculate_native')]
                methods['tensorflow'] = sorted(tf_methods)
            except Exception as e:
                methods['tensorflow'] = f"Error loading TensorFlow methods: {e}"
    
    if framework is None or framework.lower() == 'pytorch':
        torch_module = _load_torch_signxai()
        if torch_module is not None:
            try:
                import signxai.torch_signxai.methods.wrappers as pt_wrappers
                pt_methods = [name for name, obj in inspect.getmembers(pt_wrappers)
                             if inspect.isfunction(obj) and not name.startswith('_') and
                             name not in ['calculate_relevancemap', 'calculate_relevancemaps']]
                methods['pytorch'] = sorted(pt_methods)
            except Exception as e:
                methods['pytorch'] = f"Error loading PyTorch methods: {e}"
    
    return methods


def get_method_info(method_name: str, framework: Optional[str] = None) -> Dict[str, Any]:
    """
    Get detailed information about a specific XAI method.
    
    Args:
        method_name: Name of the method to get info for
        framework: Framework to check ('tensorflow', 'pytorch', or None for both)
        
    Returns:
        Dictionary with method information including parameters, description, etc.
    """
    import inspect
    from . import _load_tf_signxai, _load_torch_signxai
    
    info = {'method_name': method_name, 'available_in': []}
    
    # Check TensorFlow
    if framework is None or framework.lower() == 'tensorflow':
        tf_module = _load_tf_signxai()
        if tf_module is not None:
            try:
                import signxai.tf_signxai.methods.wrappers as tf_wrappers
                if hasattr(tf_wrappers, method_name):
                    func = getattr(tf_wrappers, method_name)
                    info['available_in'].append('tensorflow')
                    info['tensorflow'] = {
                        'signature': str(inspect.signature(func)),
                        'docstring': inspect.getdoc(func) or "No documentation available"
                    }
            except Exception as e:
                info['tensorflow_error'] = str(e)
    
    # Check PyTorch
    if framework is None or framework.lower() == 'pytorch':
        torch_module = _load_torch_signxai()
        if torch_module is not None:
            try:
                import signxai.torch_signxai.methods.wrappers as pt_wrappers
                if hasattr(pt_wrappers, method_name):
                    func = getattr(pt_wrappers, method_name)
                    info['available_in'].append('pytorch')
                    info['pytorch'] = {
                        'signature': str(inspect.signature(func)),
                        'docstring': inspect.getdoc(func) or "No documentation available"
                    }
            except Exception as e:
                info['pytorch_error'] = str(e)
    
    return info


# Common method parameter presets for easy use
METHOD_PRESETS = {
    'gradient': {},
    'smoothgrad': {'num_samples': 25, 'noise_level': 0.1},
    'integrated_gradients': {'steps': 50},
    'vargrad': {'num_samples': 25, 'noise_level': 0.2},
    'guided_backprop': {},
    'deconvnet': {},
    'grad_cam': {},  # Requires layer_name
    'lrp_epsilon': {'epsilon': 0.1},
    'lrp_alpha_1_beta_0': {'alpha': 1.0, 'beta': 0.0},
    'lrp_alpha_2_beta_1': {'alpha': 2.0, 'beta': 1.0},
}


def explain_with_preset(model, x, method_name: str, preset: str = 'default', **override_kwargs):
    """
    Explain using predefined parameter presets for common use cases.
    
    Args:
        model: Model to explain
        x: Input data
        method_name: XAI method name
        preset: Preset name ('default', 'fast', 'high_quality')
        **override_kwargs: Parameters to override preset values
        
    Returns:
        Explanation as numpy array
    """
    # Get base parameters for method
    base_params = METHOD_PRESETS.get(method_name, {}).copy()
    
    # Apply preset modifications
    if preset == 'fast':
        if method_name == 'smoothgrad':
            base_params.update({'num_samples': 10, 'noise_level': 0.15})
        elif method_name == 'integrated_gradients':
            base_params.update({'steps': 20})
    elif preset == 'high_quality':
        if method_name == 'smoothgrad':
            base_params.update({'num_samples': 50, 'noise_level': 0.05})
        elif method_name == 'integrated_gradients':
            base_params.update({'steps': 100})
    
    # Apply any user overrides
    base_params.update(override_kwargs)
    
    return explain(model, x, method_name, **base_params)