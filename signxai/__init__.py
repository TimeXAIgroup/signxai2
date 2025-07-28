# signxai/__init__.py (Simplified and Fixed)
__version__ = "0.8.0"

_DEFAULT_BACKEND = None
_AVAILABLE_BACKENDS = []

# Module placeholders
tf_signxai = None
torch_signxai = None

# Lazy loading functions to avoid circular imports
def _load_tf_signxai():
    """Lazy loader for TensorFlow SignXAI module."""
    global tf_signxai
    if tf_signxai is None:
        try:
            import tensorflow
            import signxai.tf_signxai as tf_module
            tf_signxai = tf_module
            if "tensorflow" not in _AVAILABLE_BACKENDS:
                _AVAILABLE_BACKENDS.append("tensorflow")
        except ImportError:
            pass
    return tf_signxai

def _load_torch_signxai():
    """Lazy loader for PyTorch SignXAI module.""" 
    global torch_signxai
    if torch_signxai is None:
        try:
            import torch
            import zennit  # Required for PyTorch LRP methods
            import signxai.torch_signxai as torch_module
            torch_signxai = torch_module
            if "pytorch" not in _AVAILABLE_BACKENDS:
                _AVAILABLE_BACKENDS.append("pytorch")
        except ImportError:
            pass
    return torch_signxai

# Attempt immediate loading to populate _AVAILABLE_BACKENDS
try:
    import tensorflow
    _load_tf_signxai()
    if not _DEFAULT_BACKEND:
        _DEFAULT_BACKEND = "tensorflow"
except ImportError:
    pass

try:
    import torch
    import zennit
    _load_torch_signxai()
    if not _DEFAULT_BACKEND:
        _DEFAULT_BACKEND = "pytorch"
except ImportError:
    pass

# Helper functions for API (defined here to avoid circular imports)
def _detect_framework(model):
    """Detect which framework a model belongs to."""
    # Check TensorFlow
    try:
        import tensorflow as tf
        if isinstance(model, (tf.keras.Model, tf.keras.Sequential)) or hasattr(model, 'predict'):
            return 'tensorflow'
    except ImportError:
        pass
    
    # Check PyTorch
    try:
        import torch
        if isinstance(model, torch.nn.Module):
            return 'pytorch'
    except ImportError:
        pass
    
    return None


def _prepare_model(model, framework):
    """Prepare model for explanation (remove softmax if needed)."""
    if framework == 'tensorflow':
        from signxai.utils.utils import remove_softmax
        return remove_softmax(model)
    else:  # pytorch
        from signxai.torch_signxai.utils import remove_softmax
        model_copy = model.__class__(**{k: v for k, v in model.__dict__.items() if not k.startswith('_')})
        model_copy.load_state_dict(model.state_dict())
        return remove_softmax(model_copy)


def _prepare_input(x, framework):
    """Prepare input data for the specified framework."""
    import numpy as np
    
    if framework == 'tensorflow':
        # Ensure numpy array for TensorFlow
        if hasattr(x, 'detach'):  # PyTorch tensor
            x = x.detach().cpu().numpy()
        elif not isinstance(x, np.ndarray):
            x = np.array(x)
        return x
    else:  # pytorch
        # Ensure PyTorch tensor
        import torch
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return x


def _get_predicted_class(model, x, framework):
    """Get the predicted class from the model."""
    import numpy as np
    
    if framework == 'tensorflow':
        preds = model.predict(x, verbose=0)
        return int(np.argmax(preds[0]))
    else:  # pytorch
        import torch
        model.eval()
        with torch.no_grad():
            preds = model(x)
        return int(torch.argmax(preds, dim=1).item())


def _map_parameters(method_name, framework, **kwargs):
    """Map parameters between frameworks for method compatibility."""
    mapped = kwargs.copy()
    
    # Common parameter mappings
    param_mapping = {
        'integrated_gradients': {
            'tensorflow': {'reference_inputs': 'baseline', 'steps': 'steps'},
            'pytorch': {'baseline': 'reference_inputs', 'ig_steps': 'steps'}
        },
        'smoothgrad': {
            'tensorflow': {'augment_by_n': 'num_samples', 'noise_scale': 'noise_level'},
            'pytorch': {'num_samples': 'augment_by_n', 'noise_level': 'noise_scale'}
        },
        'grad_cam': {
            'tensorflow': {'layer_name': 'layer_name'},
            'pytorch': {'target_layer': 'layer_name'}
        }
    }
    
    if method_name in param_mapping:
        target_mapping = param_mapping[method_name].get(framework, {})
        for new_key, old_key in target_mapping.items():
            if old_key in kwargs:
                mapped[new_key] = mapped.pop(old_key)
    
    return mapped


def _call_tensorflow_method(model, x, method_name, target_class, **kwargs):
    """Call TensorFlow implementation."""
    tf_module = _load_tf_signxai()
    from signxai.tf_signxai.methods.wrappers import calculate_relevancemap
    
    # Handle neuron_selection parameter
    return calculate_relevancemap(
        method_name, x, model, 
        neuron_selection=target_class, 
        **kwargs
    )


def _call_pytorch_method(model, x, method_name, target_class, **kwargs):
    """Call PyTorch implementation."""
    torch_module = _load_torch_signxai()
    from signxai.torch_signxai import calculate_relevancemap
    
    return calculate_relevancemap(
        model=model, input_tensor=x, method=method_name,
        target_class=target_class, **kwargs
    )


# Legacy framework-specific imports (for backwards compatibility)
def _framework_specific_import_required(*args, **kwargs):
    msg = ("Use the unified API: from signxai import explain\n"
           "Or framework-specific imports:\n"
           "  TensorFlow: from signxai.tf_signxai import calculate_relevancemap\n"
           "  PyTorch: from signxai.torch_signxai import calculate_relevancemap")
    raise ImportError(msg)

calculate_relevancemap = _framework_specific_import_required
calculate_relevancemaps = _framework_specific_import_required

# Import API functions for convenience
try:
    from .api import explain, list_methods, get_method_info, explain_with_preset, METHOD_PRESETS
    _API_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import unified API: {e}")
    _API_AVAILABLE = False

# Dynamically build __all__
__all__ = ['__version__', '_DEFAULT_BACKEND', '_AVAILABLE_BACKENDS', 'calculate_relevancemap', 'calculate_relevancemaps']

# Add API functions if available
if _API_AVAILABLE:
    __all__.extend(['explain', 'list_methods', 'get_method_info', 'explain_with_preset', 'METHOD_PRESETS'])

# Add modules to __all__ if available
if _load_tf_signxai():
    __all__.append('tf_signxai')
if _load_torch_signxai():
    __all__.append('torch_signxai')