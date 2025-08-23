# signxai/__init__.py
__version__ = "0.13.8"

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
# Check PyTorch first to make it the default when both are available
try:
    import torch
    import zennit

    _load_torch_signxai()
    if not _DEFAULT_BACKEND:
        _DEFAULT_BACKEND = "pytorch"
except ImportError:
    pass

try:
    import tensorflow

    _load_tf_signxai()
    if not _DEFAULT_BACKEND:
        _DEFAULT_BACKEND = "tensorflow"
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
    """Call TensorFlow implementation using the new architecture."""
    tf_module = _load_tf_signxai()
    if not tf_module:
        _check_framework_availability()
    
    # Use the new method family architecture
    from signxai.common.method_families import get_registry
    from signxai.common.method_parser import MethodParser
    
    # Try method families first
    try:
        registry = get_registry()
        return registry.execute(
            model=model,
            x=x,
            method_name=method_name,
            framework='tensorflow',
            target_class=target_class,
            neuron_selection=target_class,
            **kwargs
        )
    except Exception as e:
        # Fallback to direct execution
        from signxai.tf_signxai.methods import execute as tf_execute
        parser = MethodParser()
        parsed_method = parser.parse(method_name)
        return tf_execute(
            model=model,
            x=x,
            parsed_method=parsed_method,
            target_class=target_class,
            neuron_selection=target_class,
            **kwargs
        )


def _call_pytorch_method(model, x, method_name, target_class, **kwargs):
    """Call PyTorch implementation using the new architecture."""
    torch_module = _load_torch_signxai()
    if not torch_module:
        _check_framework_availability()
    
    # Use the new method family architecture
    from signxai.common.method_families import get_registry
    from signxai.common.method_parser import MethodParser
    
    # Try method families first
    try:
        registry = get_registry()
        return registry.execute(
            model=model,
            x=x,
            method_name=method_name,
            framework='pytorch',
            target_class=target_class,
            **kwargs
        )
    except Exception as e:
        # Fallback to direct execution
        from signxai.torch_signxai.methods import execute as pt_execute
        parser = MethodParser()
        parsed_method = parser.parse(method_name)
        return pt_execute(
            model=model,
            x=x,
            parsed_method=parsed_method,
            target_class=target_class,
            **kwargs
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


# Check if any framework is available
def _check_framework_availability():
    """Check if at least one framework is available and provide helpful error if not."""
    if not _AVAILABLE_BACKENDS:
        error_msg = (
                "\n" + "=" * 70 + "\n"
                                  "ERROR: No deep learning framework detected!\n\n"
                                  "SignXAI2 requires at least one framework to be installed.\n"
                                  "You have installed signxai2 without specifying a framework.\n\n"
                                  "Please install SignXAI2 with one of the following options:\n\n"
                                  "  For TensorFlow support:\n"
                                  "    pip install signxai2[tensorflow]\n\n"
                                  "  For PyTorch support:\n"
                                  "    pip install signxai2[pytorch]\n\n"
                                  "  For both frameworks:\n"
                                  "    pip install signxai2[all]\n\n"
                                  "  For development (includes all frameworks + dev tools):\n"
                                  "    pip install signxai2[dev]\n\n"
                                  "Note: Python 3.9 or 3.10 is required.\n"
                                  "=" * 70 + "\n"
        )
        raise ImportError(error_msg)


# Import API functions for convenience
try:
    from .api import (
        explain as _explain_impl,
        list_methods as _list_methods_impl,
        get_method_info as _get_method_info_impl,
        explain_with_preset as _explain_with_preset_impl,
        METHOD_PRESETS
    )

    _API_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import unified API: {e}")
    _API_AVAILABLE = False


# Create wrapper functions that check framework availability
def explain(*args, **kwargs):
    """Wrapper for explain that checks framework availability."""
    if not _AVAILABLE_BACKENDS:
        _check_framework_availability()
    if not _API_AVAILABLE:
        raise ImportError("SignXAI API is not available. Please check your installation.")
    return _explain_impl(*args, **kwargs)


def list_methods(*args, **kwargs):
    """Wrapper for list_methods that checks framework availability."""
    if not _AVAILABLE_BACKENDS:
        _check_framework_availability()
    if not _API_AVAILABLE:
        raise ImportError("SignXAI API is not available. Please check your installation.")
    return _list_methods_impl(*args, **kwargs)


def get_method_info(*args, **kwargs):
    """Wrapper for get_method_info that checks framework availability."""
    if not _AVAILABLE_BACKENDS:
        _check_framework_availability()
    if not _API_AVAILABLE:
        raise ImportError("SignXAI API is not available. Please check your installation.")
    return _get_method_info_impl(*args, **kwargs)


def explain_with_preset(*args, **kwargs):
    """Wrapper for explain_with_preset that checks framework availability."""
    if not _AVAILABLE_BACKENDS:
        _check_framework_availability()
    if not _API_AVAILABLE:
        raise ImportError("SignXAI API is not available. Please check your installation.")
    return _explain_with_preset_impl(*args, **kwargs)


# Dynamically build __all__
__all__ = ['__version__', '_DEFAULT_BACKEND', '_AVAILABLE_BACKENDS', 'calculate_relevancemap',
           'calculate_relevancemaps']

# Add API functions if available
if _API_AVAILABLE:
    __all__.extend(['explain', 'list_methods', 'get_method_info', 'explain_with_preset', 'METHOD_PRESETS'])

# Add modules to __all__ if available
if _load_tf_signxai():
    __all__.append('tf_signxai')
if _load_torch_signxai():
    __all__.append('torch_signxai')

# Note: Framework availability is checked when API functions are called
