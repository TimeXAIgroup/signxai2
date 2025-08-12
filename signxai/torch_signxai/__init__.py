# signxai/torch_signxai/__init__.py

import os
import logging

logger = logging.getLogger(__name__)

# Check if we should use the new Method Family Architecture
USE_METHOD_FAMILIES = os.environ.get('SIGNXAI_USE_METHOD_FAMILIES', 'true').lower() == 'true'

if USE_METHOD_FAMILIES:
    # Use the new Method Family Architecture by default
    logger.info("Using Method Family Architecture for PyTorch")
    from .methods_family import calculate_relevancemap_with_families as calculate_relevancemap
else:
    # Fallback to original implementation
    logger.info("Using original wrapper implementation for PyTorch")
    from .methods.zennit_impl import calculate_relevancemap as calculate_relevancemap

# Import utilities that are commonly used
from .utils import remove_softmax, decode_predictions, NoSoftmaxWrapper

# Import individual method functions for compatibility
try:
    from .methods.wrappers import (
        integrated_gradients,
        grad_cam,
    )
except ImportError:
    # If wrappers are removed, define stub functions
    def integrated_gradients(*args, **kwargs):
        return calculate_relevancemap(method='integrated_gradients', *args, **kwargs)

    def grad_cam(*args, **kwargs):
        return calculate_relevancemap(method='grad_cam', *args, **kwargs)

# Define what gets imported with "from signxai.torch_signxai import *" for clarity
__all__ = [
    "calculate_relevancemap",  # This will be the Zennit one
    "remove_softmax",
    "decode_predictions",
    "NoSoftmaxWrapper",
    # Individual methods for API compatibility
    "integrated_gradients",
    "grad_cam",
]