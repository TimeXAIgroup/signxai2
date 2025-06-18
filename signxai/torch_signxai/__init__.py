# signxai/torch_signxai/__init__.py

# Import the Zennit-based implementation which is the preferred default
from .methods.zennit_impl import calculate_relevancemap as calculate_relevancemap_zennit_api

# Import utilities that are commonly used
# Make sure signxai/torch_signxai/utils.py defines these top-level functions
from .utils import remove_softmax, decode_predictions, NoSoftmaxWrapper

# Expose the PyTorch native API (Zennit-based) as the default 'calculate_relevancemap'
# This is the most important part for run_gradient_comparison.py
calculate_relevancemap = calculate_relevancemap_zennit_api

# Define what gets imported with "from signxai.torch_signxai import *" for clarity
__all__ = [
    "calculate_relevancemap",  # This will be the Zennit one
    "remove_softmax",
    "decode_predictions",
    "NoSoftmaxWrapper",
]