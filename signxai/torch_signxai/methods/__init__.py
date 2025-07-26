"""PyTorch implementation of explanation methods."""
from .wrappers import _calculate_relevancemap

# Import the base classes
from .base import BaseGradient, InputXGradient, GradientXSign
from .integrated import IntegratedGradients, IntegratedGradientsXInput, IntegratedGradientsXSign
from .smoothgrad import SmoothGrad, SmoothGradXInput, SmoothGradXSign
from .vargrad import VarGrad, VarGradXInput, VarGradXSign

# Define PyTorch-style API
def calculate_relevancemap(model, input_tensor, method="gradients", **kwargs):
    """Calculate relevance map for a single input.
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor
        method: Name of the explanation method
        **kwargs: Additional arguments for the specific method
        
    Returns:
        Relevance map as numpy array
    """
    # Ensure model and input_tensor are in the correct order
    return _calculate_relevancemap(model=model, input_tensor=input_tensor, method=method, **kwargs)

def calculate_relevancemaps(model, input_tensors, method="gradients", **kwargs):
    """Calculate relevance maps for multiple inputs.
    
    Args:
        model: PyTorch model
        input_tensors: Input tensors (list or batch)
        method: Name of the explanation method
        **kwargs: Additional arguments for the specific method
        
    Returns:
        List of relevance maps
    """
    import numpy as np
    
    # Handle list-like inputs
    if isinstance(input_tensors, list):
        return [calculate_relevancemap(model, x, method, **kwargs) for x in input_tensors]
    
    # Handle batch inputs
    results = []
    for i in range(len(input_tensors)):
        single_input = input_tensors[i:i+1]
        result = calculate_relevancemap(model, single_input, method, **kwargs)
        results.append(result)
    
    return np.array(results)

__all__ = [
    "calculate_relevancemap",
    "calculate_relevancemaps",
    "BaseGradient",
    "InputXGradient",
    "GradientXSign",
    "IntegratedGradients",
    "IntegratedGradientsXInput",
    "IntegratedGradientsXSign",
    "SmoothGrad",
    "SmoothGradXInput",
    "SmoothGradXSign",
    "VarGrad",
    "VarGradXInput",
    "VarGradXSign",
]