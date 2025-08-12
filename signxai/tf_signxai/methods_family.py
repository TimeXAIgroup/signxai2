"""
TensorFlow integration for Method Family Architecture.
This module provides an alternative entry point that uses the new family-based approach.
"""

import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

def calculate_relevancemap_with_families(method: str,
                                        x: np.ndarray,
                                        model,
                                        neuron_selection: int,
                                        **kwargs):
    """
    Calculate relevance map using the Method Family Architecture.
    Falls back to original wrappers if needed.
    
    Args:
        method: The XAI method to use
        x: Input data as numpy array
        model: TensorFlow/Keras model (without softmax)
        neuron_selection: Target class/neuron index
        **kwargs: Additional method-specific parameters
    
    Returns:
        Relevance map as numpy array
    """
    # Check if families are enabled (default to true)
    use_families = os.environ.get('SIGNXAI_USE_METHOD_FAMILIES', 'true').lower() == 'true'
    
    if use_families:
        try:
            from ..common.method_families import get_registry
            
            registry = get_registry()
            
            # Add target_class to kwargs for compatibility
            kwargs['target_class'] = neuron_selection
            
            logger.info(f"Attempting to execute {method} with Method Family Architecture")
            
            result = registry.execute(
                model=model,
                x=x,
                method_name=method,
                framework='tensorflow',
                **kwargs
            )
            
            logger.info(f"Successfully executed {method} with Method Family Architecture")
            return result
            
        except Exception as e:
            logger.warning(f"Method Family execution failed for {method}: {e}")
            logger.info("Falling back to original implementation")
    
    # Fallback to original implementation
    from .methods.wrappers import calculate_relevancemap as original_calculate
    
    return original_calculate(
        method=method,
        x=x,
        model=model,
        neuron_selection=neuron_selection,
        **kwargs
    )