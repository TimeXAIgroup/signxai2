"""
PyTorch integration for Method Family Architecture.
This module provides an alternative entry point that uses the new family-based approach.
"""

import torch
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

def calculate_relevancemap_with_families(model, 
                                        input_tensor,
                                        method: str,
                                        target_class=None,
                                        **kwargs):
    """
    Calculate relevance map using the Method Family Architecture.
    Falls back to original wrappers if needed.
    
    Args:
        model: PyTorch model (without softmax)
        input_tensor: Input tensor
        method: The XAI method to use
        target_class: Target class index (optional)
        **kwargs: Additional method-specific parameters
    
    Returns:
        Relevance map as numpy array
    """
    # Always try families first (they're the default now)
    use_families = os.environ.get('SIGNXAI_USE_METHOD_FAMILIES', 'true').lower() == 'true'
    
    if use_families:
        try:
            from ..common.method_families import get_registry
            
            registry = get_registry()
            
            # Add target_class to kwargs for compatibility (only if not already present)
            if target_class is not None and 'target_class' not in kwargs:
                kwargs['target_class'] = target_class
            
            logger.info(f"Attempting to execute {method} with Method Family Architecture")
            
            result = registry.execute(
                model=model,
                x=input_tensor,
                method_name=method,
                framework='pytorch',
                **kwargs
            )
            
            logger.info(f"Successfully executed {method} with Method Family Architecture")
            return result
            
        except Exception as e:
            logger.warning(f"Method Family execution failed for {method}: {e}")
            logger.info("Falling back to original implementation")
    
    # Fallback to zennit_impl
    from .methods.zennit_impl import calculate_relevancemap as zennit_calculate
    
    # Remove target_class from kwargs if present to avoid duplicate
    fallback_kwargs = kwargs.copy()
    fallback_kwargs.pop('target_class', None)
    
    return zennit_calculate(
            model=model,
            input_tensor=input_tensor,
            method=method,
            target_class=target_class,
            **fallback_kwargs
        )