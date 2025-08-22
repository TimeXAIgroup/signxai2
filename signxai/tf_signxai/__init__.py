import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

# Check if we should use the new Method Family Architecture
USE_METHOD_FAMILIES = os.environ.get('SIGNXAI_USE_METHOD_FAMILIES', 'true').lower() == 'true'

# Attempt to import common validation, handle if not found for robustness
try:
    from ..common.validation import validate_input
except ImportError:
    def validate_input(*args, **kwargs):  # pragma: no cover
        """Dummy validate_input function. Does nothing."""
        pass

# Define supported methods (ensure this list is comprehensive for your package)
SUPPORTED_METHODS = [
    "gradient", "smoothgrad", "integrated_gradients", "guided_backprop",
    "lrp.epsilon", "lrp.alpha_beta", "lrp.alpha_2_beta_1", "lrp.z", "lrp.w_square", "lrp.flat",  # Common LRP rules
    "deep_taylor", "input_t_gradient", "deconvnet",
    "occlusion", "grad_cam"
]


def calculate_relevancemap(method: str,
                           x: np.ndarray,
                           model,  # This is the Keras model
                           neuron_selection: int,
                           **kwargs):
    """
    Calculates the relevance map for a given input and model using the specified TensorFlow-based method.

    Args:
        method (str): The XAI method to use (e.g., "gradient", "lrp.epsilon").
        x (np.ndarray): The input data (e.g., image) as a NumPy array.
        model: The TensorFlow/Keras model (must be a model without softmax for many methods).
        neuron_selection (int): The index of the output neuron for which to generate the explanation.
        **kwargs: Additional arguments specific to the chosen XAI method.

    Returns:
        np.ndarray: The calculated relevance map.

    Raises:
        ValueError: If the method is not supported or if inputs are invalid.
    """
    
    # Use Method Family Architecture if enabled
    if USE_METHOD_FAMILIES:
        try:
            from .methods_family import calculate_relevancemap_with_families
            logger.info(f"Using Method Family Architecture for method: {method}")
            return calculate_relevancemap_with_families(
                method=method,
                x=x,
                model=model,
                neuron_selection=neuron_selection,
                **kwargs
            )
        except Exception as e:
            logger.warning(f"Method Family failed, falling back to original: {e}")
            # Continue with original implementation below
    
    # validate_input(x, model)
    
    import re
    
    # First check for dynamic parameter parsing in method names
    original_method = method
    
    # Parse dynamic LRP alpha-beta methods (e.g., lrp_alpha_3_beta_2)
    alpha_beta_match = re.match(r'(lrp|lrpsign)_alpha_(\d+)_beta_(\d+)', method)
    if alpha_beta_match:
        prefix = alpha_beta_match.group(1)
        alpha = int(alpha_beta_match.group(2))
        beta = int(alpha_beta_match.group(3))
        
        # Use the closest predefined method or default to alpha_beta
        if alpha == 1 and beta == 0:
            method = "lrp.alpha_1_beta_0"
        elif alpha == 2 and beta == 1:
            method = "lrp.alpha_2_beta_1"
        else:
            # For other alpha-beta combinations, use the general alpha_beta
            method = "lrp.alpha_beta"
            # Pass alpha and beta as kwargs
            kwargs['alpha'] = float(alpha)
            kwargs['beta'] = float(beta)
    
    # Parse dynamic epsilon values (e.g., lrp_epsilon_0_1 means epsilon=0.1)
    epsilon_match = re.match(r'(lrp|lrpsign)_epsilon_(\d+)(?:_(\d+))?(?:_std_x)?', method)
    if epsilon_match and not alpha_beta_match:
        whole_part = int(epsilon_match.group(2))
        decimal_part = int(epsilon_match.group(3)) if epsilon_match.group(3) else 0
        
        # Convert to decimal (e.g., 0_1 -> 0.1, 0_25 -> 0.25)
        if decimal_part > 0:
            epsilon_value = float(f"{whole_part}.{decimal_part}")
        else:
            epsilon_value = float(whole_part)
        
        method = "lrp.epsilon"
        kwargs['epsilon'] = epsilon_value
        
        # Check for std_x variant
        if '_std_x' in original_method:
            method = "lrp.stdxepsilon"
    
    # Static mappings for other methods
    method_mapping = {
        "lrp_alpha_1_beta_0": "lrp.alpha_1_beta_0",
        "lrp_alpha_2_beta_1": "lrp.alpha_2_beta_1",
        "lrpsign_alpha_1_beta_0": "lrp.alpha_1_beta_0",  
        "lrp_sequential_composite_a": "lrp.sequential_composite_a",
        "lrp_sequential_composite_b": "lrp.sequential_composite_b",
        "lrpsign_sequential_composite_a": "lrp.sequential_composite_a",
        "lrpsign_sequential_composite_b": "lrp.sequential_composite_b",
        "lrp_epsilon": "lrp.epsilon",
        "lrp_z": "lrp.z",
        "lrpsign_z": "lrp.z",
        "lrp_gamma": "lrp.gamma",
        "lrp_flat": "lrp.flat",
        "lrp_w_square": "lrp.w_square",
    }
    
    # Apply static mapping if no dynamic parsing was done
    if method == original_method and method in method_mapping:
        method = method_mapping[method]

    # Check if method string is recognized, but allow to proceed if not strictly in list for flexibility
    if not isinstance(method, str):  # pragma: no cover
        raise ValueError("Method argument must be a string.")
    # if method not in SUPPORTED_METHODS:
    # print(f"Warning: Method '{method}' not in explicitly defined SUPPORTED_METHODS. Attempting to proceed.")

    if not isinstance(x, np.ndarray):  # pragma: no cover
        raise ValueError("Input x must be a NumPy array.")

    # Instead of importing wrappers, use the new methods module directly
    try:
        from . import methods as tf_methods
        from ..common.method_parser import MethodParser
        tf_method_available = True
    except ImportError:  # pragma: no cover
        tf_method_available = False
        print(
            "Warning: Could not import signxai.tf_signxai.methods. TF-specific methods may not be available.")

    relevancemap = None
    # Use the new methods module directly
    specific_wrapper_used = False
    if tf_method_available:
        try:
            # Parse the method name to extract components
            parser = MethodParser()
            parsed_method = parser.parse(method)
            
            # Execute using the new methods module
            relevancemap = tf_methods.execute(
                model=model,
                x=x,
                parsed_method=parsed_method,
                target_class=neuron_selection,
                neuron_selection=neuron_selection,
                **kwargs
            )
            specific_wrapper_used = True
        except Exception as e:
            logger.warning(f"Direct method execution failed for {method}: {e}")
            # Fall through to iNNvestigate handler below

    # If not handled by a specific TF wrapper, or if wrappers failed to import,
    # try the generic iNNvestigate handler for methods it supports.
    if not specific_wrapper_used:
        # This list can be cross-referenced with iNNvestigate's own supported method strings
        # Note: 'gradient', 'smoothgrad', 'integrated_gradients', 'guided_backprop' might also be routed here
        # if their specific wrappers above are commented out or tf_method_wrappers is None.
        # The current logic prioritizes specific wrappers if tf_method_wrappers exists.
        innvestigate_methods = [
            "gradient", "smoothgrad", "integrated_gradients", "guided_backprop",
            "lrp.epsilon", "lrp.alpha_beta", "lrp.alpha_2_beta_1", "lrp.z", "lrp.w_square", "lrp.flat",
            "deep_taylor", "input_t_gradient", "deconvnet"
        ]
        # Occlusion and Grad-CAM are typically custom wrappers not directly in iNNvestigate's generic analyzer call.

        if method in innvestigate_methods or method.startswith('lrp'):
            try:
                from ..utils.utils import calculate_explanation_innvestigate
                # The 'model' passed to calculate_explanation_innvestigate is expected to be
                # the model without softmax by that utility's internal logic for iNNvestigate.
                
                relevancemap = calculate_explanation_innvestigate(method=method, x=x, model=model,
                                                                  neuron_selection=neuron_selection, **kwargs)
            except ImportError:  # pragma: no cover
                raise ImportError(
                    "Failed to import 'calculate_explanation_innvestigate' from signxai.utils.utils. Cannot proceed with iNNvestigate-based method.")
            except Exception as e:  # pragma: no cover
                raise ValueError(f"Method '{method}' failed in generic iNNvestigate handler. Error: {e}")
        else:  # pragma: no cover
            # If method was not handled by specific wrappers AND not in innvestigate_methods list.
            if method not in SUPPORTED_METHODS: # Check against the package's declared supported methods.
                raise ValueError(
                    f"Unsupported method: {method}. Supported methods are: {SUPPORTED_METHODS} or check iNNvestigate specific methods if applicable.")
            else: # Method in SUPPORTED_METHODS but no handler was found/triggered.
                  # This case implies a logic error in the dispatch (e.g. tf_method_wrappers was None but method expected it)
                raise ValueError(
                    f"Method '{method}' is listed as supported but could not be processed. Check tf_method_wrappers import and dispatch logic for this method.")

    if relevancemap is None:  # pragma: no cover
        # This should ideally not be reached if the dispatch logic is complete and error handling within methods is robust.
        raise ValueError(f"Method '{method}' could not be processed and did not produce a result. Check specific handlers.")

    return relevancemap