import numpy as np

# Attempt to import common validation, handle if not found for robustness
try:
    from ..common.validation import validate_input
    # If validate_input is successfully imported, you might want to log or know.
    # print("Successfully imported 'validate_input' from signxai.common.validation")
except ImportError:
    # This block executes if 'validate_input' cannot be imported from signxai.common.validation
    # Define a dummy validate_input if the real one isn't found or needed immediately
    def validate_input(*args, **kwargs):  # pragma: no cover
        """Dummy validate_input function. Does nothing."""
        # You can add a print here to confirm the dummy is being used:
        # print("Warning: 'validate_input' not found in signxai.common.validation. Using dummy placeholder.")
        pass
    # print("Defined dummy 'validate_input' due to ImportError.")

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
    # If you intend to use validate_input, ensure it's called appropriately.
    # For now, its call is commented out as its definition might be a dummy.
    # validate_input(x, model)

    # Check if method string is recognized, but allow to proceed if not strictly in list for flexibility
    if not isinstance(method, str):  # pragma: no cover
        raise ValueError("Method argument must be a string.")
    # if method not in SUPPORTED_METHODS:
    # print(f"Warning: Method '{method}' not in explicitly defined SUPPORTED_METHODS. Attempting to proceed.")

    if not isinstance(x, np.ndarray):  # pragma: no cover
        raise ValueError("Input x must be a NumPy array.")

    try:
        from .methods import wrappers as tf_method_wrappers
    except ImportError:  # pragma: no cover
        tf_method_wrappers = None
        print(
            "Warning: Could not import signxai.tf_signxai.methods.wrappers. TF-specific wrapped methods may not be available.")

    relevancemap = None
    # Method dispatch using specific wrappers if available
    specific_wrapper_used = False
    if tf_method_wrappers:
        if method == 'gradient':
            # This wrapper likely expects model_no_softmax directly, as it was working
            relevancemap = tf_method_wrappers.gradient(x=x, model_no_softmax=model, neuron_selection=neuron_selection, **kwargs)
            specific_wrapper_used = True
        elif method == 'smoothgrad':
            # Corrected: pass 'model' as 'model_no_softmax'
            relevancemap = tf_method_wrappers.smoothgrad(x=x, model_no_softmax=model, neuron_selection=neuron_selection, **kwargs)
            specific_wrapper_used = True
        elif method == 'integrated_gradients':
            # Corrected: pass 'model' as 'model_no_softmax'
            relevancemap = tf_method_wrappers.integrated_gradients(x=x, model_no_softmax=model, neuron_selection=neuron_selection,
                                                                   **kwargs)
            specific_wrapper_used = True
        elif method == 'guided_backprop':
            # Corrected: pass 'model' as 'model_no_softmax'
            relevancemap = tf_method_wrappers.guided_backprop(x=x, model_no_softmax=model, neuron_selection=neuron_selection,
                                                              **kwargs)
            specific_wrapper_used = True
        elif method == 'grad_cam':
            # Corrected: pass 'model' as 'model_no_softmax'
            relevancemap = tf_method_wrappers.grad_cam(x=x, model_no_softmax=model, neuron_selection=neuron_selection, **kwargs)
            specific_wrapper_used = True
        elif method == 'occlusion':
            # Corrected: pass 'model' as 'model_no_softmax' (assuming it follows the same pattern)
            relevancemap = tf_method_wrappers.occlusion(x=x, model_no_softmax=model, neuron_selection=neuron_selection, **kwargs)
            specific_wrapper_used = True
        # Add other specific tf_method_wrappers calls here if you have them, ensuring to use model_no_softmax=model

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