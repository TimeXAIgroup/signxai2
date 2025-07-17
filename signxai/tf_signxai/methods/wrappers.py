import numpy as np
import tensorflow as tf
from signxai.tf_signxai.methods.grad_cam import calculate_grad_cam_relevancemap, calculate_grad_cam_relevancemap_timeseries
from signxai.tf_signxai.methods.guided_backprop import guided_backprop_on_guided_model
from signxai.tf_signxai.methods.signed import calculate_sign_mu
from signxai.utils.utils import calculate_explanation_innvestigate


def calculate_native_gradient(model, x, neuron_selection=None, **kwargs):
    """
    Calculate gradients directly using TensorFlow's GradientTape.
    This is used as a fallback for timeseries data when innvestigate fails.
    
    Args:
        model: TensorFlow model
        x: Input data
        neuron_selection: Index of neuron to explain
        **kwargs: Additional parameters (ignored)
        
    Returns:
        Gradients as numpy array
    """
    # Handle neuron_selection
    if neuron_selection is None:
        # Get the model's prediction to determine the class with highest activation
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        # Ensure batch dimension
        if x.ndim == 2:  # (time_steps, channels)
            x_input = np.expand_dims(x, axis=0)  # Add batch -> (1, time_steps, channels)
        else:
            x_input = x
            
        # Make prediction to find the neuron with max activation
        preds = model.predict(x_input)
        neuron_selection = np.argmax(preds[0])
        print(f"  DEBUG: Native gradient using predicted neuron: {neuron_selection}")
    else:
        print(f"  DEBUG: Native gradient using provided neuron: {neuron_selection}")
    
    # Convert numpy array to tensor if needed
    if isinstance(x, np.ndarray):
        # Ensure batch dimension
        if x.ndim == 2:  # (time_steps, channels)
            x_tensor = tf.convert_to_tensor(np.expand_dims(x, axis=0), dtype=tf.float32)
        else:
            x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    else:
        x_tensor = x
        
    # Compute gradients using TensorFlow's GradientTape
    with tf.GradientTape() as tape:
        # Watch the input tensor
        tape.watch(x_tensor)
        
        # Forward pass
        predictions = model(x_tensor, training=False)
        
        # Select the target class (neuron)
        target_output = predictions[:, neuron_selection]
        
    # Compute gradients of the target output with respect to the input
    gradients = tape.gradient(target_output, x_tensor)
    
    # Convert to numpy array
    gradients_np = gradients.numpy()
    
    print(f"  DEBUG: Native gradient computed successfully with shape: {gradients_np.shape}")
    
    # If the input had no batch dimension, remove it from the output
    if isinstance(x, np.ndarray) and x.ndim == 2 and gradients_np.shape[0] == 1:
        gradients_np = gradients_np[0]
        
    return gradients_np


def random_uniform(model_no_softmax, x, **kwargs):
    np.random.seed(1)

    channel_values = []

    uniform_values = np.random.uniform(low=-1, high=1, size=(x.shape[0], x.shape[1]))

    for i in range(x.shape[2]):
        channel_values.append(np.array(uniform_values))

    return np.stack(channel_values, axis=2)


def gradient(model_no_softmax, x, **kwargs):
    """
    Compute gradient-based explanation for input x.
    
    This will try to use innvestigate's gradient implementation first,
    but fall back to a direct TensorFlow implementation for timeseries data
    if that fails.
    
    Args:
        model_no_softmax: Model to analyze
        x: Input data
        **kwargs: Additional parameters
        
    Returns:
        Gradient-based explanation
    """
    # Direct iNNvestigate implementation - no fallbacks
    return calculate_explanation_innvestigate(model_no_softmax, x, method='gradient', **kwargs)


def input_t_gradient(model_no_softmax, x, **kwargs):
    return calculate_explanation_innvestigate(model_no_softmax, x, method='input_t_gradient', **kwargs)


def gradient_x_input(model_no_softmax, x, **kwargs):
    """
    Compute gradient × input explanation for input x.
    
    Args:
        model_no_softmax: Model to analyze
        x: Input data
        **kwargs: Additional parameters
        
    Returns:
        Gradient × input explanation
    """
    # Get the gradient using the gradient method (which has a fallback for timeseries)
    g = gradient(model_no_softmax, x, **kwargs)
    
    # Ensure x is a numpy array with matching shape
    if isinstance(x, np.ndarray):
        # The gradient should have the same shape as the input
        # If not, align shapes
        if g.shape != x.shape and (g.ndim == 3 and g.shape[0] == 1 and x.ndim == 2):
            # Gradient has batch dim but input doesn't
            g = g[0]
        elif g.shape != x.shape and (x.ndim == 3 and x.shape[0] == 1 and g.ndim == 2):
            # Input has batch dim but gradient doesn't
            x_squeezed = x[0]
            return g * x_squeezed
    
    # Compute grad * input and return
    return g * x


def gradient_x_sign(model_no_softmax, x, **kwargs):
    """
    Compute gradient × sign(input) explanation for input x.
    
    Args:
        model_no_softmax: Model to analyze
        x: Input data
        **kwargs: Additional parameters
        
    Returns:
        Gradient × sign(input) explanation
    """
    # Get the gradient using the gradient method (which has a fallback for timeseries)
    g = gradient(model_no_softmax, x, **kwargs)
    
    # Ensure x is a numpy array
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    
    # Align shapes if needed
    if g.shape != x.shape and (g.ndim == 3 and g.shape[0] == 1 and x.ndim == 2):
        # Gradient has batch dim but input doesn't
        g = g[0]
    elif g.shape != x.shape and (x.ndim == 3 and x.shape[0] == 1 and g.ndim == 2):
        # Input has batch dim but gradient doesn't
        x = x[0]
    
    # Compute sign of input, replacing NaN with 1.0
    s = np.nan_to_num(x / np.abs(x), nan=1.0)
    
    # Compute grad * sign(input) and return
    return g * s


def gradient_x_sign_mu(model_no_softmax, x, mu, batchmode=False, **kwargs):
    if batchmode:
        G = []
        S = []
        for xi in x:
            G.append(gradient(model_no_softmax, xi, **kwargs))
            S.append(calculate_sign_mu(xi, mu, **kwargs))
        return np.array(G) * np.array(S)
    else:
        return gradient(model_no_softmax, x, **kwargs) * calculate_sign_mu(x, mu, **kwargs)


def gradient_x_sign_mu_0(model_no_softmax, x, **kwargs):
    kwargs.pop('mu', None)  # Remove mu from kwargs to avoid conflict
    return gradient_x_sign_mu(model_no_softmax, x, mu=0, **kwargs)


def gradient_x_sign_mu_0_5(model_no_softmax, x, **kwargs):
    kwargs.pop('mu', None)  # Remove mu from kwargs to avoid conflict
    return gradient_x_sign_mu(model_no_softmax, x, mu=0.5, **kwargs)


def gradient_x_sign_mu_neg_0_5(model_no_softmax, x, **kwargs):
    kwargs.pop('mu', None)  # Remove mu from kwargs to avoid conflict
    return gradient_x_sign_mu(model_no_softmax, x, mu=-0.5, **kwargs)


def guided_backprop(model_no_softmax, x, **kwargs):
    return calculate_explanation_innvestigate(model_no_softmax, x, method='guided_backprop', **kwargs)


def guided_backprop_x_sign(model_no_softmax, x, **kwargs):
    g = guided_backprop(model_no_softmax, x, **kwargs)
    s = np.nan_to_num(x / np.abs(x), nan=1.0)

    return g * s


def guided_backprop_x_sign_mu(model_no_softmax, x, mu, batchmode=False, **kwargs):
    if batchmode:
        G = []
        S = []
        for xi in x:
            G.append(guided_backprop(model_no_softmax, xi, **kwargs))
            S.append(calculate_sign_mu(xi, mu, **kwargs))
        return np.array(G) * np.array(S)
    else:
        return guided_backprop(model_no_softmax, x, **kwargs) * calculate_sign_mu(x, mu, **kwargs)


def guided_backprop_x_sign_mu_0(model_no_softmax, x, **kwargs):
    kwargs.pop('mu', None)  # Remove mu from kwargs to avoid conflict
    return guided_backprop_x_sign_mu(model_no_softmax, x, mu=0, **kwargs)


def guided_backprop_x_sign_mu_0_5(model_no_softmax, x, **kwargs):
    kwargs.pop('mu', None)  # Remove mu from kwargs to avoid conflict
    return guided_backprop_x_sign_mu(model_no_softmax, x, mu=0.5, **kwargs)


def guided_backprop_x_sign_mu_neg_0_5(model_no_softmax, x, **kwargs):
    kwargs.pop('mu', None)  # Remove mu from kwargs to avoid conflict
    return guided_backprop_x_sign_mu(model_no_softmax, x, mu=-0.5, **kwargs)

def calculate_native_integrated_gradients(model, x, steps=50, reference_inputs=None, neuron_selection=None, **kwargs):
    """
    Native implementation of Integrated Gradients for timeseries data.
    
    Args:
        model: TensorFlow model
        x: Input data (with batch dimension)
        steps: Number of steps for the integration
        reference_inputs: Reference inputs (baseline) for the integration
        neuron_selection: Target neuron index
        **kwargs: Additional parameters (ignored)
        
    Returns:
        Integrated gradients explanation
    """
    # Create reference inputs if not provided
    if reference_inputs is None:
        reference_inputs = np.zeros_like(x)
    
    # Handle neuron_selection
    if neuron_selection is None:
        # Get the model's prediction to determine the class with highest activation
        preds = model.predict(x)
        neuron_selection = np.argmax(preds[0])
        print(f"  DEBUG: Native integrated_gradients using predicted neuron: {neuron_selection}")
    else:
        print(f"  DEBUG: Native integrated_gradients using provided neuron: {neuron_selection}")
    
    # Convert to tensors
    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    reference_tensor = tf.convert_to_tensor(reference_inputs, dtype=tf.float32)
    
    # Create path inputs
    alphas = tf.linspace(0.0, 1.0, steps)
    path_inputs = [reference_tensor + alpha * (x_tensor - reference_tensor) for alpha in alphas]
    path_inputs = tf.stack(path_inputs)
    
    # Combine inputs for efficient batch processing
    path_inputs_merged = tf.reshape(path_inputs, [-1] + list(x_tensor.shape[1:]))
    
    # Calculate gradients for all steps in one batch
    with tf.GradientTape() as tape:
        tape.watch(path_inputs_merged)
        preds = model(path_inputs_merged)
        target_preds = preds[:, neuron_selection]
    
    gradients = tape.gradient(target_preds, path_inputs_merged)
    
    # Reshape gradients back to match path_inputs
    gradients = tf.reshape(gradients, path_inputs.shape)
    
    # Compute mean gradient across all steps
    avg_gradients = tf.reduce_mean(gradients, axis=0)
    
    # Compute integrated gradients 
    integrated_grads = (x_tensor - reference_tensor) * avg_gradients
    
    return integrated_grads.numpy()

def integrated_gradients(model_no_softmax, x, **kwargs):
    """
    Compute integrated gradients explanation using iNNvestigate.
    
    Args:
        model_no_softmax: Model to analyze
        x: Input data
        **kwargs: Additional parameters including:
            - steps: Number of steps for integration (default: 50)
            - reference_inputs: Baseline input (default: zeros)
        
    Returns:
        Integrated gradients explanation
    """
    # Ensure input is valid
    if x is None:
        raise ValueError("Input x cannot be None for integrated_gradients")
    
    # Set default values if not specified
    if 'steps' not in kwargs:
        kwargs['steps'] = 50
    
    # Create reference inputs if not provided with validation
    if 'reference_inputs' not in kwargs or kwargs['reference_inputs'] is None:
        try:
            kwargs['reference_inputs'] = np.zeros_like(x)
            print(f"DEBUG: Created reference_inputs with shape {kwargs['reference_inputs'].shape}")
        except Exception as e:
            print(f"WARNING: Failed to create reference_inputs: {e}")
            kwargs['reference_inputs'] = np.zeros(x.shape)
    
    try:
        # Direct iNNvestigate implementation with error handling
        return calculate_explanation_innvestigate(model_no_softmax, x, method='integrated_gradients', **kwargs)
    except Exception as e:
        print(f"iNNvestigate integrated_gradients failed: {e}")
        print("Falling back to native implementation...")
        # Fallback to native implementation
        return calculate_native_integrated_gradients(model_no_softmax, x, **kwargs)


def calculate_native_smoothgrad(model, x, augment_by_n=50, noise_scale=0.2, neuron_selection=None, **kwargs):
    """
    Native implementation of SmoothGrad for timeseries data.
    
    Args:
        model: TensorFlow model
        x: Input data (with batch dimension)
        augment_by_n: Number of noisy samples to generate
        noise_scale: Standard deviation of the Gaussian noise
        neuron_selection: Target neuron index
        **kwargs: Additional parameters (ignored)
        
    Returns:
        SmoothGrad explanation
    """
    # Handle neuron_selection
    if neuron_selection is None:
        # Get the model's prediction to determine the class with highest activation
        preds = model.predict(x)
        neuron_selection = np.argmax(preds[0])
        print(f"  DEBUG: Native smoothgrad using predicted neuron: {neuron_selection}")
    else:
        print(f"  DEBUG: Native smoothgrad using provided neuron: {neuron_selection}")
    
    # Convert to tensor
    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    
    # Compute the standard deviation of the noise
    input_range = tf.reduce_max(x_tensor) - tf.reduce_min(x_tensor)
    if input_range == 0:
        input_range = 1.0  # Avoid division by zero
    noise_std = noise_scale * input_range
    
    # Create a list to store gradients from noisy samples
    gradients_list = []
    
    # Generate augmented samples and compute gradients
    for i in range(augment_by_n):
        # Add random noise to the input
        noise = tf.random.normal(shape=x_tensor.shape, mean=0.0, stddev=noise_std)
        noisy_x = x_tensor + noise
        
        # Compute gradient for the noisy sample
        with tf.GradientTape() as tape:
            tape.watch(noisy_x)
            predictions = model(noisy_x)
            target_output = predictions[:, neuron_selection]
            
        sample_gradients = tape.gradient(target_output, noisy_x)
        gradients_list.append(sample_gradients)
    
    # Stack and compute the mean of all gradients
    all_gradients = tf.stack(gradients_list)
    avg_gradients = tf.reduce_mean(all_gradients, axis=0)
    
    return avg_gradients.numpy()

def smoothgrad(model_no_softmax, x, **kwargs):
    """
    Compute SmoothGrad explanation using iNNvestigate.
    
    Args:
        model_no_softmax: Model to analyze
        x: Input data
        **kwargs: Additional parameters including:
            - augment_by_n: Number of noisy samples (default: 50)
            - noise_scale: Scale of noise (default: 0.2)
        
    Returns:
        SmoothGrad explanation
    """
    # Set default values if not specified
    if 'augment_by_n' not in kwargs:
        kwargs['augment_by_n'] = 50
    if 'noise_scale' not in kwargs:
        kwargs['noise_scale'] = 0.2
    
    # Direct iNNvestigate implementation - no fallbacks
    return calculate_explanation_innvestigate(model_no_softmax, x, method='smoothgrad', **kwargs)


def smoothgrad_x_input(model_no_softmax, x, **kwargs):
    g = smoothgrad(model_no_softmax, x, **kwargs)

    return g * x


def smoothgrad_x_sign(model_no_softmax, x, **kwargs):
    g = smoothgrad(model_no_softmax, x, **kwargs)
    s = np.nan_to_num(x / np.abs(x), nan=1.0)

    return g * s


def smoothgrad_x_sign_mu(model_no_softmax, x, mu, batchmode=False, **kwargs):
    if batchmode:
        G = []
        S = []
        for xi in x:
            G.append(smoothgrad(model_no_softmax, xi, **kwargs))
            S.append(calculate_sign_mu(xi, mu, **kwargs))
        return np.array(G) * np.array(S)
    else:
        return smoothgrad(model_no_softmax, x, **kwargs) * calculate_sign_mu(x, mu, **kwargs)


def smoothgrad_x_sign_mu_0(model_no_softmax, x, **kwargs):
    return smoothgrad_x_sign_mu(model_no_softmax, x, mu=0, **kwargs)


def smoothgrad_x_sign_mu_0_5(model_no_softmax, x, **kwargs):
    return smoothgrad_x_sign_mu(model_no_softmax, x, mu=0.5, **kwargs)


def smoothgrad_x_sign_mu_neg_0_5(model_no_softmax, x, **kwargs):
    return smoothgrad_x_sign_mu(model_no_softmax, x, mu=-0.5, **kwargs)


def vargrad(model_no_softmax, x, **kwargs):
    return calculate_explanation_innvestigate(model_no_softmax, x, method='vargrad', augment_by_n=50, noise_scale=0.2, **kwargs)


def deconvnet(model_no_softmax, x, **kwargs):
    return calculate_explanation_innvestigate(model_no_softmax, x, method='deconvnet', **kwargs)


def deconvnet_x_sign(model_no_softmax, x, **kwargs):
    g = deconvnet(model_no_softmax, x, **kwargs)
    s = np.nan_to_num(x / np.abs(x), nan=1.0)

    return g * s


def deconvnet_x_sign_mu(model_no_softmax, x, mu, batchmode=False, **kwargs):
    if batchmode:
        G = []
        S = []
        for xi in x:
            G.append(deconvnet(model_no_softmax, xi, **kwargs))
            S.append(calculate_sign_mu(xi, mu, **kwargs))
        return np.array(G) * np.array(S)
    else:
        return deconvnet(model_no_softmax, x, **kwargs) * calculate_sign_mu(x, mu, **kwargs)


def deconvnet_x_sign_mu_0(model_no_softmax, x, **kwargs):
    kwargs.pop('mu', None)  # Remove mu from kwargs to avoid conflict
    return deconvnet_x_sign_mu(model_no_softmax, x, mu=0, **kwargs)


def deconvnet_x_sign_mu_0_5(model_no_softmax, x, **kwargs):
    kwargs.pop('mu', None)  # Remove mu from kwargs to avoid conflict
    return deconvnet_x_sign_mu(model_no_softmax, x, mu=0.5, **kwargs)


def deconvnet_x_sign_mu_neg_0_5(model_no_softmax, x, **kwargs):
    kwargs.pop('mu', None)  # Remove mu from kwargs to avoid conflict
    return deconvnet_x_sign_mu(model_no_softmax, x, mu=-0.5, **kwargs)


def grad_cam(model_no_softmax, x, **kwargs):
    # Ensure x has batch dimension
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    
    # Check if input has proper batch dimension
    # For 1D timeseries: expected shape is (batch, time_steps, channels)
    # Don't add batch dimension if first dimension is 1 (already batched)
    if x.ndim == 3 and x.shape[0] != 1:
        x = np.expand_dims(x, axis=0)
    elif x.ndim == 2:
        # If 2D, assume it's (time_steps, channels) and add batch dimension
        x = np.expand_dims(x, axis=0)
    elif x.ndim > 4:
        # Handle case where x has too many dimensions
        raise ValueError(f"Input shape {x.shape} has too many dimensions for GradCAM")
    
    # Make layer_name parameter compatible with TensorFlow implementation
    last_conv_layer_name = kwargs.pop('layer_name', None)
    if not last_conv_layer_name:
        # Try to find the last convolutional layer as fallback
        for layer in reversed(model_no_softmax.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                print(f"Using automatically detected last conv layer: {last_conv_layer_name}")
                break
        
        if not last_conv_layer_name:
            raise ValueError("No layer_name provided and could not automatically detect a convolutional layer")
    
    # Default to resize=True for consistency with PyTorch implementation
    kwargs.setdefault('resize', True)
    
    # Handle target_class parameter name compatibility
    if 'target_class' in kwargs:
        kwargs['neuron_selection'] = kwargs.pop('target_class')
            
    result = calculate_grad_cam_relevancemap(x, model_no_softmax, last_conv_layer_name=last_conv_layer_name, **kwargs)
    
    # Return result without adding extra batch dimension for comparison script compatibility
    return result


def grad_cam_timeseries(model_no_softmax, x, **kwargs):
    """
    Adapted Grad-CAM for time series data.
    
    Args:
        model_no_softmax: Model to analyze
        x: Input data in various possible formats
        **kwargs: Additional arguments for calculate_grad_cam_relevancemap_timeseries
        
    Returns:
        GradCAM relevance map for time series data
    """
    # Ensure x is a numpy array
    if not isinstance(x, np.ndarray):
        x = np.array(x)
        
    # Debug input shape
    print(f"  DEBUG: Input shape to grad_cam_timeseries wrapper: {x.shape}")
    
    # Handle various input shapes to normalize to expected format: (batch, time_steps, channels)
    if x.ndim == 4:
        if x.shape[1] == 1:  # Shape like (1, 1, time_steps, channels)
            # Remove the redundant dimension, making it (1, time_steps, channels)
            x = x[:, 0]
            print(f"  DEBUG: Removed redundant dimension, new shape: {x.shape}")
        elif x.shape[3] == 1 and x.shape[0] == 1:  # Shape like (1, time_steps, channels, 1)
            # Remove the last dimension, making it (1, time_steps, channels)
            x = x[:, :, :, 0]
            print(f"  DEBUG: Removed trailing dimension, new shape: {x.shape}")
        else:
            # Unsupported shape
            raise ValueError(f"Unsupported 4D shape for timeseries: {x.shape}. Expected (1, 1, time_steps, channels) or (1, time_steps, channels, 1)")
    
    elif x.ndim == 3:
        # This is the expected format (batch, time_steps, channels)
        # Just verify batch size is 1 for consistency
        if x.shape[0] != 1:
            print(f"  DEBUG: Warning: Batch size is {x.shape[0]}, expected 1. Using first sample only.")
            x = x[0:1]  # Keep only the first sample but preserve batch dim
            
    elif x.ndim == 2:
        # Shape like (time_steps, channels) or (channels, time_steps)
        # Determine which dimension is likely time_steps based on relative sizes
        if x.shape[0] > x.shape[1] and x.shape[1] <= 12:  # Assume this is (time_steps, channels)
            x = np.expand_dims(x, axis=0)  # Add batch dimension -> (1, time_steps, channels)
            print(f"  DEBUG: Added batch dimension, new shape: {x.shape}")
        elif x.shape[1] > x.shape[0] and x.shape[0] <= 12:  # Assume this is (channels, time_steps)
            x = np.expand_dims(x.T, axis=0)  # Transpose and add batch -> (1, time_steps, channels)
            print(f"  DEBUG: Transposed and added batch dimension, new shape: {x.shape}")
        else:
            # Default case: assume (time_steps, channels)
            x = np.expand_dims(x, axis=0)
            print(f"  DEBUG: Added batch dimension, new shape: {x.shape}")
            
    elif x.ndim == 1:
        # Single channel, single batch (time_steps,)
        x = np.expand_dims(np.expand_dims(x, axis=0), axis=-1)  # -> (1, time_steps, 1)
        print(f"  DEBUG: Added batch and channel dimensions, new shape: {x.shape}")
        
    else:
        # Handle case where x has too many dimensions
        raise ValueError(f"Input shape {x.shape} has too many dimensions for GradCAM Timeseries")
    
    # Make layer_name parameter compatible with GradCAM implementation
    last_conv_layer_name = kwargs.pop('layer_name', None)
    if not last_conv_layer_name:
        last_conv_layer_name = kwargs.pop('last_conv_layer_name', None)
        if not last_conv_layer_name:
            # Try to find a Conv1D layer as fallback
            for layer in reversed(model_no_softmax.layers):
                if isinstance(layer, tf.keras.layers.Conv1D):
                    last_conv_layer_name = layer.name
                    print(f"  DEBUG: Using automatically detected Conv1D layer: {last_conv_layer_name}")
                    break
                    
            if not last_conv_layer_name:
                # Fall back to the last layer with 'conv' in the name
                for layer in reversed(model_no_softmax.layers):
                    if 'conv' in layer.name.lower():
                        last_conv_layer_name = layer.name
                        print(f"  DEBUG: Falling back to layer with 'conv' in name: {last_conv_layer_name}")
                        break
                        
            if not last_conv_layer_name:
                raise ValueError("No layer_name provided and could not automatically detect a convolutional layer")
                
    # Default to resize=True
    kwargs.setdefault('resize', True)
    
    # Call the implementation with normalized input
    return calculate_grad_cam_relevancemap_timeseries(x, model_no_softmax, last_conv_layer_name=last_conv_layer_name, **kwargs)


def grad_cam_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    # Ensure x has batch dimension
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if x.ndim == 3:
        x = np.expand_dims(x, axis=0)
    elif x.ndim > 4:
        # Handle case where x has too many dimensions
        raise ValueError(f"Input shape {x.shape} has too many dimensions for GradCAM VGG16")
        
    return calculate_grad_cam_relevancemap(x, model_no_softmax, last_conv_layer_name='block5_conv3', resize=True, **kwargs)


def guided_grad_cam_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    # Ensure x has batch dimension
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if x.ndim == 3:
        x = np.expand_dims(x, axis=0)
    elif x.ndim > 4:
        # Handle case where x has too many dimensions
        raise ValueError(f"Input shape {x.shape} has too many dimensions for Guided GradCAM")
        
    gc = grad_cam_VGG16ILSVRC(model_no_softmax, x, **kwargs)
    gbp = guided_backprop_on_guided_model(model_no_softmax, x, layer_name='block5_conv3')

    # Handle broadcasting - expand gc to match gbp dimensions if needed
    if gc.ndim < gbp.ndim:
        gc = np.expand_dims(gc, axis=-1)  # Add channel dimension
        gc = np.repeat(gc, gbp.shape[-1], axis=-1)  # Repeat for all channels
    
    return gbp * gc


def grad_cam_VGG16MITPL365(model_no_softmax, x, **kwargs):
    # Ensure x has batch dimension
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if x.ndim == 3:
        x = np.expand_dims(x, axis=0)
    elif x.ndim > 4:
        # Handle case where x has too many dimensions
        raise ValueError(f"Input shape {x.shape} has too many dimensions for GradCAM VGG16 MITPL365")
        
    return calculate_grad_cam_relevancemap(x, model_no_softmax, last_conv_layer_name='relu5_3', resize=True, **kwargs)


def guided_grad_cam_VGG16MITPL365(model_no_softmax, x, **kwargs):
    # Ensure x has batch dimension
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if x.ndim == 3:
        x = np.expand_dims(x, axis=0)
    elif x.ndim > 4:
        # Handle case where x has too many dimensions
        raise ValueError(f"Input shape {x.shape} has too many dimensions for Guided GradCAM VGG16 MITPL365")
        
    gc = grad_cam_VGG16MITPL365(model_no_softmax, x, **kwargs)
    gbp = guided_backprop_on_guided_model(model_no_softmax, x, layer_name='relu5_3')

    return gbp * gc


def grad_cam_MNISTCNN(model_no_softmax, x, batchmode=False, **kwargs):
    if batchmode:
        H = []
        for xi in x:
            # Ensure each individual example has batch dimension
            xi_batched = np.expand_dims(xi, axis=0)
            H.append(calculate_grad_cam_relevancemap(xi_batched, model_no_softmax, 
                                                     last_conv_layer_name='conv2d_1', 
                                                     resize=True, **kwargs))
        return np.array(H)
    else:
        # Ensure x has batch dimension
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if x.ndim == 3:
            x = np.expand_dims(x, axis=0)
        elif x.ndim > 4:
            # Handle case where x has too many dimensions
            raise ValueError(f"Input shape {x.shape} has too many dimensions for GradCAM MNIST CNN")
            
        return calculate_grad_cam_relevancemap(x, model_no_softmax, 
                                              last_conv_layer_name='conv2d_1', 
                                              resize=True, **kwargs)


def guided_grad_cam_MNISTCNN(model_no_softmax, x, batchmode=False, **kwargs):
    if batchmode:
        gc = grad_cam_MNISTCNN(model_no_softmax, x, batchmode=True, **kwargs)
        gbp = guided_backprop_on_guided_model(model_no_softmax, x, layer_name='conv2d_1')
    else:
        # Ensure x has batch dimension
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if x.ndim == 3:
            x = np.expand_dims(x, axis=0)
        elif x.ndim > 4:
            # Handle case where x has too many dimensions
            raise ValueError(f"Input shape {x.shape} has too many dimensions for Guided GradCAM MNIST CNN")
            
        gc = grad_cam_MNISTCNN(model_no_softmax, x, **kwargs)
        gbp = guided_backprop_on_guided_model(model_no_softmax, x, layer_name='conv2d_1')

    return gbp * gc


def lrp_z(model_no_softmax, x, **kwargs):
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.z', **kwargs)


def lrpsign_z(model_no_softmax, x, **kwargs):
    return lrp_z(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def zblrp_z_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    return lrp_z(model_no_softmax, x, input_layer_rule='Bounded', low=-123.68, high=151.061, **kwargs)


def w2lrp_z(model_no_softmax, x, **kwargs):
    return lrp_z(model_no_softmax, x, input_layer_rule='WSquare', **kwargs)


def flatlrp_z(model_no_softmax, x, **kwargs):
    return lrp_z(model_no_softmax, x, input_layer_rule='Flat', **kwargs)


def lrp_epsilon_0_001(model_no_softmax, x, **kwargs):
    kwargs.pop('epsilon', None)  # Remove epsilon from kwargs to avoid conflict
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.epsilon', epsilon=0.001, **kwargs)


def lrpsign_epsilon_0_001(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_001(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def zblrp_epsilon_0_001_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_001(model_no_softmax, x, input_layer_rule='Bounded', low=-123.68, high=151.061, **kwargs)


def lrpz_epsilon_0_001(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_001(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def lrp_epsilon_0_01(model_no_softmax, x, **kwargs):
    kwargs.pop('epsilon', None)  # Remove epsilon from kwargs to avoid conflict
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.epsilon', epsilon=0.01, **kwargs)


def lrpsign_epsilon_0_01(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_01(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def zblrp_epsilon_0_01_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_01(model_no_softmax, x, input_layer_rule='Bounded', low=-123.68, high=151.061, **kwargs)


def w2lrp_epsilon_0_01(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_01(model_no_softmax, x, input_layer_rule='WSquare', **kwargs)


def flatlrp_epsilon_0_01(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_01(model_no_softmax, x, input_layer_rule='Flat', **kwargs)


def lrpz_epsilon_0_01(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_01(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def lrp_epsilon_0_1(model_no_softmax, x, **kwargs):
    kwargs.pop('epsilon', None)  # Remove epsilon from kwargs to avoid conflict
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.epsilon', epsilon=0.1, **kwargs)


def lrpsign_epsilon_0_1(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_1(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def zblrp_epsilon_0_1_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_1(model_no_softmax, x, input_layer_rule='Bounded', low=-123.68, high=151.061, **kwargs)


def w2lrp_epsilon_0_1(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_1(model_no_softmax, x, input_layer_rule='WSquare', **kwargs)


def flatlrp_epsilon_0_1(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_1(model_no_softmax, x, input_layer_rule='Flat', **kwargs)


def lrpz_epsilon_0_1(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_1(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def lrp_epsilon_0_2(model_no_softmax, x, **kwargs):
    kwargs.pop('epsilon', None)  # Remove epsilon from kwargs to avoid conflict
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.epsilon', epsilon=0.2, **kwargs)


def zblrp_epsilon_0_2_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_2(model_no_softmax, x, input_layer_rule='Bounded', low=-123.68, high=151.061, **kwargs)


def lrpsign_epsilon_0_2(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_2(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def lrpz_epsilon_0_2(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_2(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def lrp_epsilon_0_5(model_no_softmax, x, **kwargs):
    kwargs.pop('epsilon', None)  # Remove epsilon from kwargs to avoid conflict
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.epsilon', epsilon=0.5, **kwargs)


def zblrp_epsilon_0_5_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_5(model_no_softmax, x, input_layer_rule='Bounded', low=-123.68, high=151.061, **kwargs)


def lrpsign_epsilon_0_5(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_5(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def lrpz_epsilon_0_5(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_5(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def lrp_epsilon_1(model_no_softmax, x, **kwargs):
    kwargs.pop('epsilon', None)  # Remove epsilon from kwargs to avoid conflict
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.epsilon', epsilon=1, **kwargs)


def lrpsign_epsilon_1(model_no_softmax, x, **kwargs):
    return lrp_epsilon_1(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def zblrp_epsilon_1_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    return lrp_epsilon_1(model_no_softmax, x, input_layer_rule='Bounded', low=-123.68, high=151.061, **kwargs)


def w2lrp_epsilon_1(model_no_softmax, x, **kwargs):
    return lrp_epsilon_1(model_no_softmax, x, input_layer_rule='WSquare', **kwargs)


def flatlrp_epsilon_1(model_no_softmax, x, **kwargs):
    return lrp_epsilon_1(model_no_softmax, x, input_layer_rule='Flat', **kwargs)


def lrpz_epsilon_1(model_no_softmax, x, **kwargs):
    return lrp_epsilon_1(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def lrp_epsilon_5(model_no_softmax, x, **kwargs):
    kwargs.pop('epsilon', None)  # Remove epsilon from kwargs to avoid conflict
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.epsilon', epsilon=5, **kwargs)


def zblrp_epsilon_5_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    return lrp_epsilon_5(model_no_softmax, x, input_layer_rule='Bounded', low=-123.68, high=151.061, **kwargs)


def lrpsign_epsilon_5(model_no_softmax, x, **kwargs):
    return lrp_epsilon_5(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def lrpz_epsilon_5(model_no_softmax, x, **kwargs):
    return lrp_epsilon_5(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def lrp_epsilon_10(model_no_softmax, x, **kwargs):
    kwargs.pop('epsilon', None)  # Remove epsilon from kwargs to avoid conflict
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.epsilon', epsilon=10, **kwargs)


def zblrp_epsilon_10_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    return lrp_epsilon_10(model_no_softmax, x, input_layer_rule='Bounded', low=-123.68, high=151.061, **kwargs)


def w2lrp_epsilon_10(model_no_softmax, x, **kwargs):
    return lrp_epsilon_10(model_no_softmax, x, input_layer_rule='WSquare', **kwargs)


def flatlrp_epsilon_10(model_no_softmax, x, **kwargs):
    return lrp_epsilon_10(model_no_softmax, x, input_layer_rule='Flat', **kwargs)


def lrpsign_epsilon_10(model_no_softmax, x, **kwargs):
    return lrp_epsilon_10(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def lrpz_epsilon_10(model_no_softmax, x, **kwargs):
    return lrp_epsilon_10(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def lrp_epsilon_20(model_no_softmax, x, **kwargs):
    kwargs.pop('epsilon', None)  # Remove epsilon from kwargs to avoid conflict
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.epsilon', epsilon=20, **kwargs)


def lrpsign_epsilon_20(model_no_softmax, x, **kwargs):
    return lrp_epsilon_20(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)



def lrpz_epsilon_20(model_no_softmax, x, **kwargs):
    return lrp_epsilon_20(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def zblrp_epsilon_20_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    return lrp_epsilon_20(model_no_softmax, x, input_layer_rule='Bounded', low=-123.68, high=151.061, **kwargs)


def w2lrp_epsilon_20(model_no_softmax, x, **kwargs):
    return lrp_epsilon_20(model_no_softmax, x, input_layer_rule='WSquare', **kwargs)


def flatlrp_epsilon_20(model_no_softmax, x, **kwargs):
    return lrp_epsilon_20(model_no_softmax, x, input_layer_rule='Flat', **kwargs)


def lrp_epsilon_50(model_no_softmax, x, **kwargs):
    kwargs.pop('epsilon', None)  # Remove epsilon from kwargs to avoid conflict
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.epsilon', epsilon=50, **kwargs)


def lrpsign_epsilon_50(model_no_softmax, x, **kwargs):
    return lrp_epsilon_50(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def lrpz_epsilon_50(model_no_softmax, x, **kwargs):
    return lrp_epsilon_50(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def lrp_epsilon_75(model_no_softmax, x, **kwargs):
    kwargs.pop('epsilon', None)  # Remove epsilon from kwargs to avoid conflict
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.epsilon', epsilon=75, **kwargs)


def lrpsign_epsilon_75(model_no_softmax, x, **kwargs):
    return lrp_epsilon_75(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def lrpz_epsilon_75(model_no_softmax, x, **kwargs):
    return lrp_epsilon_75(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def lrp_epsilon_100(model_no_softmax, x, **kwargs):
    kwargs.pop('epsilon', None)  # Remove epsilon from kwargs to avoid conflict
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.epsilon', epsilon=100, **kwargs)


def lrpsign_epsilon_100(model_no_softmax, x, **kwargs):
    return lrp_epsilon_100(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def lrpsign_epsilon_100_mu_0(model_no_softmax, x, **kwargs):
    return lrp_epsilon_100(model_no_softmax, x, input_layer_rule='SIGNmu', mu=0, **kwargs)


def lrpsign_epsilon_100_mu_0_5(model_no_softmax, x, **kwargs):
    return lrp_epsilon_100(model_no_softmax, x, input_layer_rule='SIGNmu', mu=0.5, **kwargs)


def lrpsign_epsilon_100_mu_neg_0_5(model_no_softmax, x, **kwargs):
    return lrp_epsilon_100(model_no_softmax, x, input_layer_rule='SIGNmu', mu=-0.5, **kwargs)


def lrpz_epsilon_100(model_no_softmax, x, **kwargs):
    return lrp_epsilon_100(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def zblrp_epsilon_100_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    return lrp_epsilon_100(model_no_softmax, x, input_layer_rule='Bounded', low=-123.68, high=151.061, **kwargs)


def w2lrp_epsilon_100(model_no_softmax, x, **kwargs):
    return lrp_epsilon_100(model_no_softmax, x, input_layer_rule='WSquare', **kwargs)


def flatlrp_epsilon_100(model_no_softmax, x, **kwargs):
    return lrp_epsilon_100(model_no_softmax, x, input_layer_rule='Flat', **kwargs)


def lrp_epsilon_0_1_std_x(model_no_softmax, x, **kwargs):
    kwargs.pop('stdfactor', None)  # Remove stdfactor from kwargs to avoid conflict
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.stdxepsilon', stdfactor=0.1, **kwargs)


def lrpsign_epsilon_0_1_std_x(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_1_std_x(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def lrpz_epsilon_0_1_std_x(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_1_std_x(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def zblrp_epsilon_0_1_std_x_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_1_std_x(model_no_softmax, x, input_layer_rule='Bounded', low=-123.68, high=151.061, **kwargs)


def w2lrp_epsilon_0_1_std_x(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_1_std_x(model_no_softmax, x, input_layer_rule='WSquare', **kwargs)


def flatlrp_epsilon_0_1_std_x(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_1_std_x(model_no_softmax, x, input_layer_rule='Flat', **kwargs)


def lrp_epsilon_0_25_std_x(model_no_softmax, x, **kwargs):
    kwargs.pop('stdfactor', None)  # Remove stdfactor from kwargs to avoid conflict
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.stdxepsilon', stdfactor=0.25, **kwargs)


def lrpsign_epsilon_0_25_std_x(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_25_std_x(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def lrpz_epsilon_0_25_std_x(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_25_std_x(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def zblrp_epsilon_0_25_std_x_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_25_std_x(model_no_softmax, x, input_layer_rule='Bounded', low=-123.68, high=151.061, **kwargs)


def w2lrp_epsilon_0_25_std_x(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_25_std_x(model_no_softmax, x, input_layer_rule='WSquare', **kwargs)


def flatlrp_epsilon_0_25_std_x(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_25_std_x(model_no_softmax, x, input_layer_rule='Flat', **kwargs)


def lrpsign_epsilon_0_25_std_x_mu_0(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_25_std_x(model_no_softmax, x, input_layer_rule='SIGNmu', mu=0, **kwargs)


def lrpsign_epsilon_0_25_std_x_mu_0_5(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_25_std_x(model_no_softmax, x, input_layer_rule='SIGNmu', mu=0.5, **kwargs)


def lrpsign_epsilon_0_25_std_x_mu_neg_0_5(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_25_std_x(model_no_softmax, x, input_layer_rule='SIGNmu', mu=-0.5, **kwargs)


def lrp_epsilon_0_5_std_x(model_no_softmax, x, **kwargs):
    kwargs.pop('stdfactor', None)  # Remove stdfactor from kwargs to avoid conflict
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.stdxepsilon', stdfactor=0.5, **kwargs)


def lrpsign_epsilon_0_5_std_x(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_5_std_x(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def lrpz_epsilon_0_5_std_x(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_5_std_x(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def zblrp_epsilon_0_5_std_x_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_5_std_x(model_no_softmax, x, input_layer_rule='Bounded', low=-123.68, high=151.061, **kwargs)


def w2lrp_epsilon_0_5_std_x(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_5_std_x(model_no_softmax, x, input_layer_rule='WSquare', **kwargs)


def flatlrp_epsilon_0_5_std_x(model_no_softmax, x, **kwargs):
    return lrp_epsilon_0_5_std_x(model_no_softmax, x, input_layer_rule='Flat', **kwargs)


def lrp_epsilon_1_std_x(model_no_softmax, x, **kwargs):
    kwargs.pop('stdfactor', None)  # Remove stdfactor from kwargs to avoid conflict
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.stdxepsilon', stdfactor=1.0, **kwargs)


def lrpsign_epsilon_1_std_x(model_no_softmax, x, **kwargs):
    return lrp_epsilon_1_std_x(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def lrpz_epsilon_1_std_x(model_no_softmax, x, **kwargs):
    return lrp_epsilon_1_std_x(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def lrp_epsilon_2_std_x(model_no_softmax, x, **kwargs):
    kwargs.pop('stdfactor', None)  # Remove stdfactor from kwargs to avoid conflict
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.stdxepsilon', stdfactor=2.0, **kwargs)


def lrpsign_epsilon_2_std_x(model_no_softmax, x, **kwargs):
    return lrp_epsilon_2_std_x(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def lrpz_epsilon_2_std_x(model_no_softmax, x, **kwargs):
    return lrp_epsilon_2_std_x(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def lrp_epsilon_3_std_x(model_no_softmax, x, **kwargs):
    kwargs.pop('stdfactor', None)  # Remove stdfactor from kwargs to avoid conflict
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.stdxepsilon', stdfactor=3.0, **kwargs)


def lrpsign_epsilon_3_std_x(model_no_softmax, x, **kwargs):
    return lrp_epsilon_3_std_x(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def lrpz_epsilon_3_std_x(model_no_softmax, x, **kwargs):
    return lrp_epsilon_3_std_x(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def lrp_alpha_1_beta_0(model_no_softmax, x, **kwargs):
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.alpha_1_beta_0', **kwargs)


def lrpsign_alpha_1_beta_0(model_no_softmax, x, **kwargs):
    return lrp_alpha_1_beta_0(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def lrpz_alpha_1_beta_0(model_no_softmax, x, **kwargs):
    return lrp_alpha_1_beta_0(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def zblrp_alpha_1_beta_0_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    return lrp_alpha_1_beta_0(model_no_softmax, x, input_layer_rule='Bounded', low=-123.68, high=151.061, **kwargs)


def w2lrp_alpha_1_beta_0(model_no_softmax, x, **kwargs):
    return lrp_alpha_1_beta_0(model_no_softmax, x, input_layer_rule='WSquare', **kwargs)


def flatlrp_alpha_1_beta_0(model_no_softmax, x, **kwargs):
    return lrp_alpha_1_beta_0(model_no_softmax, x, input_layer_rule='Flat', **kwargs)


def lrp_sequential_composite_a(model_no_softmax, x, **kwargs):
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.sequential_composite_a', **kwargs)


def lrpsign_sequential_composite_a(model_no_softmax, x, **kwargs):
    return lrp_sequential_composite_a(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def lrpz_sequential_composite_a(model_no_softmax, x, **kwargs):
    return lrp_sequential_composite_a(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def zblrp_sequential_composite_a_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    return lrp_sequential_composite_a(model_no_softmax, x, input_layer_rule='Bounded', low=-123.68, high=151.061, **kwargs)


def w2lrp_sequential_composite_a(model_no_softmax, x, **kwargs):
    return lrp_sequential_composite_a(model_no_softmax, x, input_layer_rule='WSquare', **kwargs)


def flatlrp_sequential_composite_a(model_no_softmax, x, **kwargs):
    return lrp_sequential_composite_a(model_no_softmax, x, input_layer_rule='Flat', **kwargs)


def lrp_sequential_composite_b(model_no_softmax, x, **kwargs):
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.sequential_composite_b', **kwargs)


def lrpsign_sequential_composite_b(model_no_softmax, x, **kwargs):
    return lrp_sequential_composite_b(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)


def lrpz_sequential_composite_b(model_no_softmax, x, **kwargs):
    return lrp_sequential_composite_b(model_no_softmax, x, input_layer_rule='Z', **kwargs)


def zblrp_sequential_composite_b_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    return lrp_sequential_composite_b(model_no_softmax, x, input_layer_rule='Bounded', low=-123.68, high=151.061, **kwargs)


def w2lrp_sequential_composite_b(model_no_softmax, x, **kwargs):
    return lrp_sequential_composite_b(model_no_softmax, x, input_layer_rule='WSquare', **kwargs)


def flatlrp_sequential_composite_b(model_no_softmax, x, **kwargs):
    return lrp_sequential_composite_b(model_no_softmax, x, input_layer_rule='Flat', **kwargs)


# Additional LRP methods for comparison script compatibility
def lrp_gamma(model_no_softmax, x, **kwargs):
    """LRP Gamma method"""
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.gamma', **kwargs)

def lrp_flat(model_no_softmax, x, **kwargs):
    """LRP Flat method"""
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.flat', **kwargs)

def lrp_w_square(model_no_softmax, x, **kwargs):
    """LRP W-Square method"""
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.w_square', **kwargs)

def lrp_z_plus(model_no_softmax, x, **kwargs):
    """LRP Z-Plus method"""
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.z_plus', **kwargs)

def lrp_alpha_2_beta_1(model_no_softmax, x, **kwargs):
    """LRP Alpha-2 Beta-1 method"""
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.alpha_2_beta_1', **kwargs)

def deeplift_method(model_no_softmax, x, **kwargs):
    """DeepLIFT method"""
    return calculate_explanation_innvestigate(model_no_softmax, x, method='deep_lift', **kwargs)

def lrp_epsilon_wrapper(model_no_softmax, x, **kwargs):
    """LRP Epsilon method with configurable epsilon parameter"""
    epsilon = kwargs.pop('epsilon', 0.1)  # Remove epsilon from kwargs to avoid conflict
    return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.epsilon', epsilon=epsilon, **kwargs)


def deconvnet_x_sign_mu_wrapper(model_no_softmax, x, **kwargs):
    """Deconvnet x Sign Mu method with configurable mu parameter"""
    mu = kwargs.pop('mu', 0.0)  # Remove mu from kwargs to avoid conflict, default to 0.0
    return deconvnet_x_sign_mu(model_no_softmax, x, mu=mu, **kwargs)


def gradient_x_sign_mu_wrapper(model_no_softmax, x, **kwargs):
    """Gradient x Sign Mu method with configurable mu parameter"""
    mu = kwargs.pop('mu', 0.0)  # Remove mu from kwargs to avoid conflict, default to 0.0
    return gradient_x_sign_mu(model_no_softmax, x, mu=mu, **kwargs)


def guided_backprop_x_sign_mu_wrapper(model_no_softmax, x, **kwargs):
    """Guided Backprop x Sign Mu method with configurable mu parameter"""
    mu = kwargs.pop('mu', 0.0)  # Remove mu from kwargs to avoid conflict, default to 0.0
    return guided_backprop_x_sign_mu(model_no_softmax, x, mu=mu, **kwargs)


# Missing _x_input variations to match PyTorch
def integrated_gradients_x_input(model_no_softmax, x, **kwargs):
    """Integrated Gradients times input"""
    ig = integrated_gradients(model_no_softmax, x, **kwargs)
    return ig * x


def grad_cam_x_input(model_no_softmax, x, **kwargs):
    """Grad-CAM times input"""
    gc = grad_cam(model_no_softmax, x, **kwargs)
    # Handle broadcasting for different shapes
    if gc.shape != x.shape:
        if x.ndim == 4 and gc.ndim == 2:  # (B,H,W,C) vs (H,W)
            gc = np.broadcast_to(gc[:, :, None], x.shape)
        elif x.ndim == 3 and gc.ndim == 2:  # (H,W,C) vs (H,W)
            gc = np.broadcast_to(gc[:, :, None], x.shape)
    return gc * x


def guided_backprop_x_input(model_no_softmax, x, **kwargs):
    """Guided Backprop times input"""
    g = guided_backprop(model_no_softmax, x, **kwargs)
    return g * x


def deconvnet_x_input(model_no_softmax, x, **kwargs):
    """DeconvNet times input"""
    d = deconvnet(model_no_softmax, x, **kwargs)
    return d * x


def vargrad_x_input(model_no_softmax, x, **kwargs):
    """VarGrad times input"""
    v = vargrad(model_no_softmax, x, **kwargs)
    return v * x


def deeplift_x_input(model_no_softmax, x, **kwargs):
    """DeepLIFT times input"""
    dl = deeplift_method(model_no_softmax, x, **kwargs)
    return dl * x


# Missing _x_input_x_sign combinations to match PyTorch
def gradient_x_input_x_sign(model_no_softmax, x, **kwargs):
    """Gradient times input times sign"""
    g = gradient(model_no_softmax, x, **kwargs)
    s = np.nan_to_num(x / np.abs(x), nan=1.0)
    return g * x * s


def smoothgrad_x_input_x_sign(model_no_softmax, x, **kwargs):
    """SmoothGrad times input times sign"""
    sg = smoothgrad(model_no_softmax, x, **kwargs)
    s = np.nan_to_num(x / np.abs(x), nan=1.0)
    return sg * x * s


def integrated_gradients_x_input_x_sign(model_no_softmax, x, **kwargs):
    """Integrated Gradients times input times sign"""
    ig = integrated_gradients(model_no_softmax, x, **kwargs)
    s = np.nan_to_num(x / np.abs(x), nan=1.0)
    return ig * x * s


def grad_cam_x_input_x_sign(model_no_softmax, x, **kwargs):
    """Grad-CAM times input times sign"""
    gc = grad_cam(model_no_softmax, x, **kwargs)
    s = np.nan_to_num(x / np.abs(x), nan=1.0)
    # Handle broadcasting for different shapes
    if gc.shape != x.shape:
        if x.ndim == 4 and gc.ndim == 2:  # (B,H,W,C) vs (H,W)
            gc = np.broadcast_to(gc[:, :, None], x.shape)
        elif x.ndim == 3 and gc.ndim == 2:  # (H,W,C) vs (H,W)
            gc = np.broadcast_to(gc[:, :, None], x.shape)
    return gc * x * s


def guided_backprop_x_input_x_sign(model_no_softmax, x, **kwargs):
    """Guided Backprop times input times sign"""
    g = guided_backprop(model_no_softmax, x, **kwargs)
    s = np.nan_to_num(x / np.abs(x), nan=1.0)
    return g * x * s


def deconvnet_x_input_x_sign(model_no_softmax, x, **kwargs):
    """DeconvNet times input times sign"""
    d = deconvnet(model_no_softmax, x, **kwargs)
    s = np.nan_to_num(x / np.abs(x), nan=1.0)
    return d * x * s


def vargrad_x_input_x_sign(model_no_softmax, x, **kwargs):
    """VarGrad times input times sign"""
    v = vargrad(model_no_softmax, x, **kwargs)
    s = np.nan_to_num(x / np.abs(x), nan=1.0)
    return v * x * s


def deeplift_x_input_x_sign(model_no_softmax, x, **kwargs):
    """DeepLIFT times input times sign"""
    dl = deeplift_method(model_no_softmax, x, **kwargs)
    s = np.nan_to_num(x / np.abs(x), nan=1.0)
    return dl * x * s


# Missing LRP _x_input and _x_sign variations
def lrp_gamma_x_input(model_no_softmax, x, **kwargs):
    """LRP Gamma times input"""
    lrp = lrp_gamma(model_no_softmax, x, **kwargs)
    return lrp * x


def lrp_gamma_x_sign(model_no_softmax, x, **kwargs):
    """LRP Gamma times sign"""
    lrp = lrp_gamma(model_no_softmax, x, **kwargs)
    s = np.nan_to_num(x / np.abs(x), nan=1.0)
    return lrp * s


def lrp_gamma_x_input_x_sign(model_no_softmax, x, **kwargs):
    """LRP Gamma times input times sign"""
    lrp = lrp_gamma(model_no_softmax, x, **kwargs)
    s = np.nan_to_num(x / np.abs(x), nan=1.0)
    return lrp * x * s


def lrp_flat_x_input(model_no_softmax, x, **kwargs):
    """LRP Flat times input"""
    lrp = lrp_flat(model_no_softmax, x, **kwargs)
    return lrp * x


def lrp_flat_x_sign(model_no_softmax, x, **kwargs):
    """LRP Flat times sign"""
    lrp = lrp_flat(model_no_softmax, x, **kwargs)
    s = np.nan_to_num(x / np.abs(x), nan=1.0)
    return lrp * s


def lrp_flat_x_input_x_sign(model_no_softmax, x, **kwargs):
    """LRP Flat times input times sign"""
    lrp = lrp_flat(model_no_softmax, x, **kwargs)
    s = np.nan_to_num(x / np.abs(x), nan=1.0)
    return lrp * x * s


def lrp_w_square_x_input(model_no_softmax, x, **kwargs):
    """LRP W-Square times input"""
    lrp = lrp_w_square(model_no_softmax, x, **kwargs)
    return lrp * x


def lrp_w_square_x_sign(model_no_softmax, x, **kwargs):
    """LRP W-Square times sign"""
    lrp = lrp_w_square(model_no_softmax, x, **kwargs)
    s = np.nan_to_num(x / np.abs(x), nan=1.0)
    return lrp * s


def lrp_w_square_x_input_x_sign(model_no_softmax, x, **kwargs):
    """LRP W-Square times input times sign"""
    lrp = lrp_w_square(model_no_softmax, x, **kwargs)
    s = np.nan_to_num(x / np.abs(x), nan=1.0)
    return lrp * x * s


def lrp_z_plus_x_input(model_no_softmax, x, **kwargs):
    """LRP Z-Plus times input"""
    lrp = lrp_z_plus(model_no_softmax, x, **kwargs)
    return lrp * x


def lrp_z_plus_x_sign(model_no_softmax, x, **kwargs):
    """LRP Z-Plus times sign"""
    lrp = lrp_z_plus(model_no_softmax, x, **kwargs)
    s = np.nan_to_num(x / np.abs(x), nan=1.0)
    return lrp * s


def lrp_z_plus_x_input_x_sign(model_no_softmax, x, **kwargs):
    """LRP Z-Plus times input times sign"""
    lrp = lrp_z_plus(model_no_softmax, x, **kwargs)
    s = np.nan_to_num(x / np.abs(x), nan=1.0)
    return lrp * x * s


def lrp_alpha_1_beta_0_x_input(model_no_softmax, x, **kwargs):
    """LRP Alpha-1 Beta-0 times input"""
    lrp = lrp_alpha_1_beta_0(model_no_softmax, x, **kwargs)
    return lrp * x


def lrp_alpha_1_beta_0_x_sign(model_no_softmax, x, **kwargs):
    """LRP Alpha-1 Beta-0 times sign"""
    lrp = lrp_alpha_1_beta_0(model_no_softmax, x, **kwargs)
    s = np.nan_to_num(x / np.abs(x), nan=1.0)
    return lrp * s


def lrp_alpha_1_beta_0_x_input_x_sign(model_no_softmax, x, **kwargs):
    """LRP Alpha-1 Beta-0 times input times sign"""
    lrp = lrp_alpha_1_beta_0(model_no_softmax, x, **kwargs)
    s = np.nan_to_num(x / np.abs(x), nan=1.0)
    return lrp * x * s


def lrp_alpha_2_beta_1_x_input(model_no_softmax, x, **kwargs):
    """LRP Alpha-2 Beta-1 times input"""
    lrp = lrp_alpha_2_beta_1(model_no_softmax, x, **kwargs)
    return lrp * x


def lrp_alpha_2_beta_1_x_sign(model_no_softmax, x, **kwargs):
    """LRP Alpha-2 Beta-1 times sign"""
    lrp = lrp_alpha_2_beta_1(model_no_softmax, x, **kwargs)
    s = np.nan_to_num(x / np.abs(x), nan=1.0)
    return lrp * s


def lrp_alpha_2_beta_1_x_input_x_sign(model_no_softmax, x, **kwargs):
    """LRP Alpha-2 Beta-1 times input times sign"""
    lrp = lrp_alpha_2_beta_1(model_no_softmax, x, **kwargs)
    s = np.nan_to_num(x / np.abs(x), nan=1.0)
    return lrp * x * s


def lrp_epsilon_0_1_x_input(model_no_softmax, x, **kwargs):
    """LRP Epsilon 0.1 times input"""
    lrp = lrp_epsilon_0_1(model_no_softmax, x, **kwargs)
    return lrp * x


def lrp_epsilon_0_1_x_sign(model_no_softmax, x, **kwargs):
    """LRP Epsilon 0.1 times sign"""
    lrp = lrp_epsilon_0_1(model_no_softmax, x, **kwargs)
    s = np.nan_to_num(x / np.abs(x), nan=1.0)
    return lrp * s


def lrp_epsilon_0_1_x_input_x_sign(model_no_softmax, x, **kwargs):
    """LRP Epsilon 0.1 times input times sign"""
    lrp = lrp_epsilon_0_1(model_no_softmax, x, **kwargs)
    s = np.nan_to_num(x / np.abs(x), nan=1.0)
    return lrp * x * s


def deeplift_x_sign(model_no_softmax, x, **kwargs):
    """DeepLIFT times sign"""
    dl = deeplift_method(model_no_softmax, x, **kwargs)
    s = np.nan_to_num(x / np.abs(x), nan=1.0)
    return dl * s


def vargrad_x_sign(model_no_softmax, x, **kwargs):
    """VarGrad times sign"""
    v = vargrad(model_no_softmax, x, **kwargs)
    s = np.nan_to_num(x / np.abs(x), nan=1.0)
    return v * s


def integrated_gradients_x_sign(model_no_softmax, x, **kwargs):
    """Integrated Gradients times sign"""
    ig = integrated_gradients(model_no_softmax, x, **kwargs)
    s = np.nan_to_num(x / np.abs(x), nan=1.0)
    return ig * s


def grad_cam_x_sign(model_no_softmax, x, **kwargs):
    """Grad-CAM times sign"""
    gc = grad_cam(model_no_softmax, x, **kwargs)
    s = np.nan_to_num(x / np.abs(x), nan=1.0)
    # Handle broadcasting for different shapes
    if gc.shape != s.shape:
        if x.ndim == 4 and gc.ndim == 2:  # (B,H,W,C) vs (H,W)
            gc = np.broadcast_to(gc[:, :, None], s.shape)
        elif x.ndim == 3 and gc.ndim == 2:  # (H,W,C) vs (H,W)
            gc = np.broadcast_to(gc[:, :, None], s.shape)
    return gc * s


def calculate_relevancemap(m, x, model_no_softmax, **kwargs):
    # Handle dot notation for LRP methods used by comparison script
    method_mapping = {
        'lrp.epsilon': 'lrp_epsilon_wrapper',  # Use a wrapper that handles epsilon parameter
        'lrp.gamma': 'lrp_gamma',
        'lrp.flat': 'lrp_flat',
        'lrp.w_square': 'lrp_w_square',
        'lrp.z_plus': 'lrp_z_plus',
        'lrp.alpha_1_beta_0': 'lrp_alpha_1_beta_0',
        'lrp.alpha_2_beta_1': 'lrp_alpha_2_beta_1',
        'lrp.stdxepsilon': 'lrp_epsilon_0_1_std_x',
        'deeplift': 'deeplift_method',
        'deconvnet_x_sign_mu': 'deconvnet_x_sign_mu_wrapper',
        'gradient_x_sign_mu': 'gradient_x_sign_mu_wrapper',
        'guided_backprop_x_sign_mu': 'guided_backprop_x_sign_mu_wrapper',
    }
    
    # Map dot notation to actual function names
    actual_method = method_mapping.get(m, m)
    f = eval(actual_method)
    return f(model_no_softmax, x, **kwargs)


def calculate_relevancemaps(m, X, model_no_softmax, **kwargs):
    Rs = []
    for x in X:
        R = calculate_relevancemap(m, x, model_no_softmax, **kwargs)
        Rs.append(R)

    return np.array(Rs)
