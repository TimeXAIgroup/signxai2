"""
TensorFlow-specific utility functions for SignXAI2.
"""

import os
import numpy as np
import requests
from PIL import Image

try:
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.preprocessing import image as keras_image
    from tensorflow.keras.applications.vgg16 import preprocess_input
    from tensorflow.python.keras.activations import linear
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


def check_tensorflow_available():
    """Check if TensorFlow is available and raise an error if not."""
    if not TENSORFLOW_AVAILABLE:
        raise ImportError(
            "TensorFlow is required for this functionality but not installed. "
            "Please install signxai2 with TensorFlow support: pip install signxai2[tensorflow]"
        )


def remove_softmax(model):
    """Remove the softmax activation from the last layer of a TensorFlow model."""
    check_tensorflow_available()
    # Remove last layer's softmax
    model.layers[-1].activation = linear
    return model


def calculate_explanation_innvestigate(model, x, method='lrp.epsilon', neuron_selection=None, batchmode=False,
                                       **kwargs):
    """
    Calculate an explanation using the innvestigate backend

    Args:
        model: TensorFlow/Keras model
        x: Input to explain
        method: Name of the method to use
        neuron_selection: Index of the neuron to explain (default: None)
        batchmode: Whether to process a batch of inputs
        **kwargs: Additional arguments for the method

    Returns:
        Explanation (relevance map)
    """
    check_tensorflow_available()
    
    # Import here to avoid circular imports
    from signxai.tf_signxai.methods_impl.innvestigate.analyzer import create_analyzer

    # Create the analyzer
    analyzer = create_analyzer(method, model, **kwargs)

    # Ensure neuron_selection is a valid format that innvestigate accepts
    # Valid values: None, 'all', 'max_activation', <int>, <list>, <one-dimensional array>
    if neuron_selection is not None:
        # Convert to native Python int to ensure compatibility
        try:
            if hasattr(neuron_selection, 'numpy'):
                # It's a TensorFlow tensor
                neuron_selection = int(neuron_selection.numpy())
                print(f"  DEBUG: Converted tensor to integer: {neuron_selection}")
            elif isinstance(neuron_selection, np.ndarray):
                # It's a numpy array
                neuron_selection = int(neuron_selection.item())
                print(f"  DEBUG: Converted numpy array to integer: {neuron_selection}")
            elif isinstance(neuron_selection, (int, np.integer)):
                # Ensure it's a native Python int, not numpy int
                neuron_selection = int(neuron_selection)
                print(f"  DEBUG: Using neuron selection as native Python int: {neuron_selection}")
            else:
                # Try a simple cast
                neuron_selection = int(neuron_selection)
                print(f"  DEBUG: Forced cast to integer: {neuron_selection}")
        except:
            # If conversion fails, use 'max_activation'
            print(f"  DEBUG: Could not convert neuron_selection '{neuron_selection}', using 'max_activation'")
            neuron_selection = 'max_activation'
    else:
        # Default behavior for None
        neuron_selection = 'max_activation'
        print(f"  DEBUG: Using default neuron selection: 'max_activation'")
        

    # Make sure x is a numpy array
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    # Validate input shape for iNNvestigate
    if x.ndim < 2 or x.ndim > 4:
        raise ValueError(f"Invalid input dimensions for iNNvestigate: {x.ndim}D. Expected 2D-4D.")
    
    # For 4D inputs, ensure they have the expected structure (batch, height, width, channels)
    if x.ndim == 4 and x.shape[0] != 1:
        print(f"  WARNING: Unexpected batch size {x.shape[0]}, taking first sample only")
        x = x[0:1]  # Take only first sample
    
    print(f"  DEBUG: Input to analyzer: shape={x.shape}, ndim={x.ndim}, dtype={x.dtype}")

    # Use similar format to original implementation
    if not batchmode:
        try:
            # Always use input as-is for iNNvestigate - it should handle batch dimensions correctly
            # The input x should already have proper batch dimension (1, H, W, C) from comparison script
            
            ex = analyzer.analyze(X=x, neuron_selection=neuron_selection, **kwargs)

            # Handle the returned explanation properly
            if isinstance(ex, dict):
                # Some analyzers return a dict - get the first value
                expl = ex[list(ex.keys())[0]]
                # If it has batch dimension and we only have one sample, take the first
                if isinstance(expl, np.ndarray) and expl.shape[0] == 1:
                    expl = expl[0]
                elif isinstance(expl, list) and len(expl) == 1:
                    expl = expl[0]
            else:
                # Direct array return - handle batch dimension properly
                if isinstance(ex, np.ndarray) and ex.shape[0] == 1:
                    expl = ex[0]  # Remove batch dimension for single sample
                elif isinstance(ex, list) and len(ex) == 1:
                    expl = ex[0]
                else:
                    expl = ex

            return np.asarray(expl)

        except Exception as e:
            print(f"  DEBUG: First analysis failed: {e}")
            # The first attempt failed, likely due to dimension issues
            # Don't retry with the same problematic approach - just fail fast
            error_message = f"iNNvestigate analysis failed with input shape {x.shape}: {e}"
            print(f"  DEBUG: {error_message}")
            raise ValueError(error_message)
    else:
        # Batch mode
        try:
            # For timeseries data in batch mode, shape handling is critical
            # Print debug info about input shape
            print(f"  DEBUG: Batch mode input shape: {x.shape}")
            
            # Handle possibly missing batch dimension
            if x.ndim == 3 and x.shape[0] != 1:
                # This is already in the right format (probably multiple samples)
                x_batch = x
            elif x.ndim == 2:
                # Single timeseries without channels, add batch and channel dims
                x_batch = np.expand_dims(np.expand_dims(x, axis=0), axis=-1)
                print(f"  DEBUG: Adjusted single timeseries shape to: {x_batch.shape}")
            else:
                # Keep as is
                x_batch = x
                
            # Original approach with adjusted input
            ex = analyzer.analyze(X=x_batch, neuron_selection=neuron_selection, **kwargs)

            # Return all examples
            if isinstance(ex, dict):
                expl = ex[list(ex.keys())[0]]
            else:
                expl = ex

            return np.asarray(expl)

        except Exception as e:
            error_message = f"Error in innvestigate batch analysis: {e}. Input shape: {x.shape}"
            print(f"  DEBUG: {error_message}")
            raise ValueError(error_message)


def load_image(img_path, target_size=(224, 224), expand_dims=False, use_original_preprocessing=True):
    """
    Load an image from a file path and preprocess it for VGG16.

    Args:
        img_path: Path to the image file
        target_size: Size to resize the image to (default: (224, 224))
        expand_dims: Whether to add a batch dimension
        use_original_preprocessing: If True, use the original preprocessing from SIGN-XAI

    Returns:
        Tuple of (original image, preprocessed image)
    """
    check_tensorflow_available()
    
    # Load image
    img = Image.open(img_path)
    img = img.resize(target_size)

    # Preprocess image for the network
    x = keras_image.img_to_array(img)

    if use_original_preprocessing:
        # This is the original preprocessing from SIGN-XAI
        # Ensure we use float32 for consistency
        x = x.astype(np.float32)

        # 'RGB'->'BGR' - Create a copy to avoid stride issues
        x = x.copy()
        # Swap the R and B channels manually
        r_channel = x[..., 0].copy()
        b_channel = x[..., 2].copy()
        x[..., 0] = b_channel
        x[..., 2] = r_channel

        # Zero-centering based on ImageNet mean RGB values
        mean = [103.939, 116.779, 123.68]
        x[..., 0] -= mean[0]
        x[..., 1] -= mean[1]
        x[..., 2] -= mean[2]
    else:
        # Use TensorFlow's built-in preprocessing
        if expand_dims:
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            if not expand_dims:
                x = x[0]  # Remove batch dimension if not needed
        else:
            x_expanded = np.expand_dims(x, axis=0)
            x_processed = preprocess_input(x_expanded)
            x = x_processed[0]

    # Add batch dimension if requested (for original preprocessing)
    if expand_dims and use_original_preprocessing:
        x = np.expand_dims(x, axis=0)

    return img, x


def download_image(path):
    """
    Download example image if it doesn't exist

    Args:
        path: Path to save the image
    """
    if not os.path.exists(path):
        # Create directory if it doesn't exist
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Example image URL
        url = "https://raw.githubusercontent.com/nilsgumpfer/SIGN-experiment-resources/main/example.jpg"
        response = requests.get(url)
        with open(path, 'wb') as f:
            f.write(response.content)


def download_model(path):
    """
    Download example model if it doesn't exist

    Args:
        path: Path to save the model
    """
    if not os.path.exists(path):
        # Create directory if it doesn't exist
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Example model URL
        url = "https://raw.githubusercontent.com/nilsgumpfer/SIGN-experiment-resources/main/model.h5"
        response = requests.get(url)
        with open(path, 'wb') as f:
            f.write(response.content)