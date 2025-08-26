"""
Framework-agnostic utility functions for SignXAI2.

This module contains utility functions that work independently of any deep learning framework.
Framework-specific utilities have been moved to:
- signxai/tf_signxai/tf_utils.py for TensorFlow utilities
- signxai/torch_signxai/torch_utils.py for PyTorch utilities
"""

import os
import sys
import numpy as np
import requests
from PIL import Image


def get_examples_data_dir():
    """
    Get the path to the examples/data directory regardless of current working directory

    Returns:
        str: Path to the examples/data directory
    """
    # Try to find examples/data directory
    current_dir = os.getcwd()

    # Check if we're in examples/tutorials/tensorflow
    if os.path.basename(current_dir) == 'tensorflow' and \
            os.path.basename(os.path.dirname(current_dir)) == 'tutorials':
        return os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'data')

    # Check if we're in examples/tutorials
    if os.path.basename(current_dir) == 'tutorials' and \
            os.path.basename(os.path.dirname(current_dir)) == 'examples':
        return os.path.join(os.path.dirname(current_dir), 'data')

    # Check if we're in examples
    if os.path.basename(current_dir) == 'examples':
        return os.path.join(current_dir, 'data')

    # Check if we're in project root
    if os.path.exists(os.path.join(current_dir, 'examples', 'data')):
        return os.path.join(current_dir, 'examples', 'data')

    # Last resort: try to find examples/data relative to script location
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    examples_data_dir = os.path.join(script_dir, 'examples', 'data')
    if os.path.exists(examples_data_dir):
        return examples_data_dir

    # If we couldn't find it, create it in the current directory
    os.makedirs(os.path.join(current_dir, 'data'), exist_ok=True)
    return os.path.join(current_dir, 'data')


def aggregate_and_normalize_relevancemap_rgb(relevancemap):
    """
    Aggregate and normalize a RGB relevance map

    Args:
        relevancemap: RGB relevance map

    Returns:
        Normalized relevance map
    """
    # Aggregate channels
    if relevancemap.ndim == 3:
        relevancemap = np.sum(relevancemap, axis=2)

    return normalize_heatmap(relevancemap)


def normalize_heatmap(heatmap):
    """
    Normalize a heatmap to the range [-1, 1]

    Args:
        heatmap: Heatmap to normalize

    Returns:
        Normalized heatmap
    """
    if heatmap.min() != heatmap.max():
        max_abs = np.max(np.abs(heatmap))
        if max_abs > 0:
            heatmap = heatmap / max_abs
        return np.nan_to_num(heatmap, nan=0)
    else:
        return np.zeros_like(heatmap)


# Deprecated functions with framework-specific imports
# These are kept for backward compatibility but will delegate to framework-specific modules

def remove_softmax(model):
    """
    Remove the softmax activation from the last layer of a model.
    
    This function delegates to the appropriate framework-specific implementation.
    
    Args:
        model: TensorFlow or PyTorch model
        
    Returns:
        Model with softmax removed
    """
    # Try to determine the framework
    model_type = type(model).__module__
    
    if 'tensorflow' in model_type or 'keras' in model_type:
        from ..tf_signxai.tf_utils import remove_softmax as tf_remove_softmax
        return tf_remove_softmax(model)
    elif 'torch' in model_type:
        from ..torch_signxai.torch_utils import remove_softmax as torch_remove_softmax
        return torch_remove_softmax(model)
    else:
        raise ValueError(f"Unable to determine framework for model type: {type(model)}")


def calculate_explanation_innvestigate(model, x, method='lrp.epsilon', neuron_selection=None, batchmode=False, **kwargs):
    """
    Calculate an explanation using the innvestigate backend (TensorFlow only).
    
    This function has been moved to tf_utils.py as it's TensorFlow-specific.
    This wrapper is kept for backward compatibility.
    """
    from ..tf_signxai.tf_utils import calculate_explanation_innvestigate as tf_calc_explanation
    return tf_calc_explanation(model, x, method, neuron_selection, batchmode, **kwargs)


def load_image(img_path, target_size=(224, 224), expand_dims=False, use_original_preprocessing=True):
    """
    Load an image from a file path and preprocess it.
    
    This function has been moved to tf_utils.py as it uses TensorFlow preprocessing.
    This wrapper is kept for backward compatibility.
    """
    from ..tf_signxai.tf_utils import load_image as tf_load_image
    return tf_load_image(img_path, target_size, expand_dims, use_original_preprocessing)


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