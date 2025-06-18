# signxai/common/validation.py

import numpy as np
import tensorflow as tf
import torch
from typing import Union, Tuple, List, Optional, Any


def validate_input_shape(tensor: Union[np.ndarray, tf.Tensor, torch.Tensor],
                         expected_dims: int = 4,
                         framework: str = 'numpy') -> bool:
    """Validates the shape of the input tensor.

    Args:
        tensor: Input tensor to validate.
        expected_dims: Expected number of dimensions. Defaults to 4.
        framework: Framework of the input tensor ('numpy', 'tensorflow', or 'pytorch').
            Defaults to 'numpy'.

    Returns:
        True if shape is valid.

    Raises:
        ValueError: If input shape is invalid or framework is unknown.
    """
    if framework == 'numpy':
        shape = tensor.shape
    elif framework == 'tensorflow':
        shape = tensor.shape.as_list()
    elif framework == 'pytorch':
        shape = tuple(tensor.shape)
    else:
        raise ValueError(f"Unknown framework: {framework}")

    if len(shape) != expected_dims:
        raise ValueError(
            f"Expected {expected_dims} dimensions, but got {len(shape)}"
        )

    return True


def validate_input_range(tensor: Union[np.ndarray, tf.Tensor, torch.Tensor],
                         min_val: float = -1.0,
                         max_val: float = 1.0) -> bool:
    """Validates the range of input values.

    Args:
        tensor: Input tensor to validate.
        min_val: Minimum expected value. Defaults to -1.0.
        max_val: Maximum expected value. Defaults to 1.0.

    Returns:
        True if range is valid.

    Raises:
        ValueError: If input range is invalid.
    """
    if isinstance(tensor, tf.Tensor):
        tensor = tensor.numpy()
    elif isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    if tensor.min() < min_val or tensor.max() > max_val:
        raise ValueError(
            f"Input values must be in range [{min_val}, {max_val}], "
            f"but got range [{tensor.min()}, {tensor.max()}]"
        )

    return True


def validate_model_output(output: Union[np.ndarray, tf.Tensor, torch.Tensor],
                          num_classes: Optional[int] = None) -> bool:
    """Validates model output format and dimensions.

    Args:
        output: Model output to validate.
        num_classes: Expected number of classes (if None, skips this check).
            Defaults to None.

    Returns:
        True if output is valid.

    Raises:
        ValueError: If output format is invalid.
    """
    if isinstance(output, tf.Tensor):
        output = output.numpy()
    elif isinstance(output, torch.Tensor):
        output = output.detach().cpu().numpy()

    if len(output.shape) != 2:
        raise ValueError(
            f"Expected 2D output (batch_size, num_classes), but got shape {output.shape}"
        )

    if num_classes is not None and output.shape[1] != num_classes:
        raise ValueError(
            f"Expected {num_classes} output classes, but got {output.shape[1]}"
        )

    return True


def validate_attribution_output(attribution: Union[np.ndarray, tf.Tensor, torch.Tensor],
                                input_shape: Tuple[int, ...]) -> bool:
    """Validates attribution map output shape and values.

    Args:
        attribution: Attribution map to validate.
        input_shape: Expected shape matching input tensor.

    Returns:
        True if attribution is valid.

    Raises:
        ValueError: If attribution format is invalid.
    """
    if isinstance(attribution, tf.Tensor):
        attribution = attribution.numpy()
    elif isinstance(attribution, torch.Tensor):
        attribution = attribution.detach().cpu().numpy()

    if attribution.shape != input_shape:
        raise ValueError(
            f"Attribution shape {attribution.shape} does not match "
            f"input shape {input_shape}"
        )

    if np.isnan(attribution).any():
        raise ValueError("Attribution map contains NaN values")

    if np.isinf(attribution).any():
        raise ValueError("Attribution map contains infinite values")

    return True


def validate_framework_compatibility(model: Any) -> str:
    """Determines and validates the deep learning framework of a model.

    Args:
        model: Model to validate.

    Returns:
        Framework name ('tensorflow' or 'pytorch').

    Raises:
        ValueError: If model framework cannot be determined or is unsupported.
    """
    if isinstance(model, (tf.keras.Model, tf.keras.Sequential)):
        return 'tensorflow'
    elif isinstance(model, torch.nn.Module):
        return 'pytorch'
    else:
        raise ValueError(
            "Model must be either a TensorFlow Keras model or PyTorch Module"
        )


def convert_to_numpy(tensor: Union[np.ndarray, tf.Tensor, torch.Tensor]) -> np.ndarray:
    """Converts a tensor from any supported framework to a NumPy array.

    Args:
        tensor: Input tensor from any supported framework.

    Returns:
        NumPy array containing the tensor data.

    Raises:
        ValueError: If the input tensor type is not supported.
    """
    if isinstance(tensor, tf.Tensor):
        return tensor.numpy()
    elif isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise ValueError(
            f"Unsupported tensor type: {type(tensor)}. "
            "Must be numpy.ndarray, tf.Tensor, or torch.Tensor"
        )