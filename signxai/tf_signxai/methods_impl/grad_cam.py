"""
Title: Grad-CAM class activation visualization
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/04/26
Last modified: 2021/03/07
Description: How to obtain a class activation heatmap for an image classification model.

Adapted from Deep Learning with Python (2017).
"""

import numpy as np
import tensorflow as tf
from scipy.interpolate.interpolate import interp1d
from tensorflow import keras
from tensorflow.python.keras import Model


class GradCAM:
    """Grad-CAM implementation for TensorFlow models.
    
    Grad-CAM uses the gradients of a target concept flowing into the final
    convolutional layer to produce a coarse localization map highlighting
    important regions in the image for prediction.
    """
    
    def __init__(self, model, last_conv_layer_name):
        """Initialize GradCAM.
        
        Args:
            model: TensorFlow model
            last_conv_layer_name: Name of the last convolutional layer
        """
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name
        
        # Create a model that maps the input to the activations of the last conv layer and model output
        self.grad_model = tf.keras.models.Model(
            [model.inputs], 
            [model.get_layer(last_conv_layer_name).output, model.output]
        )
    
    def compute_heatmap(self, x, target_class=None, resize=True):
        """Compute Grad-CAM heatmap.
        
        Args:
            x: Input tensor or array (must include batch dimension)
            target_class: Target class index (None for argmax)
            resize: Whether to resize the heatmap to input size
            
        Returns:
            Grad-CAM heatmap
        """
        # Check if input has batch dimension, if not add it
        if len(x.shape) == 3:
            x = np.expand_dims(x, axis=0)
            
        # Convert to tensor if numpy array
        if isinstance(x, np.ndarray):
            x = tf.convert_to_tensor(x)
            
        # Compute gradient of target class with respect to last conv layer
        with tf.GradientTape() as tape:
            # Forward pass
            last_conv_layer_output, preds = self.grad_model(x)
            
            # Determine target class if not specified
            if target_class is None:
                target_class = tf.argmax(preds[0])
                
            # Select target class output
            class_channel = preds[:, target_class]
        
        # Gradient of target class with respect to last conv layer output
        grads = tape.gradient(class_channel, last_conv_layer_output)
        
        # Vector of importance weights for each feature map
        if len(grads.shape) == 4:  # For images (B, H, W, C)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        else:  # For time series (B, T, C)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
        
        # Extract first sample's conv output
        last_conv_output = last_conv_layer_output[0]
        
        # Weight feature maps by importance
        weighted_output = last_conv_output * pooled_grads[..., tf.newaxis]
        
        # Sum across feature map channels
        heatmap = tf.reduce_sum(weighted_output, axis=-1)
        
        # Apply ReLU and normalize
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
        
        # Convert to numpy
        heatmap = heatmap.numpy()
        
        # Resize if requested
        if resize and len(x.shape) == 4:  # Image input
            import cv2
            # Resize to input spatial dimensions
            heatmap = cv2.resize(heatmap, (x.shape[2], x.shape[1]))
            
        elif resize and len(x.shape) == 3:  # Time series input
            # Interpolate to match input time steps
            f = interp1d(
                x=np.arange(0, len(heatmap)), 
                y=heatmap,
                bounds_error=False,
                fill_value="extrapolate"
            )
            heatmap = f(np.linspace(0, len(heatmap) - 1, num=x.shape[1]))
            
            # Match channel dimension
            if x.shape[2] > 1:
                heatmap = np.expand_dims(heatmap, axis=1)
                heatmap = np.tile(heatmap, (1, x.shape[2]))
                
        return heatmap


def calculate_grad_cam_relevancemap_timeseries(x, model, last_conv_layer_name, neuron_selection=None, resize=True):
    """
    Calculate Grad-CAM relevance map specifically adapted for time series data.
    
    Args:
        x: Input data, expected shape: (batch_size, time_steps, channels)
        model: Model to analyze
        last_conv_layer_name: Name of the last convolutional layer
        neuron_selection: Index of neuron to analyze (None for predicted class)
        resize: Whether to resize heatmap to input size
        
    Returns:
        Relevance map with shape matching the input if resize=True
    """
    # Debug input shape
    print(f"  DEBUG: GradCAM-Timeseries input shape: {x.shape}")
    
    # Ensure input has batch dimension
    if not isinstance(x, np.ndarray):
        x = np.array(x)
        
    if x.ndim == 2:  # Shape (time_steps, channels)
        x = np.expand_dims(x, axis=0)  # Add batch -> (1, time_steps, channels)
        print(f"  DEBUG: Added batch dimension, new shape: {x.shape}")
    
    # Convert numpy array to tensor
    if isinstance(x, np.ndarray):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
    
    # Create a model that maps the input to the activations of the last conv layer and model output
    grad_model = Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute the gradient of the target class with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Need to watch the input to the gradient tape in TF2
        tape.watch(x)
        
        # Forward pass
        last_conv_layer_output, preds = grad_model(x)
        print(f"  DEBUG: Conv layer output shape: {last_conv_layer_output.shape}")
        print(f"  DEBUG: Model prediction shape: {preds.shape}")
        
        # Determine target class if not specified
        if neuron_selection is None:
            neuron_selection = tf.argmax(preds[0])
        
        # Get the specific class output
        class_channel = preds[:, neuron_selection]
        print(f"  DEBUG: Selected neuron {neuron_selection}, activation: {class_channel.numpy()}")

    # Calculate gradient of the target class with respect to feature maps
    grads = tape.gradient(class_channel, last_conv_layer_output)
    print(f"  DEBUG: Gradients shape: {grads.shape}")

    # Calculate importance weights for each feature map
    # Pool across temporal dimension (axis 1) and batches (axis 0)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    print(f"  DEBUG: Pooled gradients shape: {pooled_grads.shape}")

    # Apply the importance weights to each feature map
    last_conv_layer_output = last_conv_layer_output[0]  # Get first sample
    
    # Ensure proper broadcasting by adding a dimension
    pooled_grads_reshaped = tf.reshape(pooled_grads, (1, -1))
    print(f"  DEBUG: Reshaped pooled gradients: {pooled_grads_reshaped.shape}")
    
    # Calculate weighted feature maps (properly vectorized)
    weighted_maps = tf.einsum('tc,c->t', last_conv_layer_output, pooled_grads)
    heatmap = weighted_maps
    
    print(f"  DEBUG: Pre-normalization heatmap shape: {heatmap.shape}")

    # ReLU and normalize
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    else:
        print("  WARNING: Max value is zero, heatmap will be all zeros")

    # Convert to numpy for further processing
    heatmap_np = heatmap.numpy()

    # Resize to match input time steps
    if resize is True:
        try:
            # Get number of time steps and channels from input
            input_time_steps = x.shape[1]
            input_channels = x.shape[2]
            
            # Check if we need to resize
            if len(heatmap_np) != input_time_steps:
                print(f"  DEBUG: Resizing heatmap from {len(heatmap_np)} to {input_time_steps} time steps")
                
                # Create interpolation function for the temporal dimension
                f = interp1d(
                    x=np.arange(len(heatmap_np)), 
                    y=heatmap_np,
                    bounds_error=False,
                    fill_value="extrapolate"
                )
                
                # Interpolate to match input time steps
                heatmap_resized = f(np.linspace(0, len(heatmap_np) - 1, num=input_time_steps))
                
                # Expand to match channel dimension if needed
                if input_channels > 1:
                    heatmap_resized = np.expand_dims(heatmap_resized, axis=1)
                    heatmap_resized = np.tile(heatmap_resized, (1, input_channels))
                    print(f"  DEBUG: Expanded heatmap to match {input_channels} channels, shape: {heatmap_resized.shape}")
                
                return heatmap_resized
            else:
                # Just expand to match channels if needed
                if input_channels > 1 and heatmap_np.ndim == 1:
                    heatmap_np = np.expand_dims(heatmap_np, axis=1)
                    heatmap_np = np.tile(heatmap_np, (1, input_channels))
                    print(f"  DEBUG: Expanded heatmap to match {input_channels} channels, shape: {heatmap_np.shape}")
                
                return heatmap_np
        except Exception as e:
            print(f"  ERROR in resizing: {e}")
            # Fall back to unresized heatmap
            return heatmap_np
    else:
        return heatmap_np


def calculate_grad_cam_relevancemap(x, model, last_conv_layer_name, neuron_selection=None, resize=False, **kwargs):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(x)
        if neuron_selection is None:
            neuron_selection = tf.argmax(preds[0])
        class_channel = preds[:, neuron_selection]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    # For 1D timeseries: grads shape is [batch, time, channels], so reduce over (0, 1)
    # For 2D images: grads shape is [batch, height, width, channels], so reduce over (0, 1, 2)
    if len(grads.shape) == 3:
        # 1D timeseries case
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    else:
        # 2D image case (original)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the relevancemap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    
    # Ensure compatible shapes for matrix multiplication
    if len(last_conv_layer_output.shape) == 2:
        # 1D timeseries: [time, channels] @ [channels] -> [time]
        relevancemap = tf.reduce_sum(last_conv_layer_output * pooled_grads, axis=1)
    else:
        # 2D image case (original)
        relevancemap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        relevancemap = tf.squeeze(relevancemap)

    # Relu (filter positve values)
    relevancemap = tf.maximum(relevancemap, 0)

    # For visualization purpose, we will also normalize the relevancemap between 0 & 1
    relevancemap = relevancemap / tf.math.reduce_max(relevancemap)

    if resize is True:
        # Resize to input spatial dimensions using TensorFlow operations
        import cv2
        h = relevancemap.numpy()
        # Resize to match input spatial dimensions (H, W) -> input shape is (B, H, W, C)
        h_resized = cv2.resize(h, (x.shape[2], x.shape[1]))
        return h_resized
    else:
        return relevancemap.numpy()
