#!/usr/bin/env python
"""
SignXAI TensorFlow ECG Quickstart Example
==========================================
Simple example showing how to use SignXAI with TensorFlow/Keras models for ECG timeseries data.
Demonstrates dynamic method parsing similar to the images quickstart.
"""

import warnings
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Import the unified SignXAI API
from signxai.api import explain
from signxai.utils.utils import remove_softmax as tf_remove_softmax

# Add project root to path for utility imports
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_script_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import ECG utilities
from utils.ecg_data import load_and_preprocess_ecg
from utils.ecg_visualization import plot_ecg
from utils.ecg_explainability import normalize_ecg_relevancemap

# Step 1: Load or create an ECG model
print("Loading ECG model...")

def create_ecg_model(input_channels=1, num_classes=3, sequence_length=3000):
    """Create a simple CNN model for ECG classification."""
    model = keras.Sequential([
        keras.layers.Input(shape=(sequence_length, input_channels)),
        keras.layers.Conv1D(32, kernel_size=7, activation='relu', padding='same'),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Conv1D(128, kernel_size=3, activation='relu', padding='same', name='last_conv'),
        keras.layers.GlobalMaxPooling1D(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Create model
model = create_ecg_model(input_channels=1, num_classes=3)

# Try to load pre-trained weights if available
weights_path = os.path.join(project_root, 'examples', 'data', 'models', 
                           'tensorflow', 'ECG', 'ecg_model.h5')
if os.path.exists(weights_path):
    try:
        # Try loading the entire model first
        model = keras.models.load_model(weights_path)
        print("  Loaded pre-trained model")
    except:
        try:
            # If that fails, try loading just the weights
            model.load_weights(weights_path)
            print("  Loaded pre-trained weights")
        except:
            print("  Could not load weights, using random initialization")
else:
    print("  Using random weights (model file not found)")

# Remove softmax layer for XAI methods
model_no_softmax = tf_remove_softmax(model)

# Step 2: Load and preprocess ECG data
record_id = '03509_hr'  # Example ECG record - change this to use different records
ecg_src_dir = os.path.join(project_root, 'examples', 'data', 'timeseries', '')

print(f"Loading ECG data for record: {record_id}...")
ecg_data = load_and_preprocess_ecg(
    record_id=record_id,
    src_dir=ecg_src_dir,
    ecg_filters=['BWR', 'BLA', 'AC50Hz', 'LP40Hz'],
    subsampling_window_size=3000,  # Model expects 3000 timesteps
    subsample_start=0
)

if ecg_data is None:
    print("Failed to load ECG data. Please check the data path.")
    sys.exit(1)

print(f"  ECG data shape: {ecg_data.shape}")  # (3000, 12) - 3000 timesteps, 12 leads

# Store the full 12-lead ECG for visualization
original_ecg_data = ecg_data.copy()

# Use only the first lead for the model (which expects 1 channel)
ecg_single_lead = ecg_data[:, 0:1]  # Shape: (3000, 1)

# TensorFlow expects shape: (timesteps, channels)
input_data = ecg_single_lead.astype(np.float32)  # Shape: (3000, 1)
print(f"  Input data shape: {input_data.shape}")

# Step 3: Get model prediction
print("\nGetting model prediction...")
# Add batch dimension for prediction
predictions = model_no_softmax.predict(np.expand_dims(input_data, 0), verbose=0)
predicted_idx = np.argmax(predictions[0])
probabilities = tf.nn.softmax(predictions[0]).numpy()

print(f"Predicted class: {predicted_idx}")
print(f"Class probabilities: {probabilities}")

# Step 4: Calculate explanation
# Choose your method - here are some examples with dynamic parameter parsing:
method = "gradient_x_input_x_sign_mu_neg_0_5"  # Current method

# Other method examples (uncomment to use):
# method = "gradient"                                      # Basic gradient
# method = "gradient_x_input"                             # Gradient × Input  
# method = "gradient_x_sign"                              # Gradient × Sign(Input)
# method = "gradient_x_input_x_sign_mu_neg_0_5"          # Complex combination with parameter
# method = "smoothgrad"                                   # SmoothGrad (default params)
# method = "smoothgrad_noise_0_3_samples_50"             # SmoothGrad with custom params
# method = "integrated_gradients"                         # Integrated Gradients (default)
# method = "integrated_gradients_steps_100"               # IG with 100 steps
# method = "guided_backprop"                              # Guided Backpropagation
# method = "deconvnet"                                    # Deconvolution
# method = "lrp_epsilon_0_25"                            # LRP with epsilon=0.25
# method = "lrp_epsilon_50_x_sign"                       # LRP (ε=50) × Sign
# method = "lrpsign_epsilon_0_25_std_x"                  # LRP-Sign (ε=0.25) with std normalization
# method = "lrp_alpha_2_beta_1"                          # LRP with α=2, β=1

print(f"\nCalculating explanation using: {method}")
print(f"  Input shape: {input_data.shape}")
print(f"  Target class: {predicted_idx}")

# For TensorFlow, SignXAI handles batch dimension automatically
explanation = explain(
    model_no_softmax,
    input_data,  # No batch dimension needed
    method_name=method,
    target_class=predicted_idx
)

print(f"  Explanation calculated successfully!")
print(f"  Output shape: {explanation.shape if hasattr(explanation, 'shape') else 'unknown'}")

# Step 5: Process and visualize
# Process relevance map for visualization
if isinstance(explanation, tf.Tensor):
    explanation_np = explanation.numpy()
else:
    explanation_np = explanation

# Handle shape processing for relevance map
# Ensure shape is (timesteps, channels)
if explanation_np.ndim == 1:
    relevance_map = explanation_np.reshape(-1, 1)
elif explanation_np.ndim == 2:
    if explanation_np.shape[0] == 1:
        relevance_map = explanation_np.transpose()
    else:
        relevance_map = explanation_np
else:
    relevance_map = explanation_np.reshape(-1, 1)

# Expand single-channel relevance to 12 channels to match ECG visualization
if relevance_map.shape[1] == 1 and original_ecg_data.shape[1] == 12:
    print(f"  Expanding single-channel relevance to 12 channels for visualization")
    relevance_map = np.tile(relevance_map, (1, 12))

# Normalize relevance values
normalized_relevance = normalize_ecg_relevancemap(relevance_map)

# Format data for 12-lead visualization
# plot_ecg expects (leads, timesteps) format
ecg_for_visual = original_ecg_data.transpose()  # (timesteps, leads) -> (leads, timesteps)
expl_for_visual = normalized_relevance.transpose()  # (timesteps, leads) -> (leads, timesteps)

print(f"  Shapes for visualization - ECG: {ecg_for_visual.shape}, Explanation: {expl_for_visual.shape}")

# Create 12-lead visualization using ECG visualization utilities
title = f"TensorFlow XAI: {method} on {record_id}"

# Use the plot_ecg function for professional 12-lead visualization
plot_ecg(
    ecg=ecg_for_visual,
    explanation=expl_for_visual,
    sampling_rate=500,  # Standard ECG sampling rate
    title=title,
    show_colorbar=True,
    cmap='seismic',  # Red-blue colormap for relevance
    bubble_size=30,  # Size of relevance dots
    line_width=1.0,
    style='fancy',
    save_to=None,  # Set to filename to save
    clim_min=-1,
    clim_max=1,
    colorbar_label='Relevance',
    shape_switch=False  # We already handled the shape switching
)

print("\n✅ Done! Try different methods by uncommenting them in the code.")
print("\nTips:")
print("  - The visualization shows all 12 ECG leads")
print("  - Red indicates positive relevance, blue indicates negative relevance")
print("  - Try different methods to see how they highlight different features")
print("  - You can change the record_id to analyze different ECG recordings")