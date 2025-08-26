#!/usr/bin/env python
"""
SignXAI PyTorch ECG Quickstart Example
=======================================
Simple example showing how to use SignXAI with PyTorch models for ECG timeseries data.
Demonstrates dynamic method parsing similar to the images quickstart.
"""

import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Suppress deprecation warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Import the unified SignXAI API
from signxai.api import explain
from signxai.torch_signxai.torch_utils import remove_softmax as torch_remove_softmax

# Add project root to path for utility imports
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_script_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import ECG utilities
from utils.ecg_data import load_and_preprocess_ecg
from utils.ecg_visualization import plot_ecg
from utils.ecg_explainability import normalize_ecg_relevancemap

# Add ECG model directory to path
ecg_model_dir = os.path.join(project_root, 'examples', 'data', 'models', 'pytorch', 'ECG')
if ecg_model_dir not in sys.path:
    sys.path.insert(0, ecg_model_dir)

from ecg_model import ECG_PyTorch
from pathology_ecg_model import Pathology_ECG_PyTorch

# Step 1: Load a pre-trained ECG model
print("Loading ECG model...")
# Option 1: Use default ECG model (1 channel input, for demonstration)
model = ECG_PyTorch(input_channels=1, num_classes=3)

# Option 2: Use pathology model (uncomment to use 12-channel model)
# pathology = 'AVB'  # Options: 'AVB', 'ISCH', 'LBBB', 'RBBB'
# model = Pathology_ECG_PyTorch(input_channels=12, num_classes=2)

# Load pre-trained weights if available
weights_path = os.path.join(project_root, 'examples', 'data', 'models', 
                           'pytorch', 'ECG', 'ecg_model_weights.pth')
if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    print("  Loaded pre-trained weights")
else:
    print("  Using random weights (weights file not found)")

model.eval()

# Remove softmax layer for XAI methods
model_no_softmax = torch_remove_softmax(model)

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

# Convert to PyTorch tensor: (timesteps, channels) -> (batch, channels, timesteps)
input_tensor = torch.from_numpy(ecg_single_lead).float().permute(1, 0).unsqueeze(0)
print(f"  Input tensor shape: {input_tensor.shape}")  # (1, 1, 3000)

# Step 3: Get model prediction
print("\nGetting model prediction...")
with torch.no_grad():
    output = model_no_softmax(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

predicted_idx = torch.argmax(output, dim=1)
print(f"Predicted class: {predicted_idx.item()}")
print(f"Class probabilities: {probabilities.numpy()}")

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
print(f"  Input shape: {input_tensor.shape}")
print(f"  Target class: {predicted_idx.item()}")

explanation = explain(
    model_no_softmax,
    input_tensor,
    method_name=method,
    target_class=predicted_idx.item()
)

print(f"  Explanation calculated successfully!")
print(f"  Output shape: {explanation.shape if hasattr(explanation, 'shape') else 'unknown'}")

# Step 5: Process and visualize
# Convert to numpy for visualization
if hasattr(explanation, 'detach'):
    explanation_np = explanation.detach().cpu().numpy()
else:
    explanation_np = explanation

# Process relevance map for visualization
if isinstance(explanation_np, torch.Tensor):
    explanation_np = explanation_np.detach().cpu().numpy()

# Handle shape processing for relevance map
if explanation_np.ndim == 3:
    # Shape: (batch, channels, timesteps) -> (timesteps, channels)
    relevance_map = explanation_np[0].transpose()  # (1, 1, 3000) -> (3000, 1)
elif explanation_np.ndim == 2:
    # Shape: (channels, timesteps) -> (timesteps, channels)
    relevance_map = explanation_np.transpose()
else:
    relevance_map = explanation_np.reshape(-1, 1)  # Ensure 2D

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
title = f"PyTorch XAI: {method} on {record_id}"

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