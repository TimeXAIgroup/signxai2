#!/usr/bin/env python
"""
SignXAI TensorFlow Quickstart Example
======================================
Simple example showing how to use SignXAI with TensorFlow/Keras models.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

# Import the unified SignXAI API
from signxai.api import explain

# Step 1: Load a pre-trained model
print("Loading VGG16 model...")
model = VGG16(weights='imagenet')

# Step 2: Load and preprocess an image
img_path = 'examples/data/images/example.jpg'  # Update with your image path
img = Image.open(img_path).convert('RGB')
img_resized = img.resize((224, 224))

# Convert to array and preprocess for VGG16
img_array = np.array(img_resized)
img_preprocessed = preprocess_input(img_array.copy())
img_batch = np.expand_dims(img_preprocessed, axis=0)  # Add batch dimension

# Step 3: Get model prediction
predictions = model.predict(img_batch)
predicted_class = np.argmax(predictions[0])
decoded = decode_predictions(predictions, top=3)[0]

print(f"Top predictions:")
for i, (_, label, prob) in enumerate(decoded):
    print(f"  {i+1}. {label}: {prob*100:.2f}%")
print(f"Using class {predicted_class} for explanation")

# Step 4: Calculate explanation
# Choose your method - here are some examples:
method = "gradient_x_input_x_sign_mu_neg_0_5"  # Current method

# Other method examples (uncomment to use):
# method = "gradient"                           # Basic gradient
# method = "gradient_x_input"                   # Gradient × Input
# method = "gradient_x_sign"                    # Gradient × Sign(Input)
# method = "smoothgrad"                         # SmoothGrad (averaged noisy gradients)
# method = "integrated_gradients"               # Integrated Gradients
# method = "guided_backprop"                    # Guided Backpropagation
# method = "deconvnet"                          # Deconvolution
# method = "lrp_epsilon_0_25"                   # LRP with epsilon=0.25
# method = "lrp_epsilon_50_x_sign"              # LRP (ε=50) × Sign
# method = "lrpsign_epsilon_0_25_std_x"         # LRP-Sign (ε=0.25) with std normalization
# method = "lrp_alpha_2_beta_1"                 # LRP with α=2, β=1
# method = "grad_cam"                           # Grad-CAM (requires conv layers)

# For single image without batch dimension
img_for_xai = img_preprocessed

print(f"\nCalculating explanation using: {method}")
explanation = explain(
    model,
    img_for_xai,
    method_name=method,
    target_class=predicted_class
)

# Step 5: Process and visualize
# Sum over channels to create 2D heatmap
if explanation.ndim == 3:
    heatmap = explanation.sum(axis=-1)  # Sum over channel dimension
else:
    heatmap = explanation

# Normalize for visualization
abs_max = np.max(np.abs(heatmap))
if abs_max > 0:
    normalized = heatmap / abs_max
else:
    normalized = heatmap

# Convert the original image for display
img_display = img_array / 255.0

# Create visualization
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_display)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(normalized, cmap='seismic', clim=(-1, 1))
plt.title(f'Explanation: {method}')
plt.axis('off')

plt.tight_layout()
plt.show()

print("\n✅ Done! Try different methods by uncommenting them in the code.")