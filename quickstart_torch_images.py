#!/usr/bin/env python
"""
SignXAI PyTorch Quickstart Example
===================================
Simple example showing how to use SignXAI with PyTorch models.
"""

import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms

# Suppress deprecation warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

# Import the unified SignXAI API
from signxai.api import explain

# Step 1: Load a pre-trained model
print("Loading VGG16 model...")
model = models.vgg16(pretrained=True)
model.eval()

# Step 2: Load and preprocess an image
img_path = 'examples/data/images/example.jpg'  # Update with your image path
img = Image.open(img_path).convert('RGB')

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension

# Step 3: Get model prediction
print("Getting model prediction...")
with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

# Get the predicted class and top predictions
predicted_idx = torch.argmax(output, dim=1)
top_probs, top_idxs = torch.topk(probabilities, 3)

print("Top predictions:")
# Load ImageNet class names (simplified version)
class_names = {757: 'recreational_vehicle', 654: 'mobile_home', 511: 'moving_van', 
               609: 'ambulance', 717: 'pickup_truck'}  # Add more as needed

for i, (prob, idx) in enumerate(zip(top_probs, top_idxs)):
    class_name = class_names.get(idx.item(), f'class_{idx.item()}')
    print(f"  {i+1}. {class_name}: {prob.item()*100:.2f}%")

print(f"Using class {predicted_idx.item()} for explanation")

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

print(f"\nCalculating explanation using: {method}")
print(f"  Input shape: {input_tensor.shape}")
print(f"  Target class: {predicted_idx.item()}")

explanation = explain(
    model,
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

# Remove batch dimension if present
if explanation_np.ndim == 4:
    explanation_np = explanation_np[0]

# Sum over channels to create 2D heatmap
if explanation_np.ndim == 3:
    heatmap = explanation_np.sum(axis=0)
else:
    heatmap = explanation_np

# Normalize for visualization
abs_max = np.max(np.abs(heatmap))
if abs_max > 0:
    normalized = heatmap / abs_max
else:
    normalized = heatmap

# Convert the original image for display
img_np = np.array(img.resize((224, 224))) / 255.0

# Create visualization
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_np)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(normalized, cmap='seismic', clim=(-1, 1))
plt.title(f'Explanation: {method}')
plt.axis('off')

plt.tight_layout()
plt.show()

print("\n✅ Done! Try different methods by uncommenting them in the code.")