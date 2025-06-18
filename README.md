# SIGNed explanations: Unveiling relevant features by reducing bias

This repository and python package is an extended version of the published python package of the following journal article:
https://doi.org/10.1016/j.inffus.2023.101883

If you use the code from this repository in your work, please cite:
```bibtex
 @article{Gumpfer2023SIGN,
    title = {SIGNed explanations: Unveiling relevant features by reducing bias},
    author = {Nils Gumpfer and Joshua Prim and Till Keller and Bernhard Seeger and Michael Guckert and Jennifer Hannig},
    journal = {Information Fusion},
    pages = {101883},
    year = {2023},
    issn = {1566-2535},
    doi = {https://doi.org/10.1016/j.inffus.2023.101883},
    url = {https://www.sciencedirect.com/science/article/pii/S1566253523001999}
}
```

<img src="https://ars.els-cdn.com/content/image/1-s2.0-S1566253523001999-ga1_lrg.jpg" title="Graphical Abstract" width="900px"/>

## 📦 Quick Setup

After installation, run the setup script to download documentation, examples, and sample data:

```bash
bash prepare.sh
```

This will download:
- 📚 Full documentation (viewable at `docs/index.html`)
- 📝 Example scripts and notebooks (`examples/`)  
- 📊 Sample ECG data and images (`examples/data/`)

## 🚀 Installation

```bash
pip install signxai2
```

## Examples

To get started with SignXAI2 Methods, please follow the examples.


## Setup

To install the package in your environment, run:

```shell
 pip3 install signxai2
```

### PyTorch Support

The library now includes a PyTorch implementation based on [zennit](https://github.com/chr5tphr/zennit). To use the PyTorch implementation, you'll need to install the additional dependencies:

```shell
pip install -r requirements/torch_requirements.txt
```

## Features

- Support for **TensorFlow** and **PyTorch** models
- Consistent API across frameworks
- Wide range of explanation methods:
  - Gradient-based: Vanilla gradient, Integrated gradients, SmoothGrad
  - Class activation maps: Grad-CAM
  - Guided backpropagation
  - Layer-wise Relevance Propagation (LRP)
  - Sign-based thresholding for binary relevance maps


## Usage

### TensorFlow (VGG16)

The below example illustrates the usage of the ```signxai``` package with TensorFlow:

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16
from signxai.tf_signxai import calculate_relevancemap
from signxai.utils.utils import (load_image, aggregate_and_normalize_relevancemap_rgb, download_image, 
                                 calculate_explanation_innvestigate)

# Load model
model = VGG16(weights='imagenet')

#  Remove last layer's softmax activation (we need the raw values!)
model.layers[-1].activation = None

# Load example image
path = 'example.jpg'
download_image(path)
img, x = load_image(path)

# Calculate relevancemaps
R1 = calculate_relevancemap('lrpz_epsilon_0_1_std_x', np.array(x), model)
R2 = calculate_relevancemap('lrpsign_epsilon_0_1_std_x', np.array(x), model)

# Equivalent relevance maps as for R1 and R2, but with direct access to innvestigate and parameters
R3 = calculate_explanation_innvestigate(model, x, method='lrp.stdxepsilon', stdfactor=0.1, input_layer_rule='Z')
R4 = calculate_explanation_innvestigate(model, x, method='lrp.stdxepsilon', stdfactor=0.1, input_layer_rule='SIGN')

# Visualize heatmaps
fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(18, 12))
axs[0][0].imshow(img)
axs[1][0].imshow(img)
axs[0][1].matshow(aggregate_and_normalize_relevancemap_rgb(R1), cmap='seismic', clim=(-1, 1))
axs[0][2].matshow(aggregate_and_normalize_relevancemap_rgb(R2), cmap='seismic', clim=(-1, 1))
axs[1][1].matshow(aggregate_and_normalize_relevancemap_rgb(R3), cmap='seismic', clim=(-1, 1))
axs[1][2].matshow(aggregate_and_normalize_relevancemap_rgb(R4), cmap='seismic', clim=(-1, 1))

plt.show()
```
(Image credit for example used in this code: Greg Gjerdingen from Willmar, USA)

### PyTorch (VGG16)

The PyTorch implementation provides a similar API with the same functionality:

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from signxai.torch_signxai import calculate_relevancemap
from signxai.common.visualization import normalize_relevance_map, relevance_to_heatmap, overlay_heatmap

# Load pre-trained VGG16 model
model = models.vgg16(pretrained=True)
model.eval()

# Load and preprocess image
img_path = "example.jpg"
img = Image.open(img_path)

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(img)

# Keep a copy of the original image for visualization
img_np = np.array(img.resize((224, 224))) / 255.0

# Generate explanations with different methods
methods = ["gradients", "integrated_gradients", "grad_cam", "guided_backprop"]
fig, axes = plt.subplots(1, len(methods)+1, figsize=(15, 5))

# Display original image
axes[0].imshow(img_np)
axes[0].set_title("Original")
axes[0].axis('off')

for i, method in enumerate(methods):
    # Calculate relevance map
    relevance_map = calculate_relevancemap(model, input_tensor, method=method)
    
    # Convert to absolute values and sum across channels for visualization
    if relevance_map.ndim == 3:  # (C, H, W)
        relevance_map = np.abs(relevance_map).sum(axis=0)
    
    # Normalize relevance map
    relevance_map = normalize_relevance_map(relevance_map)
    
    # Convert to heatmap
    heatmap = relevance_to_heatmap(relevance_map)
    
    # Overlay on original image
    overlaid = overlay_heatmap(img_np, heatmap)
    
    # Display
    axes[i+1].imshow(overlaid)
    axes[i+1].set_title(method)
    axes[i+1].axis('off')

plt.tight_layout()
plt.show()
```

### MNIST

The below example illustrates the usage of the ```signxai``` package in combination with a dense model trained on MNIST:

```python
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.models import load_model

from signxai.methods.wrappers import calculate_relevancemap
from signxai.utils.utils import normalize_heatmap, download_model

# Load train and test data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Scale images to the [-1, 0] range
x_train = x_train.astype("float32") / -255.0
x_test = x_test.astype("float32") / -255.0
x_train = -(np.ones_like(x_train) + x_train)
x_test = -(np.ones_like(x_test) + x_test)

# Load model
path = 'model.h5'
download_model(path)
model = load_model(path)

# Remove softmax
model.layers[-1].activation = None

# Calculate relevancemaps
x = x_test[231]
R1 = calculate_relevancemap('gradient_x_input', np.array(x), model, neuron_selection=3)
R2 = calculate_relevancemap('gradient_x_sign_mu_neg_0_5', np.array(x), model, neuron_selection=3)
R3 = calculate_relevancemap('gradient_x_input', np.array(x), model, neuron_selection=8)
R4 = calculate_relevancemap('gradient_x_sign_mu_neg_0_5', np.array(x), model, neuron_selection=8)

# Visualize heatmaps
fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(18, 12))
axs[0][0].imshow(x, cmap='seismic', clim=(-1, 1))
axs[1][0].imshow(x, cmap='seismic', clim=(-1, 1))
axs[0][1].matshow(normalize_heatmap(R1), cmap='seismic', clim=(-1, 1))
axs[0][2].matshow(normalize_heatmap(R2), cmap='seismic', clim=(-1, 1))
axs[1][1].matshow(normalize_heatmap(R3), cmap='seismic', clim=(-1, 1))
axs[1][2].matshow(normalize_heatmap(R4), cmap='seismic', clim=(-1, 1))

plt.show()
```

## Methods

| Method | Base| Parameters |
|--------|-----------------------------------------|--------------------------------|
| gradient | Gradient | |
| input_t_gradient | Gradient x Input | |
| gradient_x_input | Gradient x Input | |
| gradient_x_sign | Gradient x SIGN  | mu = 0 |
| gradient_x_sign_mu | Gradient x SIGN  | requires *mu* parameter |
| gradient_x_sign_mu_0 | Gradient x SIGN  | mu = 0 |
| gradient_x_sign_mu_0_5 | Gradient x SIGN  | mu = 0.5 |
| gradient_x_sign_mu_neg_0_5 | Gradient x SIGN  | mu = -0.5 |
| guided_backprop | Guided Backpropagation | |
| guided_backprop_x_sign | Guided Backpropagation x SIGN  | mu = 0 |
| guided_backprop_x_sign_mu | Guided Backpropagation x SIGN  | requires *mu* parameter |
| guided_backprop_x_sign_mu_0 | Guided Backpropagation x SIGN  | mu = 0 |
| guided_backprop_x_sign_mu_0_5 | Guided Backpropagation x SIGN  | mu = 0.5 |
| guided_backprop_x_sign_mu_neg_0_5 | Guided Backpropagation x SIGN  | mu = -0.5 |
| integrated_gradients | Integrated Gradients | |
| smoothgrad | SmoothGrad | |
| smoothgrad_x_sign | SmoothGrad x SIGN  | mu = 0 |
| smoothgrad_x_sign_mu | SmoothGrad x SIGN  | requires *mu* parameter |
| smoothgrad_x_sign_mu_0 | SmoothGrad x SIGN  | mu = 0 |
| smoothgrad_x_sign_mu_0_5 | SmoothGrad x SIGN  | mu = 0.5  |
| smoothgrad_x_sign_mu_neg_0_5 | SmoothGrad x SIGN  | mu = -0.5  |
| vargrad | VarGrad  | |
| deconvnet | DeconvNet  | |
| deconvnet_x_sign | DeconvNet x SIGN | mu = 0 |
| deconvnet_x_sign_mu | DeconvNet x SIGN | requires *mu* parameter |
| deconvnet_x_sign_mu_0 | DeconvNet x SIGN | mu = 0 |
| deconvnet_x_sign_mu_0_5 | DeconvNet x SIGN | mu = 0.5 |
| deconvnet_x_sign_mu_neg_0_5 | DeconvNet x SIGN | mu = -0.5 |
| grad_cam | Grad-CAM| requires *last_conv* parameter |
| grad_cam_timeseries | Grad-CAM| (for time series data), requires *last_conv* parameter |
| grad_cam_VGG16ILSVRC | | *last_conv* based on VGG16 |
| guided_grad_cam_VGG16ILSVRC | | *last_conv* based on VGG16 |
| lrp_z | LRP-z  | |
| lrpsign_z | LRP-z / LRP-SIGN (Inputlayer-Rule) | |
| zblrp_z_VGG16ILSVRC | LRP-z / LRP-ZB (Inputlayer-Rule) | bounds based on ImageNet |
| w2lrp_z | LRP-z / LRP-w² (Inputlayer-Rule) | |
| flatlrp_z | LRP-z / LRP-flat (Inputlayer-Rule) | |
| lrp_epsilon_0_001 | LRP-epsilon | epsilon = 0.001 |
| lrpsign_epsilon_0_001 | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 0.001 |
| zblrp_epsilon_0_001_VGG16ILSVRC | LRP-epsilon / LRP-ZB (Inputlayer-Rule) | bounds based on ImageNet, epsilon = 0.001 |
| lrpz_epsilon_0_001 |LRP-epsilon / LRP-z (Inputlayer-Rule)  | epsilon = 0.001 |
| lrp_epsilon_0_01 | LRP-epsilon | epsilon = 0.01 |
| lrpsign_epsilon_0_01 | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 0.01 |
| zblrp_epsilon_0_01_VGG16ILSVRC | LRP-epsilon / LRP-ZB (Inputlayer-Rule) | bounds based on ImageNet, epsilon = 0.01 |
| lrpz_epsilon_0_01 | LRP-epsilon / LRP-z (Inputlayer-Rule)  | epsilon = 0.01 |
| w2lrp_epsilon_0_01 | LRP-epsilon / LRP-w² (Inputlayer-Rule)  | epsilon = 0.01 |
| flatlrp_epsilon_0_01 | LRP-epsilon / LRP-flat (Inputlayer-Rule)  | epsilon = 0.01 |
| lrp_epsilon_0_1 | LRP-epsilon | epsilon = 0.1 |
| lrpsign_epsilon_0_1 | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 0.1 |
| zblrp_epsilon_0_1_VGG16ILSVRC | LRP-epsilon / LRP-ZB (Inputlayer-Rule) | bounds based on ImageNet, epsilon = 0.1 |
| lrpz_epsilon_0_1 | LRP-epsilon / LRP-z (Inputlayer-Rule)  | epsilon = 0.1 |
| w2lrp_epsilon_0_1 | LRP-epsilon / LRP-w² (Inputlayer-Rule)  | epsilon = 0.1 |
| flatlrp_epsilon_0_1 | LRP-epsilon / LRP-flat (Inputlayer-Rule)  | epsilon = 0.1 |
| lrp_epsilon_0_2 | LRP-epsilon | epsilon = 0.2 |
| lrpsign_epsilon_0_2 | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 0.2 |
| zblrp_epsilon_0_2_VGG16ILSVRC | LRP-epsilon / LRP-ZB (Inputlayer-Rule) | bounds based on ImageNet, epsilon = 0.2 |
| lrpz_epsilon_0_2 | LRP-epsilon / LRP-z (Inputlayer-Rule)  | epsilon = 0.2 |
| lrp_epsilon_0_5 | LRP-epsilon | epsilon = 0.5 |
| lrpsign_epsilon_0_5 | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 0.5 |
| zblrp_epsilon_0_5_VGG16ILSVRC | LRP-epsilon / LRP-ZB (Inputlayer-Rule) | bounds based on ImageNet, epsilon = 0.5 |
| lrpz_epsilon_0_5 | LRP-epsilon / LRP-z (Inputlayer-Rule)  | epsilon = 0.5 |
| lrp_epsilon_1 | LRP-epsilon | epsilon = 1 |
| lrpsign_epsilon_1 | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 1 |
| zblrp_epsilon_1_VGG16ILSVRC | LRP-epsilon / LRP-ZB (Inputlayer-Rule) | bounds based on ImageNet, epsilon = 1 |
| lrpz_epsilon_1 | LRP-epsilon / LRP-z (Inputlayer-Rule)  | epsilon = 1 |
| w2lrp_epsilon_1 | LRP-epsilon / LRP-w² (Inputlayer-Rule)  | epsilon = 1 |
| flatlrp_epsilon_1 | LRP-epsilon / LRP-flat (Inputlayer-Rule)  | epsilon = 1 |
| lrp_epsilon_5 | LRP-epsilon | epsilon = 5 |
| lrpsign_epsilon_5 | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 5 |
| zblrp_epsilon_5_VGG16ILSVRC | LRP-epsilon / LRP-ZB (Inputlayer-Rule) | bounds based on ImageNet, epsilon = 5 |
| lrpz_epsilon_5 | LRP-epsilon / LRP-z (Inputlayer-Rule)  | epsilon = 5 |
| lrp_epsilon_10 | LRP-epsilon | epsilon = 10 |
| lrpsign_epsilon_10 | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 10 |
| zblrp_epsilon_10_VGG106ILSVRC | LRP-epsilon / LRP-ZB (Inputlayer-Rule) | bounds based on ImageNet, epsilon = 10 |
| lrpz_epsilon_10 | LRP-epsilon / LRP-z (Inputlayer-Rule)  | epsilon = 10 |
| w2lrp_epsilon_10 | LRP-epsilon / LRP-w² (Inputlayer-Rule)  | epsilon = 10 |
| flatlrp_epsilon_10 | LRP-epsilon / LRP-flat (Inputlayer-Rule)  | epsilon = 10 |
| lrp_epsilon_20 | LRP-epsilon | epsilon = 20 |
| lrpsign_epsilon_20 | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 20 |
| zblrp_epsilon_20_VGG206ILSVRC | LRP-epsilon / LRP-ZB (Inputlayer-Rule) | bounds based on ImageNet, epsilon = 20 |
| lrpz_epsilon_20 | LRP-epsilon / LRP-z (Inputlayer-Rule)  | epsilon = 20 |
| w2lrp_epsilon_20 | LRP-epsilon / LRP-w² (Inputlayer-Rule)  | epsilon = 20 |
| flatlrp_epsilon_20 | LRP-epsilon / LRP-flat (Inputlayer-Rule)  | epsilon = 20 |
| lrp_epsilon_50 | LRP-epsilon | epsilon = 50 |
| lrpsign_epsilon_50 | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 50 |
| lrpz_epsilon_50 | LRP-epsilon / LRP-z (Inputlayer-Rule)  | epsilon = 50 |
| lrp_epsilon_75 | LRP-epsilon | epsilon = 75 |
| lrpsign_epsilon_75 | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 75 |
| lrpz_epsilon_75 | LRP-epsilon / LRP-z (Inputlayer-Rule)  | epsilon = 75 |
| lrp_epsilon_100 | LRP-epsilon | epsilon = 100 |
| lrpsign_epsilon_100 | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 100, mu = 0 |
| lrpsign_epsilon_100_mu_0 | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 100, mu = 0 |
| lrpsign_epsilon_100_mu_0_5 | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 100, mu = 0.5 |
| lrpsign_epsilon_100_mu_neg_0_5 | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 100, mu = -0.5 |
| lrpz_epsilon_100 | LRP-epsilon / LRP-z (Inputlayer-Rule) | epsilon = 100 |
| zblrp_epsilon_100_VGG16ILSVRC | LRP-epsilon / LRP-ZB (Inputlayer-Rule) | bounds based on ImageNet, epsilon = 100 |
| w2lrp_epsilon_100 | LRP-epsilon / LRP-w² (Inputlayer-Rule) | epsilon = 100 |
| flatlrp_epsilon_100 | LRP-epsilon / LRP-flat (Inputlayer-Rule) | epsilon = 100 |
| lrp_epsilon_0_1_std_x | LRP-epsilon | epsilon = 0.1 * std(x) |
| lrpsign_epsilon_0_1_std_x | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 0.1 * std(x) |
| lrpz_epsilon_0_1_std_x | LRP-epsilon / LRP-z (Inputlayer-Rule) | epsilon = 0.1 * std(x) |
| zblrp_epsilon_0_1_std_x_VGG16ILSVRC | LRP-epsilon / LRP-ZB (Inputlayer-Rule) | bounds based on ImageNet, epsilon = 0.1 * std(x) |
| w2lrp_epsilon_0_1_std_x | LRP-epsilon / LRP-w² (Inputlayer-Rule) | epsilon = 0.1 * std(x) |
| flatlrp_epsilon_0_1_std_x | LRP-epsilon / LRP-flat (Inputlayer-Rule) | epsilon = 0.1 * std(x) |
| lrp_epsilon_0_25_std_x | LRP-epsilon | epsilon = 0.25 * std(x) |
| lrpsign_epsilon_0_25_std_x | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 0.25 * std(x), mu = 0 |
| lrpz_epsilon_0_25_std_x | LRP-epsilon / LRP-z (Inputlayer-Rule) | epsilon = 0.25 * std(x) |
| zblrp_epsilon_0_25_std_x_VGG256ILSVRC | LRP-epsilon / LRP-ZB (Inputlayer-Rule) | bounds based on ImageNet, epsilon = 0.25 * std(x) |
| w2lrp_epsilon_0_25_std_x | LRP-epsilon / LRP-w² (Inputlayer-Rule) | epsilon = 0.25 * std(x) |
| flatlrp_epsilon_0_25_std_x | LRP-epsilon / LRP-flat (Inputlayer-Rule) | epsilon = 0.25 * std(x) |
| lrpsign_epsilon_0_25_std_x_mu_0 | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 0.25 * std(x), mu = 0 |
| lrpsign_epsilon_0_25_std_x_mu_0_5 | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 0.25 * std(x), mu = 0.5 |
| lrpsign_epsilon_0_25_std_x_mu_neg_0_5 | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 0.25 * std(x), mu = -0.5 |
| lrp_epsilon_0_5_std_x | LRP-epsilon | epsilon = 0.5 * std(x) |
| lrpsign_epsilon_0_5_std_x | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 0.5 * std(x) |
| lrpz_epsilon_0_5_std_x | LRP-epsilon / LRP-z (Inputlayer-Rule) | epsilon = 0.5 * std(x) |
| zblrp_epsilon_0_5_std_x_VGG56ILSVRC | LRP-epsilon / LRP-ZB (Inputlayer-Rule) | bounds based on ImageNet, epsilon = 0.5 * std(x) |
| w2lrp_epsilon_0_5_std_x | LRP-epsilon / LRP-w² (Inputlayer-Rule) | epsilon = 0.5 * std(x) |
| flatlrp_epsilon_0_5_std_x | LRP-epsilon / LRP-flat (Inputlayer-Rule) | epsilon = 0.5 * std(x) |
| lrp_epsilon_1_std_x | LRP-epsilon | epsilon = 1 * std(x) |
| lrpsign_epsilon_1_std_x | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 1 * std(x), mu = 0 |
| lrpz_epsilon_1_std_x | LRP-epsilon / LRP-z (Inputlayer-Rule) | epsilon = 1 * std(x) |
| lrp_epsilon_2_std_x | LRP-epsilon | epsilon = 2 * std(x) |
| lrpsign_epsilon_2_std_x | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 2 * std(x), mu = 0 |
| lrpz_epsilon_2_std_x | LRP-epsilon / LRP-z (Inputlayer-Rule) | epsilon = 2 * std(x) |
| lrp_epsilon_3_std_x | LRP-epsilon | epsilon = 3 * std(x) |
| lrpsign_epsilon_3_std_x | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 3 * std(x), mu = 0 |
| lrpz_epsilon_3_std_x | LRP-epsilon / LRP-z (Inputlayer-Rule) | epsilon = 3 * std(x) |
| lrp_alpha_1_beta_0 | LRP-alpha-beta | alpha = 1, beta = 0 |
| lrpsign_alpha_1_beta_0 | LRP-alpha-beta / LRP-SIGN (Inputlayer-Rule) | alpha = 1, beta = 0, mu = 0 |
| lrpz_alpha_1_beta_0 | LRP-alpha-beta / LRP-z (Inputlayer-Rule) | alpha = 1, beta = 0 |
| zblrp_alpha_1_beta_0_VGG16ILSVRC |  | bounds based on ImageNet, alpha = 1, beta = 0 |
| w2lrp_alpha_1_beta_0 | LRP-alpha-beta / LRP-ZB (Inputlayer-Rule) | alpha = 1, beta = 0 |
| flatlrp_alpha_1_beta_0 | LRP-alpha-beta / LRP-flat (Inputlayer-Rule) | alpha = 1, beta = 0 |
| lrp_sequential_composite_a | LRP Comosite Variant A |  |
| lrpsign_sequential_composite_a | LRP Comosite Variant A / LRP-SIGN (Inputlayer-Rule) |  mu = 0 |
| lrpz_sequential_composite_a | LRP Comosite Variant A / LRP-z (Inputlayer-Rule) |  |
| zblrp_sequential_composite_a_VGG16ILSVRC |  | bounds based on ImageNet  |
| w2lrp_sequential_composite_a | LRP Comosite Variant A / LRP-ZB (Inputlayer-Rule) |  |
| flatlrp_sequential_composite_a | LRP Comosite Variant A / LRP-flat (Inputlayer-Rule) |  |
| lrp_sequential_composite_b | LRP Comosite Variant B |  |
| lrpsign_sequential_composite_b | LRP Comosite Variant B / LRP-SIGN (Inputlayer-Rule) |  mu = 0 |
| lrpz_sequential_composite_b | LRP Comosite Variant B / LRP-z (Inputlayer-Rule) |  |
| zblrp_sequential_composite_b_VGG16ILSVRC |  | bounds based on ImageNet  |
| w2lrp_sequential_composite_b | LRP Comosite Variant B / LRP-ZB (Inputlayer-Rule) |  |
| flatlrp_sequential_composite_b | LRP Comosite Variant B / LRP-flat (Inputlayer-Rule) |  |
