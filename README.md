# SIGNed XAI for Image and Time Series Models

<img alt="SIGNXAI-Example" src="https://raw.githubusercontent.com/TimeXAIgroup/signxai2/main/img/sign_mnist.png" width=750>

[![Documentation Status](https://github.com/TimeXAIgroup/signxai2/actions/workflows/docs.yml/badge.svg)](https://github.com/TimeXAIgroup/signxai2/actions/workflows/docs.yml)
[![PyPI Version](https://img.shields.io/pypi/v/signxai2)](https://pypi.org/project/signxai2/)
[![License](https://img.shields.io/pypi/l/signxai2)](https://github.com/TimeXAIgroup/signxai2/blob/master/COPYING.LESSER)

SIGN (Sign-based Improvement of Gradient-based explaNations) is a novel XAI method intended to reduce bias in explanations that are intrinsically induced by several state-of-the-art XAI methods. The SIGN-XAI-2 package enables simple application of this method in your projects and is based on Zennit (https://pypi.org/project/zennit/) and LRP for Transformers (https://pypi.org/project/lxt/). If your are using TensorFlow instead of PyTorch, have a look at https://pypi.org/project/signxai/.

SIGN-based explanations are particularly well suited for generating bias-reduced heatmaps for both **image** and **time series data**, enhancing interpretability by more reliably uncovering relevant features.

If you use this package or parts of it in your own work, please consider citing our [paper](https://doi.org/10.1016/j.inffus.2023.101883):
```bibtex
 @article{Gumpfer2023SIGN,
    title = {SIGNed explanations: Unveiling relevant features by reducing bias},
    author = {Nils Gumpfer and Joshua Prim and Till Keller and Bernhard Seeger and Michael Guckert and Jennifer Hannig},
    journal = {Information Fusion},
    pages = {101883},
    year = {2023},
    issn = {1566-2535},
    doi = {https://doi.org/10.1016/j.inffus.2023.101883}
}
```

## Documentation
The latest documentation is available from [timexaigroup.github.io/signxai2](https://timexaigroup.github.io/signxai2/).

## Install
To install the package directly from PyPI using pip, use:
```shell
$ pip install signxai2
```

## Supported Methods

- Gradient
- Gradient $\times$ Input
- Gradient $\times$ SIGN
- LRP (all variants contained in [Zennit](https://pypi.org/project/zennit/). )
- LRP-$\epsilon$ with SIGN as Input Layer Rule

## Usage
SIGN-XAI-2 is based on Zennit and works with PyTorch. If you want to know more about Zennit and its usage, visit [github.com/chr5tphr/zennit](https://github.com/chr5tphr/zennit). There is also a version of SIGN available for usage with TensorFlow environments [pypi.org/project/signxai/](https://pypi.org/project/signxai/).

The below example code (see [vgg16_simple.py](https://github.com/TimeXAIgroup/signxai2/blob/main/examples/vgg16_simple.py)) demonstrates the basic usage of a SIGN-based composite as an extension to LRP-Epsilon in Zennit. To run this code, have a look at the [requirements](https://github.com/TimeXAIgroup/signxai2/blob/main/examples/requirements.txt) and the [setup script](https://github.com/TimeXAIgroup/signxai2/blob/main/examples/setup.sh).

```python
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.models import vgg16, VGG16_Weights
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from zennit.attribution import Gradient
from signxai2.misc import get_example_image
from signxai2.composites import EpsilonStdXSIGN

# Define preprocessing pipeline
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Load and preprocess image
image = get_example_image(1)
data = transform(image)[None]

# Load pretrained VGG16 model
weights = VGG16_Weights.IMAGENET1K_V1
model = vgg16(weights=weights).eval()

# Get model prediction
output = model(data)
pred = output.argmax(1)[0].item()
target = torch.eye(1000)[[pred]]

# Get the class label
label = weights.meta['categories'][pred]
print('Predicted class: {}'.format(label))

# Compute attribution
composite = EpsilonStdXSIGN(mu=0, stdfactor=0.3, signstdfactor=0.3)
with Gradient(model=model, composite=composite) as attributor:
    _, attribution = attributor(data, target)

# Prepare relevance map
attribution = np.nan_to_num(attribution)[0]
relevance = attribution.sum(0)
R = relevance / np.abs(relevance).max()

# Visualize the image and relevance map
fig, axs = plt.subplots(1, 2, figsize=(10, 6))
axs[0].imshow(Compose([Resize((224, 224))])(image))
axs[0].set_title('Image')
axs[1].matshow(R, cmap='seismic', clim=(-1, 1))
axs[1].set_title('LRP-Epsilon-SIGN')

# Switch off axes and labels
for ax in axs:
    ax.axis('off')

# Plot to screen
plt.tight_layout()
plt.show()
```

For more details and examples, have a look at our
[**documentation**](https://timexaigroup.github.io/signxai2/).

## Examples

The above code can be used to generate heatmaps as shown for the example images below. Comparing the baseline LRP-Epsilon method with its SIGN-augmented counterpart highlights notable differences in pixel-level attributions and image-induced contrast, with SIGN mitigating this explanatory bias and distortion inherent to the standard variant.

<img alt="Example 1" src="https://raw.githubusercontent.com/TimeXAIgroup/signxai2/main/img/heatmaps_example_1.png" width=750>

***Image Source:*** Coastal Roamer, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0), via Wikimedia Commons

<img alt="Example 2" src="https://raw.githubusercontent.com/TimeXAIgroup/signxai2/main/img/heatmaps_example_2.png" width=750>

***Image Source:*** randomwild, CC0, via Wikimedia Commons

## License
SIGN-XAI-2 and Zennit are licensed under the GNU LESSER GENERAL PUBLIC LICENSE VERSION 3 OR LATER -- see the [COPYING](https://github.com/TimeXAIgroup/signxai2/blob/main/COPYING) and [COPYING.LESSER](https://github.com/TimeXAIgroup/signxai2/blob/main/COPYING.LESSER) files for details.
