import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.models import vgg16, VGG16_Weights
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from zennit.attribution import Gradient
from signxai2.composites import EpsilonStdXSIGN
from signxai2.misc import get_example_image

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
weights=VGG16_Weights.IMAGENET1K_V1
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