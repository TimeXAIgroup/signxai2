import matplotlib.pyplot as plt
import numpy as np
import torch

from torchvision.models import vgg16, VGG16_Weights
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from zennit.attribution import Gradient

from signxai2.misc import get_example_image
from signxai2.composites import EpsilonStdX, EpsilonStdXSIGN


def run(num):
    # Define the preprocessing pipeline
    transform = Compose([
        Resize(224),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Load and preprocess image
    image = get_example_image(num)
    data = transform(image)[None]  # Add batch dimension

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

    # Visualize the original image and relevance map
    fig, axs = plt.subplots(1, 3, figsize=(20, 6.5))
    axs[0].imshow(image)
    axs[0].set_title('Image')

    for i, (method, composite) in enumerate(zip(['LRP-Epsilon', 'LRP-Epsilon-SIGN'], [EpsilonStdX(stdfactor=0.3), EpsilonStdXSIGN(mu=0, stdfactor=0.3, signstdfactor=0.3)])):
        # Compute attribution
        with Gradient(model=model, composite=composite) as attributor:
            _, attribution = attributor(data, target)

        # Prepare relevance map
        attribution = np.nan_to_num(attribution)
        relevance = attribution.sum(1)
        R = relevance[0] / np.abs(relevance).max()

        # Plot relevance map
        axs[1+i].matshow(R, cmap='seismic', clim=(-1, 1))
        axs[1+i].set_title(method)

    # Switch off axes and labels
    for ax in axs:
        ax.axis('off')

    # Plot to file
    plt.tight_layout()
    plt.savefig('heatmaps_example_{}.png'.format(num))
    plt.close()


if __name__ == '__main__':
    run(1)
    run(2)