import torch
import torch.nn as nn


class VGG16_PyTorch(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16_PyTorch, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 'same' padding for kernel 3, stride 1
            nn.ReLU(inplace=False),  # Using non-inplace ReLU for XAI compatibility
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(
            (7, 7))  # Matches TF VGG16 output before flatten if it came from Keras applications

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # 512 * 7 * 7 is the output of block5_pool flattened
            nn.ReLU(inplace=False),  # Using non-inplace ReLU for XAI compatibility
            # nn.Dropout(0.5), # Standard VGG16 often has dropout here, check if your TF model implies it
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False),
            # nn.Dropout(0.5), # And here
            nn.Linear(4096, num_classes)  # Outputting raw logits (linear activation)
        )

    def forward(self, x):
        # Assuming input x is already NCHW: (Batch, Channels, Height, Width)
        # If your input 'x' to the model might be NHWC, you'd uncomment the permute here:
        # if x.shape[1] != 3 and x.shape[3] == 3: # Basic check for NHWC
        #    x = x.permute(0, 3, 1, 2)

        x = self.features(x)  # Output is NCHW: (batch, 512, 7, 7)

        # To match Keras flatten behavior (which expects NHWC input to its flatten layer):
        # 1. Permute the NCHW output of features to NHWC
        x = x.permute(0, 2, 3, 1).contiguous()  # Now x is (batch, 7, 7, 512)

        # 2. Flatten this NHWC tensor. This should now match Keras flatten.
        x = torch.flatten(x, 1)  # Flattens to (batch, 7*7*512)

        x = self.classifier(x)
        return x

# Example usage:
# model_pytorch = VGG16_PyTorch(num_classes=1000)
# print(model_pytorch)

# Dummy input (Batch, Channels, Height, Width)
# dummy_input = torch.randn(1, 3, 224, 224)
# output = model_pytorch(dummy_input)
# print("Output shape:", output.shape) # Should be [1, 1000]