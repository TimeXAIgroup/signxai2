import torch
import torch.nn as nn
import torch.nn.functional as F


class ECG_PyTorch(nn.Module):
    def __init__(self, input_channels=1, num_classes=3):
        super(ECG_PyTorch, self).__init__()
        # ... (all layer definitions like self.conv1, self.pool1, ..., self.flatten, self.fc1 etc. remain the same)
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4, padding=0)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4, padding=0)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64 * 46, out_features=64)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        # Input x expected as (batch, channels, steps), e.g., (None, 1, 3000)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)  # Output of pool3 is (batch_size, 64, 46) i.e. (N, C, S_out)

        # *** ADD PERMUTATION HERE ***
        # Permute from (N, C, S_out) to (N, S_out, C) to match TF's data order before Flatten
        x = x.permute(0, 2, 1)  # x is now (batch_size, 46, 64)

        x = self.flatten(x)  # PyTorch's nn.Flatten now operates on (N, 46, 64)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Dropout is fine, model is in eval() mode during comparison
        x = self.fc2(x)

        return x


if __name__ == '__main__':
    # This block is for testing the model definition independently.
    # Expected TF input_shape: (None, 3000, 1)
    # PyTorch model expects input as (batch_size, input_channels, sequence_length)
    # So, if TF input is (N, 3000, 1), PT input should be (N, 1, 3000)

    pt_model = ECG_PyTorch(input_channels=1, num_classes=3)
    pt_model.eval()  # Set to evaluation mode for consistent dropout behavior

    # Create a dummy input tensor matching the expected PyTorch input shape
    # (e.g., batch_size=2, input_channels=1, sequence_length=3000)
    dummy_input_pt = torch.randn(2, 1, 3000)

    print(f"PyTorch Model Input Shape: {dummy_input_pt.shape}")
    output_pt = pt_model(dummy_input_pt)
    print(f"PyTorch Model Output Shape: {output_pt.shape}")  # Expected: (2, 3)

    print("\nPyTorch Model Structure:")
    for name, module in pt_model.named_children():
        print(f"{name}: {module}")

    total_params_pt = sum(p.numel() for p in pt_model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters in PyTorch model: {total_params_pt}")
    # From your TF summary: Total params: 201,667. This should match.
    # Calculation:
    # conv1: (1*5+1)*16 = 96
    # conv2: (16*5+1)*32 = 2592
    # conv3: (32*5+1)*64 = 10304
    # fc1: (2944*64)+64 = 188480
    # fc2: (64*3)+3 = 195
    # Total = 96 + 2592 + 10304 + 188480 + 195 = 201667. Correct.