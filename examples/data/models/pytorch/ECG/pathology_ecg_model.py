import torch
import torch.nn as nn
import torch.nn.functional as F


class Pathology_ECG_PyTorch(nn.Module):  # Renamed from AVB_ECG_PyTorch for generality
    def __init__(self, input_channels=12, num_classes=2):
        super(Pathology_ECG_PyTorch, self).__init__()

        # Conv Block 1
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # Conv Block 2
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.dropout1 = nn.Dropout(p=0.1)  # Corresponds to TF 'dropout'

        # Conv Block 3
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.dropout2 = nn.Dropout(p=0.1)  # Corresponds to TF 'dropout_1'

        # Conv Block 4
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.dropout3 = nn.Dropout(p=0.1)  # Corresponds to TF 'dropout_2'

        # Conv Block 5
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.dropout4 = nn.Dropout(p=0.1)  # Corresponds to TF 'dropout_3'

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Dense Block 1
        self.fc1 = nn.Linear(64, 64)
        self.dropout_fc1 = nn.Dropout(p=0.25)  # Corresponds to TF 'dropout_4'

        # Dense Block 2
        self.fc2 = nn.Linear(64, 64)
        self.dropout_fc2 = nn.Dropout(p=0.25)  # Corresponds to TF 'dropout_5'

        # Dense Block 3
        self.fc3 = nn.Linear(64, 64)
        self.dropout_fc3 = nn.Dropout(p=0.25)  # Corresponds to TF 'dropout_6'

        # Output Layer
        self.fc_out = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (batch, input_channels, sequence_length) e.g. (N, 12, 2000)

        x = F.elu(self.conv1(x))
        x = self.pool1(x)

        x = F.elu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout1(x)

        x = F.elu(self.conv3(x))
        x = self.pool3(x)
        x = self.dropout2(x)

        x = F.elu(self.conv4(x))
        x = self.pool4(x)
        x = self.dropout3(x)

        x = F.elu(self.conv5(x))
        x = self.pool5(x)
        x = self.dropout4(x)

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)

        x = F.elu(self.fc1(x))
        x = self.dropout_fc1(x)

        x = F.elu(self.fc2(x))
        x = self.dropout_fc2(x)

        x = F.elu(self.fc3(x))
        x = self.dropout_fc3(x)

        x = self.fc_out(x)
        x = F.softmax(x, dim=-1)

        return x


if __name__ == '__main__':
    # Test with dummy input
    model = Pathology_ECG_PyTorch(input_channels=12, num_classes=2)
    model.eval()
    dummy_input = torch.randn(1, 12, 2000)
    output = model(dummy_input)
    print(f"{model.__class__.__name__} dummy output shape:", output.shape)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters in {model.__class__.__name__} model: {total_params}")
    # Expected TF Total params: 64,386.