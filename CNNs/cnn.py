import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        
        # Calculate the size of the flattened features
        flattened_size = 64 * (input_shape[1] // 4) * (input_shape[2] // 4)
        self.fc = nn.Linear(flattened_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return self.softmax(x)

# Example usage
input_shape = (3, 32, 32)  # (channels, height, width)
num_classes = 10
model = CNNModel(input_shape, num_classes)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Generate dummy input and move it to the same device as the model
input_data = torch.randn(1, *input_shape).to(device)

# Forward pass
output = model(input_data)
print(output)

# Print model summary
from torchsummary import summary
summary(model, input_size=input_shape)
