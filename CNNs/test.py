import torch
import torch.nn as nn


class CNN:
    def __init__(self,input_shape,num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0],32,kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)

        flatten_size = 64*input_shape[1]//4*input_shape[2]//4
        self.fc = nn.Linear(flatten_size,num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return self.softmax(x)


