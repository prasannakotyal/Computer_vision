import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthWiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.depth_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
        )
        self.point_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

class MobileNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.first_conv = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        # Main body
        self.conv1 = DepthWiseConv(32, 64, 1)
        self.conv2 = DepthWiseConv(64, 128, 2)
        self.conv3 = DepthWiseConv(128, 128, 1)
        self.conv4 = DepthWiseConv(128, 256, 2)
        self.conv5 = DepthWiseConv(256, 256, 1)
        self.conv6 = DepthWiseConv(256, 512, 2)
        # 5 repeated convolutions
        self.conv7 = DepthWiseConv(512, 512, 1)
        self.conv8 = DepthWiseConv(512, 512, 1)
        self.conv9 = DepthWiseConv(512, 512, 1)
        self.conv10 = DepthWiseConv(512, 512, 1)
        self.conv11 = DepthWiseConv(512, 512, 1)
        self.conv12 = DepthWiseConv(512, 1024, 2)
        self.conv13 = DepthWiseConv(1024, 1024, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.relu(self.first_conv(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Make the model
my_mobile_net = MobileNet(num_classes=1000)
print(my_mobile_net)