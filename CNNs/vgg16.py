import torch
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self,num_classes = 1000):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2,stride=2),

            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2,stride=2),

            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2,stride=2),

            nn.Conv2d(256,512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2,stride=2),

            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2,stride=2),                                        

        )

        self.classifier = nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.5),        
            nn.Linear(4096,1000)
        )

    def forward(self,x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x


model = VGG16()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Print model summary
from torchsummary import summary
summary(model,input_size=(3,224,224),device = str(device))