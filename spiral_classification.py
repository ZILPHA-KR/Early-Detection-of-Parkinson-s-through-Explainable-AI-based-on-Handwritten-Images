
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from captum.attr import LayerGradCam
import matplotlib.colors as mcolors

# Define the HybridVGG model
class HybridVGG(nn.Module):
    def __init__(self, num_classes=2):
        super(HybridVGG, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True).features
        self.vgg19 = models.vgg19(pretrained=True).features
        
        # Freeze the weights
        for param in self.vgg16.parameters():
            param.requires_grad = False
        for param in self.vgg19.parameters():
            param.requires_grad = False

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 2 * 4 * 4, 4096)  
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x1 = self.vgg16(x)
        x2 = self.vgg19(x)
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = nn.Dropout()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = nn.Dropout()(x)
        x = self.fc3(x)
        return x

# Instantiate the model
model = HybridVGG(num_classes=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Dataset, data loaders, training loop, and evaluation code...
# Add your complete training and evaluation code here.

    