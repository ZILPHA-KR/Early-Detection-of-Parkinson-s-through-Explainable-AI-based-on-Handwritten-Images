

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

class HybridVGG(nn.Module):
    def __init__(self, num_classes=2):
        super(HybridVGG, self).__init__()

        # Load pre-trained VGG16 and VGG19
        self.vgg16 = models.vgg16(pretrained=True).features
        self.vgg19 = models.vgg19(pretrained=True).features

        # Freeze the weights for both models (optional)
        for param in self.vgg16.parameters():
            param.requires_grad = False
        for param in self.vgg19.parameters():
            param.requires_grad = False

        # Define the fully connected layers for the hybrid model
        self.fc1 = nn.Linear(512 * 2 * 4 * 4, 4096)  # Adjusted for 4x4 output
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        # Pass the input through both VGG16 and VGG19
        x1 = self.vgg16(x)  # VGG16 output
        x2 = self.vgg19(x)  # VGG19 output

        # Print shapes for debugging
        print(f'VGG16 Output Shape: {x1.shape}')  # Debugging
        print(f'VGG19 Output Shape: {x2.shape}')  # Debugging

        # Flatten the outputs
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)

        # Concatenate the outputs from both networks
        x = torch.cat((x1, x2), dim=1)  # Concatenate along the feature dimension
        print(f'Concatenated Output Shape: {x.shape}')  # Debugging

        # Pass through the fully connected layers
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = nn.Dropout()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = nn.Dropout()(x)
        x = self.fc3(x)

        return x

# Instantiate the hybrid model
model = HybridVGG(num_classes=2)

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define dataset path
DATASET_PATH = '/content/drive/MyDrive/spiral_drawing_dataset/Spiral'

# Data transformations
data_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust mean/std based on your dataset
])

# Load the dataset (assuming it's organized in folders for each class)
dataset = datasets.ImageFolder(root=DATASET_PATH, transform=data_transforms)

# Split the dataset into training, validation, and test sets (70%, 15%, 15%)
train_size = int(0.8 * len(dataset))
val_size = int(0.05* len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create data loaders
batch_size = 32  # Adjust based on your system's memory capacity
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Initialize lists to store metrics
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# Training loop
num_epochs = 10  # Set the number of epochs you want to train for
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Calculate average loss and accuracy for training
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Validation Loop
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    # Calculate average loss and accuracy for validation
    val_loss /= len(val_loader)
    val_accuracy = 100 * val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
          f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

# Plotting accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
plt.title('Accuracy per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid()

# Plotting loss
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.title('Loss per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

    