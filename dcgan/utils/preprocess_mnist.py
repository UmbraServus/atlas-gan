 #!/usr/bin/env python3
# Imports and setup
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision
from torch.utils.data import DataLoader
import wandb
import matplotlib.pyplot as plt
import numpy as np
import yaml
from types import SimpleNamespace

# Initialize W&B project
wandb.init(project="mnist_dcgan", config={
    "dataset": "MNIST",
    "framework": "PyTorch",
    "model": "DCGAN"
})

with open("configs/dcgan_config.yaml", "r") as file:
    c = yaml.safe_load(file)
    c = SimpleNamespace(**c)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seed for reproducibility
torch.manual_seed(42)

# Define preprocessing for MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download MNIST training dataset
dataset = datasets.MNIST(root='./data',
                         train=True,
                         download=True,
                         transform=transform)

# Create DataLoader with batch size and shuffle
batch_size = 128
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Get one batch to check shapes
real_batch = next(iter(dataloader))
images, labels = real_batch

# Print shape of batch to verify
print(images.shape)
print(labels.shape)

# Undo normalization for visual
images = images * 0.5 + 0.5
plt.figure(figsize=(8, 8))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(images[i].squeeze(), cmap='gray')
    plt.axis('off')
plt.show()