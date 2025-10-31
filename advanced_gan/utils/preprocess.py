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
wandb.init(project="fashion_mnist", config={
    "dataset": "fashion_MNIST",
    "framework": "PyTorch",
    "model": "adv_DCGAN"
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seed for reproducibility
torch.manual_seed(42)

# Define preprocessing for FashionMNIST
fashion_mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download FashionMNIST training dataset
dataset = datasets.FashionMNIST(root='./data', 
                                              train=True, 
                                              download=True, 
                                              transform=fashion_mnist_transform)

dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

print(f"Successfully downloaded FashionMNIST dataset with {len(dataset)} samples.")

# Get one batch to check shapes
real_batch = next(iter(dataloader))
images, labels = real_batch

# Print shape of batch to verify
print(images.shape)
print(labels.shape)
