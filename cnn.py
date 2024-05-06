#!/usr/bin/env python3

# Danny Mikolay
# Neural Networks Semester Project
# Part 3: First Solution

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class FacialFeatureExtractor(nn.Module):
    def __init__(self, numChannels, embeddingSize):
        super(FacialFeatureExtractor, self).__init__()

        # Define the layers for your CNN
        self.conv1 = nn.Conv2d(in_channels=numChannels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # Define the fully connected layers for embedding
        self.fc1 = nn.Linear(in_features=8192, out_features=256)  # Adjusted input size
        self.fc2 = nn.Linear(in_features=256, out_features=embeddingSize)

    def forward(self, x):
        # Forward pass through the network
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.pool(x)

        # Flatten the output before passing it to fully connected layers
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
