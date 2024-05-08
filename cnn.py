#!/usr/bin/env python3

# Danny Mikolay
# Neural Networks Semester Project
# Part 3: First Solution

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class FacialFeatureExtractor(nn.Module):
    def __init__(self, numChannels, embeddingSize, dropout_prob=0.5):
        super(FacialFeatureExtractor, self).__init__()

        # Define the layers for your CNN
        self.conv1 = nn.Conv2d(in_channels=numChannels, out_channels=96, kernel_size=(11,11), stride=(4,4))
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5,5), stride=(1,1))
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3,3), stride=(1,1))
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1))

        self.maxpool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.relu = nn.ReLU()

        self.batchnorm1 = nn.BatchNorm2d(num_features=96)
        self.batchnorm2 = nn.BatchNorm2d(num_features=256)

        # Define the fully connected layers for embedding
        self.fc1 = nn.Linear(in_features=256, out_features=128)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(in_features=128, out_features=embeddingSize)

    def forward(self, x):
        # Forward pass through the network

        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.batchnorm1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.batchnorm2(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x) 
        x = self.fc2(x)

        return x
