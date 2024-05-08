#!/usr/bin/env python3

# Danny Mikolay
# Neural Networks Semester Project
# Part 3: First Solution

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torchsummary import summary
from sklearn.model_selection import train_test_split
from triplet import TripletFaceDataset
from cnn import FacialFeatureExtractor
import numpy as np

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjust normalization parameters as needed
])

augmentation_transform = transforms.Compose([
    transforms.RandomRotation(degrees=10),  # Randomly rotate the image by a certain degree
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly adjust brightness, contrast, saturation, and hue
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Random perspective transformation
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image
])

# Load dataset
dataset = TripletFaceDataset(root_dir="./data_crop/train/", transform=augmentation_transform)
my_best_model = "./best_model.pth"

# Split dataset into training and validation sets
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Create data loaders for training and validation sets
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

# Define network, optimizer, and learning rate scheduler
network = FacialFeatureExtractor(numChannels=3, embeddingSize=64)

print("\nNetwork summary:\n")
summary(network, input_size=(3, 128, 128))

optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Define triplet loss function
def triplet_loss(anchor, positive, negative, margin=1.0):
    distance_positive = F.pairwise_distance(anchor, positive)
    distance_negative = F.pairwise_distance(anchor, negative)
    loss = torch.mean(torch.relu(distance_positive - distance_negative + margin))
    return loss

# Set up data loader
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
num_epochs = 20
best_acc = 0.00

for epoch in range(num_epochs):
    
    network.train()  # Set network to training mode
    print(f"\nEPOCH {epoch + 1} STARTING")

    for anchor_images, positive_images, negative_images in train_loader:
        # Forward pass
        anchor_embeddings = network(anchor_images)
        positive_embeddings = network(positive_images)
        negative_embeddings = network(negative_images)

        # Compute triplet loss
        loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Update learning rate scheduler
    scheduler.step()

    # Print loss
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # Validation
    network.eval()  # Set network to evaluation mode
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for val_anchor_images, val_positive_images, val_negative_images in val_loader:
            # Compute embeddings
            val_anchor_embeddings = network(val_anchor_images)
            val_positive_embeddings = network(val_positive_images)
            val_negative_embeddings = network(val_negative_images)

            # Compute accuracy
            distances_positive = F.pairwise_distance(val_anchor_embeddings, val_positive_embeddings)
            distances_negative = F.pairwise_distance(val_anchor_embeddings, val_negative_embeddings)
            correct = (distances_positive < distances_negative).sum().item()
            total_correct += correct
            total_samples += val_anchor_embeddings.size(0)  # Assuming the batch size is the same for anchor, positive, and negative samples

        # Compute validation accuracy
        accuracy = total_correct / total_samples
        print(f"Epoch {epoch + 1}, Validation Accuracy: {accuracy}")

        if accuracy > best_acc:
            print(f"Better validation accuracy achieved: {accuracy * 100:.2f}%")
            best_acc = accuracy
            print(f"Saving this model as: {my_best_model}")
            #torch.save(network.state_dict(), my_best_model)
