#!/usr/bin/env python3

# Danny Mikolay
# Neural Networks Semester Project
# Part 4: Final Solution

import torch
from torchvision import transforms
import numpy as np
from cnn import FacialFeatureExtractor
from triplet import TripletFaceDataset
from PIL import Image

# Load the trained network
network = FacialFeatureExtractor(numChannels=3, embeddingSize=64)
device = torch.device('cpu')  # Load the model onto the CPU
network.load_state_dict(torch.load("best_model.pth", map_location=device))
network.eval()

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
dataset = TripletFaceDataset(root_dir="./data_crop/train/", transform=transform)
danny_embeddings = {}

# Compute embeddings for all images in the dataset
embeddings = {}
for img_path, img_label in dataset.images:
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        img_embedding = network(img_tensor)
    embeddings[img_path] = img_embedding.detach().cpu().numpy()
    if img_label == 'danny':
        danny_embeddings[img_path] = img_embedding.detach().cpu().numpy()

# Classify images
threshold = 2.850 # Adjust threshold as needed
correct_pos = 0
correct_neg = 0
false_pos = 0
false_neg = 0

classified_images = {}
for img_path, img_embedding in embeddings.items():
    distances = []

    for danny_embedding in danny_embeddings.values():
        distance = np.linalg.norm(img_embedding - danny_embedding)
        distances.append(distance)
    avg_distance = np.mean(distances)
    if avg_distance < threshold:
        classified_images[img_path] = "danny"
        if img_path in danny_embeddings:
            correct_pos+=1
        else:
            false_pos+=1
    else:
        classified_images[img_path] = "not_danny"
        if img_path in danny_embeddings:
            false_neg+=1
        else:
            correct_neg+=1


# Print classification results
for img_path, class_label in classified_images.items():
    print(f"Image: {img_path}, Class: {class_label}")

total = false_neg + false_pos + correct_pos + correct_neg
accuracy = (correct_pos + correct_neg) / total

print(f"\nFalse Negatives: {false_neg} / {false_neg + correct_neg}")
print(f"False Positives: {false_pos} / {false_pos + correct_pos}")
print(f"Danny Classifications: {false_neg} / {false_neg + correct_pos}")
print(f"Other Classifications: {false_pos} / {false_pos + correct_neg}")
print(f"Total Accuracy: {accuracy}\n")