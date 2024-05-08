#!/usr/bin/env python3

# Danny Mikolay
# Neural Networks Semester Project
# Part 4: Final Solution

import cv2
from process import preproc, crop_face
import os
import threading
import torch
from torchvision import transforms
import numpy as np
from cnn import FacialFeatureExtractor
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import to_tensor
from triplet import TripletFaceDataset
from PIL import Image

def main():

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

    threshold = 2.850

    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    paused = False
    saved_frame_path = None
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame.")
            break
        
        cv2.imshow('Webcam', frame)

        key = cv2.waitKey(1)
        
        if key & 0xFF == ord(' '):
            paused = not paused
            if paused:
                ret, frame = cap.read()  # Read frame again after pause
                cv2.waitKey(5000)  # Pauses for 10 seconds
                if ret:
                    saved_frame_path = 'paused_frame.jpg'
                    cv2.imwrite(saved_frame_path, frame)
                    img = cv2.imread(saved_frame_path)
                    processed = preproc(img)
                    cv2.imwrite(saved_frame_path, processed)
                    img = Image.open(saved_frame_path).convert("RGB")
                    new_image_tensor = transform(img).unsqueeze(0)  # Add batch dimension
                    with torch.no_grad():
                        new_embedding = network(new_image_tensor)
                    new_image_embedding = new_embedding.detach().cpu().numpy()
                    distances = []
                    for danny_embedding in danny_embeddings.values():
                        distance = np.linalg.norm(new_image_embedding - danny_embedding)
                        distances.append(distance)
                    avg_distance = np.mean(distances)
                    if avg_distance < threshold:
                        classification = "danny"
                    else:
                        classification = "not_danny"

                    # Print classification result
                    print(f"New Image Class: {classification}")
                    cv2.imshow("Face in Frame", processed)
                else:
                    print("Error: Could not read frame after pause.")
        elif key & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

    # Delete saved image if exists
    if saved_frame_path is not None and os.path.exists(saved_frame_path):
        os.remove(saved_frame_path)

if __name__ == "__main__":
    main()
