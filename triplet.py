#!/usr/bin/env python3

# Danny Mikolay
# Neural Networks Semester Project
# Part 3: First Solution

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class TripletFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_index = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = self.load_images()

    def load_images(self):
        images = []
        for cls_dir in self.classes:
            cls_images = []
            cls_dir_path = os.path.join(self.root_dir, cls_dir)
            if os.path.isdir(cls_dir_path):
                image_files = [img_name for img_name in os.listdir(cls_dir_path) if os.path.isfile(os.path.join(cls_dir_path, img_name)) and not img_name.startswith('.')]
                cls_images.extend([(os.path.join(cls_dir_path, img), cls_dir) for img in image_files])
            images.extend(cls_images)
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        anchor_path, anchor_label = self.images[index]

        # Select a positive example from the same class as the anchor
        positive_candidates = [(img_path, img_label) for img_path, img_label in self.images if img_label == anchor_label and img_path != anchor_path]
        positive_path, positive_label = positive_candidates[np.random.randint(0, len(positive_candidates))]

        # Select a negative example from a different class
        negative_candidates = [(img_path, img_label) for img_path, img_label in self.images if img_label != anchor_label]
        negative_path, negative_label = negative_candidates[np.random.randint(0, len(negative_candidates))]

        # Load images
        anchor_img = Image.open(anchor_path).convert("RGB")
        positive_img = Image.open(positive_path).convert("RGB")
        negative_img = Image.open(negative_path).convert("RGB")

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img
