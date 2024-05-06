#!/usr/bin/env python3

# Danny Mikolay
# Neural Networks Semester Project
# Part 3: First Solution

import os
import cv2
import argparse

def resize_images_in_directory(input_dir, output_dir, target_size=(196, 196)):
    '''
    Resize all images in the input directory to the target size and save them in the output directory.

    Parameters:
        input_dir (str): Path to the directory containing input images.
        output_dir (str): Path to the directory where resized images will be saved.
        target_size (tuple): Desired dimensions of the resized images. Default is (100, 100).
    '''
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through all files and directories in the input directory
    for item in os.listdir(input_dir):
        # Construct full path for the item
        item_path = os.path.join(input_dir, item)
        # Construct corresponding output path
        output_item_path = os.path.join(output_dir, item)

        if os.path.isdir(item_path):
            # If item is a directory, recursively call resize_images_in_directory
            resize_images_in_directory(item_path, output_item_path, target_size)
        elif item.endswith(('.JPG', '.jpg', '.jpeg', '.png', '.gif')):
            # If item is an image file
            # Open image file
            img = cv2.imread(item_path)

            # Check if img is None (indicating failure to read the image file)
            if img is None:
                print(f"Failed to read {item_path}. Skipping...")
                continue  # Skip to the next iteration of the loop

            # Calculate aspect ratio
            aspect_ratio = img.shape[1] / img.shape[0]

            # Determine new dimensions while preserving aspect ratio
            new_width = int(target_size[1] * aspect_ratio)
            new_height = target_size[1]

            # Resize image
            resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Save resized image
            cv2.imwrite(output_item_path, resized_img)

            print(f"Resized {item} and saved as {output_item_path}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Resize images in a directory.")
    parser.add_argument("input_dir", type=str, help="Path to the input directory containing images.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory where resized images will be saved.")
    parser.add_argument("--width", type=int, default=196, help="Width of the resized images. Default is 200.")
    parser.add_argument("--height", type=int, default=196, help="Height of the resized images. Default is 200.")
    args = parser.parse_args()

    # Call resize_images_in_directory to process images
    resize_images_in_directory(args.input_dir, args.output_dir, target_size=(args.width, args.height))

if __name__ == "__main__":
    main()
