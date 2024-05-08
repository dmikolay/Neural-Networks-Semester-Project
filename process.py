#!/usr/bin/env python3

# Danny Mikolay
# Neural Networks Semester Project
# Part 3: First Solution

import os
import cv2
import argparse
import mediapipe as mp

def crop_face(image):
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Process the image
        results = face_detection.process(image_rgb)
        # Check if a face is detected
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                # Get the bounding box coordinates of the detected face
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                # Crop the image to include only the face region
                face_cropped = image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                return face_cropped
        else:
            print("No face detected in the image.")
            return None

def preproc(image, target_size=(128, 128)):

    # Crop face
    cropped_face = crop_face(image)

    # Check if face is detected and cropped successfully
    if cropped_face is None:
        return None

    # Resize cropped face to target size
    resized_img = cv2.resize(cropped_face, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalize image
    normalized_img = cv2.normalize(resized_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Convert image to grayscale
    gray_img = cv2.cvtColor(normalized_img, cv2.COLOR_BGR2GRAY)

    # Perform histogram equalization
    equalized_img = cv2.equalizeHist(gray_img)

    return equalized_img


def resize_images_in_directory(input_dir, output_dir, target_size=(128, 128)):
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
            # Read image
            print(item_path)
            img = cv2.imread(item_path)

            # Check if img is None (indicating failure to read the image file)
            if img is None:
                print(f"Failed to read {item_path}. Skipping...")
                continue

            processed_img = preproc(img, target_size)

            # Check if img is None (indicating failure to read the image file)
            if processed_img is None:
                print(f"No face detected in {item}. Skipping...")
                continue

            # Save resized image
            cv2.imwrite(output_item_path, processed_img)

            print(f"Resized {item} and saved as {output_item_path}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Resize images in a directory.")
    parser.add_argument("input_dir", type=str, help="Path to the input directory containing images.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory where resized images will be saved.")
    parser.add_argument("--width", type=int, default=128, help="Width of the resized images. Default is 128.")
    parser.add_argument("--height", type=int, default=128, help="Height of the resized images. Default is 128.")
    args = parser.parse_args()

    # Call resize_images_in_directory to process images
    resize_images_in_directory(args.input_dir, args.output_dir, target_size=(args.width, args.height))

if __name__ == "__main__":
    main()
