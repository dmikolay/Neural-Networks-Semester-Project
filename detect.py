#!/usr/bin/env python3

# Danny Mikolay
# Neural Networks Semester Project
# Part 3: First Solution

import os
import cv2
import argparse
import mediapipe as mp

def detect_faces(image_path):
    # Initialize Mediapipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    # Load image
    image = cv2.imread(image_path)

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(image_rgb)

        # Draw detection results
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)

    # Display the output image
    cv2.imshow('Face Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Face detection with Mediapipe")
    parser.add_argument("image_path", type=str, help="Path to input image")
    args = parser.parse_args()

    # Call function to detect faces in the image
    detect_faces(args.image_path)

if __name__ == "__main__":
    main()