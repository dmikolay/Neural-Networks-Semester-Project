Danny Mikolay
Neural Networks Semester Project

# Part 1: Conceptual Design
### Introduction
The face recognition project aims to develop a neural network-based solution to extract facial features for use as biometric templates. These features will then be compared using metrics like Euclidean or cosine distance to determine whether two faces belong to the same person or different individuals. While face detection will be handled by existing packages (most likely MediaPipe), the feature extractor will be custom-built. The final objective is to recognize a specific face among a small set of other faces, including friends residing in Carroll Hall, with the option to implement real-time processing of webcam streams.

### Solution Overview
The solution will involve several key steps:

1. **Data Acquisition**: Obtain datasets containing images of faces for training, validation, and testing. In addition to public datasets like FRGC and BioID, images of friends residing in Carroll Hall can be included to personalize the recognition system and increase the relevance of the project.

2. **Preprocessing**: Preprocess the images to ensure uniformity and enhance the features relevant for recognition. This may involve tasks such as resizing, normalization, and augmentation to increase the robustness of the model. Specifically, images of friends can be collected in various lighting conditions and poses to mimic real-world scenarios.

3. **Feature Extraction**: Develop a neural network architecture to extract facial features or embeddings from the preprocessed images. This network will need to be trained using the acquired datasets, including images of friends, to learn discriminative features that are effective for face recognition.

4. **Training**: Train the feature extractor using the training dataset, optimizing its parameters to minimize the difference between embeddings of the same person and maximize the difference between embeddings of different individuals. Techniques like triplet loss may be employed to enforce this.

5. **Validation**: Validate the trained model using a separate validation dataset to assess its performance on unseen data and prevent overfitting. Adjustments to the model architecture or training parameters may be made based on validation results.

6. **Testing**: Evaluate the final model's performance using the test dataset, which should contain faces not seen during training or validation, including those of friends living in Carroll Hall. This step provides a realistic assessment of the model's ability to generalize to new faces.

7. **Deployment**: Optionally, develop a real-time version of the software that processes webcam streams for live face recognition. This may involve optimizing the model for inference speed and integrating it with appropriate hardware for efficient processing.

### Dataset Requirements
For effective training, validation, and testing of the face recognition model, the following dataset requirements are identified:

- **Training Set**: 
  - Contains at least 20 different samples per face for multiple individuals, including friends residing in Carroll Hall.
  - Should cover a diverse range of poses, expressions, lighting conditions, and occlusions to ensure robustness.

- **Validation Set**:
  - Consists of at least 10 different samples per face for individuals not present in the training set, including friends from Carroll Hall.
  - Helps monitor the model's performance during training and prevent overfitting.

- **Test Set**:
  - Comprises at least 10 different samples per face for evaluation, including faces not seen during training or validation and friends residing in Carroll Hall.
  - Assesses the model's generalization ability in real-world scenarios.

### Conclusion
In summary, the face recognition project involves developing a custom neural network for feature extraction from facial images, training it using diverse datasets, including images of friends living in Carroll Hall, and evaluating its performance on unseen data. By integrating images of friends into the dataset, the recognition system becomes more personalized and relevant to the project's context. Through this high-level solution, I aim to build an accurate and robust face recognition system capable of identifying individuals, including friends, in real-time scenarios.

# Part 2: Data Acquisition and Curation/Preprocessing
TBD

# Part 3: The First Solution With Results on Known Data
TBD

# Part 4: Final Solution With Results on Unknown Data
TBD
