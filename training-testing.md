# Image Classification Model Training and Testing Documentation

## Overview

This document describes the process of training and testing a deep learning model based on ResNet architecture for image classification. The goal of the model is to classify images into predefined categories accurately.

The project uses PyTorch and TIMM libraries for model building and training, along with standard image preprocessing techniques.

---

## Model Architecture

- The model is based on the **ResNet architecture**, specifically a variant like `resnet50`.
- ResNet (Residual Network) introduces skip connections to solve the vanishing gradient problem in deep networks.
- The model is initialized with random weights or pretrained weights (if specified).
- The final fully connected layer is adjusted to output the correct number of classes (e.g., 5 classes).

---

## Dataset

- Images are stored in directories or listed in CSV files.
- Labels correspond to image classes.
- Dataset is split into:
  - **Training set**: used to train the model.
  - **Validation/test set**: used to evaluate the model’s performance on unseen data.

---

## Data Preprocessing

- Images are loaded and converted to RGB.
- Resized to a fixed size (e.g., 224x224).
- Normalized using ImageNet mean and standard deviation:
  - Mean: `[0.485, 0.456, 0.406]`
  - Std: `[0.229, 0.224, 0.225]`
- Converted to PyTorch tensors with shape `(Batch, Channels, Height, Width)`.

---

## Training Procedure

1. **Model Initialization**
   - Load ResNet model architecture with `num_classes` matching dataset.
   - Optionally load pretrained weights (e.g., ImageNet).

2. **Loss Function**
   - Use Cross-Entropy Loss for multi-class classification.

3. **Optimizer**
   - Typically use Adam or SGD optimizer.
   - Learning rate and weight decay configured per experiment.

4. **Training Loop**
   - For each epoch:
     - Forward pass: input images through model.
     - Compute loss with predictions and true labels.
     - Backward pass: compute gradients.
     - Update weights with optimizer step.
   - Monitor training loss and accuracy.

5. **Validation**
   - After each epoch, evaluate the model on validation data.
   - Track validation accuracy and loss to detect overfitting.

6. **Checkpointing**
   - Save the model state dict periodically or on improved validation accuracy.

---

## Testing / Inference

- Load the trained model checkpoint.
- Preprocess input images the same way as training.
- Forward pass images through the model.
- Use softmax to obtain class probabilities.
- Select the class with highest probability as prediction.
- Evaluate accuracy on test set.

---

## Evaluation Metrics

- **Accuracy:** Percentage of correctly classified images.
- **Confusion Matrix:** To visualize class-wise prediction performance.
- **Precision, Recall, F1-Score:** For detailed class-specific evaluation (optional).

---

## Grad-CAM Visualization (Explainability)

- Use Grad-CAM to visualize regions in the image that contributed most to the prediction.
- Helps to interpret model decisions.
- Overlay heatmaps on original images.

---

## Example Code Snippets

### Loading Model and Checkpoint

```python
import timm
import torch

model = timm.create_model('resnet50', pretrained=False, num_classes=5)
checkpoint = torch.load('path_to_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
