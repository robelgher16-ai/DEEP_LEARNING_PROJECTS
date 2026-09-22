# Fashion MNIST CNN Classifier

A deep learning image classification project using a Convolutional Neural Network (CNN) built with PyTorch.

The project classifies Fashion-MNIST images into 10 clothing categories and provides both a FastAPI backend and a Streamlit frontend for real-time prediction.

## Project Overview

This project demonstrates an end-to-end deep learning workflow:

- Dataset preparation
- Image preprocessing
- CNN model development
- Training and validation
- Early stopping
- Model evaluation
- Model saving
- FastAPI inference API
- Streamlit web interface
- Local deployment
- Cloud deployment preparation

## Dataset

The project uses the Fashion-MNIST dataset.

- 70,000 grayscale images
- Image size: 28 × 28 pixels
- 10 classes
- 60,000 training images
- 10,000 test images

### Classes

1. T-shirt/Top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle Boot

## CNN Architecture

The model contains:

```text
Input
  ↓
Conv2D (1 → 16)
  ↓
BatchNorm
  ↓
ReLU
  ↓
MaxPool
  ↓
Conv2D (16 → 32)
  ↓
BatchNorm
  ↓
ReLU
  ↓
MaxPool
  ↓
Flatten
  ↓
Linear (1568 → 128)
  ↓
ReLU
  ↓
Dropout
  ↓
Linear (128 → 10)
  ↓
Class Prediction
```
