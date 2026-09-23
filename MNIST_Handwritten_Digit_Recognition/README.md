# MNIST Vision AI

**Handwritten Digit Recognition using a PyTorch Convolutional Neural Network**

MNIST Vision AI is an end-to-end deep learning project that classifies handwritten digits (0–9) using a Convolutional Neural Network (CNN). The project includes model training, evaluation, a Streamlit web application, and a FastAPI backend for inference.

---

## Project Overview

This project demonstrates the complete deep learning workflow:

- Business problem definition
- Data preprocessing
- CNN model development
- Model training with validation
- Performance evaluation
- Streamlit deployment
- FastAPI REST API

---

## Project Structure

```text
MNIST_Handwritten_Digit_Recognition/
│
├── api/
│   └── main.py
│
├── app/
│   └── streamlit_app.py
│
├── data/
│
├── models/
│   └── best_digit_cnn.pth
│
├── notebooks/
│   └── MNIST_Handwritten_Digit_Recognition.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

**Dataset:** MNIST Handwritten Digits

- 70,000 grayscale images
- Image size: 28 × 28
- Classes: 10 digits (0–9)
- Training images: 60,000
- Test images: 10,000

---

## CNN Architecture

The model contains two convolutional blocks followed by a fully connected classifier.

| Layer       |       Output |
| ----------- | -----------: |
| Input       |  1 × 28 × 28 |
| Conv2D (32) | 32 × 28 × 28 |
| MaxPool     | 32 × 14 × 14 |
| Conv2D (64) | 64 × 14 × 14 |
| MaxPool     |   64 × 7 × 7 |
| Flatten     |         3136 |
| Linear      |          128 |
| Dropout     |          0.3 |
| Output      |           10 |

---

## Technologies Used

- Python
- PyTorch
- TorchVision
- NumPy
- Matplotlib
- Streamlit
- FastAPI

---

## Training Features

- CNN image classifier
- Batch Normalization
- Dropout Regularization
- Adam Optimizer
- Learning Rate Scheduler
- Early Stopping
- Best Model Saving

---

## Streamlit Application

The web application allows users to:

- Upload PNG, JPG, or JPEG images
- Automatically preprocess color images
- Convert to grayscale
- Detect and center the digit
- Resize to 28 × 28
- Predict the handwritten digit
- Display prediction confidence

Run locally:

```bash
streamlit run app/streamlit_app.py
```

---

## FastAPI Backend

The REST API provides image inference.

Run locally:

```bash
uvicorn api.main:app --reload
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

Example response:

```json
{
  "predicted_digit": 7,
  "confidence": 99.42
}
```

---

## Installation

```bash
git clone https://github.com/robelgher16-ai/DEEP_LEARNING_PROJECTS.git

cd MNIST_Handwritten_Digit_Recognition

pip install -r requirements.txt
```

---

## Future Improvements

- Draw digit canvas inside Streamlit
- Top-3 prediction visualization
- Docker containerization
- Render deployment
- Mobile-friendly interface

---

## Author

**Robel Gebregziabher**

AI Engineering • Deep Learning • Computer Vision
