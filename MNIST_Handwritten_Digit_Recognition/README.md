# MNIST Handwritten Digit Recognition using CNN

A complete deep learning project for recognizing handwritten digits (0–9) using a Convolutional Neural Network (CNN) built with PyTorch.

The project covers the complete machine learning workflow from dataset preparation and CNN training to evaluation, inference, web application development, REST API development, and cloud deployment.

## Live Demo

### Streamlit Application

https://mnist-digit-cnn-robel.streamlit.app/

Interactive web application for uploading handwritten digit images and receiving predictions from the trained CNN.

### FastAPI Backend

https://mnist-digit-api-vedw.onrender.com

Online REST API serving the trained MNIST CNN model.

### Swagger API Documentation

https://mnist-digit-api-vedw.onrender.com/docs

Interactive API documentation for testing the prediction endpoint.

---

## Project Overview

Handwritten digit recognition is a classic computer vision and image classification problem.

This project demonstrates how a Convolutional Neural Network can learn visual patterns from handwritten digits and classify images into one of ten classes.

The project follows this workflow:

```text
Dataset
   ↓
Data Preparation
   ↓
CNN Architecture
   ↓
Training
   ↓
Validation
   ↓
Evaluation
   ↓
Inference
   ↓
Streamlit Application
   ↓
FastAPI Backend
   ↓
Cloud Deployment
```

## Objective

Build a CNN capable of classifying handwritten grayscale images into one of ten digit classes:

```text
0 1 2 3 4 5 6 7 8 9
```

The project was initially developed as a focused Jupyter Notebook and then extended into a deployable deep learning application.

---

# Dataset

The project uses the **MNIST handwritten digit dataset**.

| Property        | Value                      |
| --------------- | -------------------------- |
| Training images | 60,000                     |
| Test images     | 10,000                     |
| Image size      | 28 × 28                    |
| Channels        | 1                          |
| Color           | Grayscale                  |
| Classes         | 10                         |
| Task            | Multi-class classification |

The dataset is stored locally and excluded from GitHub through `.gitignore`.

---

# CNN Architecture

The final CNN contains two convolutional blocks followed by fully connected layers.

```text
Input
1 × 28 × 28

        ↓
Conv2D
1 → 32 channels
Kernel = 3 × 3
Padding = 1

        ↓
BatchNorm
        ↓
ReLU
        ↓
MaxPool 2 × 2

32 × 14 × 14

        ↓
Conv2D
32 → 64 channels
Kernel = 3 × 3
Padding = 1

        ↓
BatchNorm
        ↓
ReLU
        ↓
MaxPool 2 × 2

64 × 7 × 7

        ↓
Flatten

3136

        ↓
Linear

128

        ↓
ReLU
        ↓
Dropout 0.3

        ↓
Linear

10

        ↓
Output
10 digit classes
```

### Shape progression

```text
1 × 28 × 28
      ↓
32 × 28 × 28
      ↓
32 × 14 × 14
      ↓
64 × 14 × 14
      ↓
64 × 7 × 7
      ↓
3136
      ↓
128
      ↓
10
```

---

# Training

The model was trained using PyTorch.

| Configuration       | Value             |
| ------------------- | ----------------- |
| Framework           | PyTorch           |
| Optimizer           | Adam              |
| Learning Rate       | 0.001             |
| Maximum Epochs      | 20                |
| Loss Function       | CrossEntropyLoss  |
| Early Stopping      | Patience = 3      |
| LR Scheduler        | ReduceLROnPlateau |
| Batch Normalization | Yes               |
| Dropout             | 0.3               |

The best model based on validation loss is saved as:

```text
models/best_digit_cnn.pth
```

---

# Evaluation

The model evaluation includes:

- Test loss
- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix
- Error analysis

The confusion matrix and error analysis were used to examine which digits were easier or harder for the CNN to classify.

---

# Inference

The project includes reusable inference functionality for:

- Single-image prediction
- Batch prediction
- Prediction confidence
- Probability distribution across the ten classes

Uploaded images are automatically converted to grayscale and preprocessed before being passed to the model.

The preprocessing also handles different image backgrounds and formats to make the deployed application more robust.

---

# Streamlit Application

The Streamlit application provides an interactive interface for the trained CNN.

Users can:

1. Upload a handwritten digit image.
2. Automatically preprocess the image.
3. Run the trained CNN.
4. View the predicted digit.
5. View the prediction confidence.

### Live Application

https://mnist-digit-cnn-robel.streamlit.app/

---

# FastAPI Backend

The project also provides a REST API for model inference.

## Root Endpoint

```text
GET /
```

Example response:

```json
{
  "message": "MNIST Digit Recognition API is running",
  "model": "DigitCNN"
}
```

## Prediction Endpoint

```text
POST /predict
```

The endpoint accepts an image file and returns the predicted digit and confidence.

Example response:

```json
{
  "predicted_digit": 7,
  "confidence": 99.42
}
```

## Swagger Documentation

Interactive API documentation:

https://mnist-digit-api-vedw.onrender.com/docs

## Live API

https://mnist-digit-api-vedw.onrender.com

---

# Deployment

The project is deployed using two services.

### Streamlit Community Cloud

Used for the interactive frontend:

https://mnist-digit-cnn-robel.streamlit.app/

### Render

Used for the FastAPI backend:

https://mnist-digit-api-vedw.onrender.com

The backend provides the `/predict` endpoint used for online model inference.

---

# Project Structure

```text
MNIST_Handwritten_Digit_Recognition/
│
├── api/
│   └── main.py
│
├── app/
│   ├── streamlit_app.py
│   └── requirements.txt
│
├── data/
│   └── MNIST/
│
├── models/
│   └── best_digit_cnn.pth
│
├── notebooks/
│   └── MNIST_Handwritten_Digit_Recognition.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# Installation

Clone the main deep learning repository:

```bash
git clone https://github.com/robelgher16-ai/DEEP_LEARNING_PROJECTS.git
```

Navigate to the MNIST project:

```bash
cd DEEP_LEARNING_PROJECTS/MNIST_Handwritten_Digit_Recognition
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Run Streamlit Locally

From the MNIST project directory:

```bash
streamlit run app/streamlit_app.py
```

The application will be available locally through the Streamlit URL shown in the terminal.

---

# Run FastAPI Locally

From the MNIST project directory:

```bash
uvicorn api.main:app --reload
```

Open the local Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

# Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- Pillow
- Scikit-learn
- Matplotlib
- Jupyter Notebook
- Streamlit
- FastAPI
- Uvicorn
- Render
- Git
- GitHub

---

# Key Learning Outcomes

This project demonstrates practical experience with:

- Convolutional Neural Networks
- Image preprocessing
- Data augmentation
- Batch normalization
- Dropout
- Model training
- Validation
- Early stopping
- Learning-rate scheduling
- Classification metrics
- Confusion matrices
- Error analysis
- Model serialization
- PyTorch inference
- Streamlit application development
- REST API development
- Swagger API documentation
- Cloud deployment
- Git and GitHub workflow
- Monorepo project organization

---

# Future Improvements

Possible future improvements include:

- More advanced CNN architectures
- Hyperparameter tuning
- More extensive augmentation experiments
- Improved image preprocessing
- Model explainability
- Docker containerization
- Automated testing
- CI/CD
- Model monitoring
- API monitoring and logging

---

# Author

**Robel Gebregziabher**

AI / Machine Learning / Deep Learning

GitHub:

https://github.com/robelgher16-ai
