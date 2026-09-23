# Fashion MNIST CNN Classifier

A production-ready Deep Learning image classification project built with **PyTorch**, **FastAPI**, **Streamlit**, and deployed on **Render** and **Streamlit Community Cloud**.

## Live Demo

**Streamlit App:** https://deeplearningprojects-robel.streamlit.app/

**FastAPI API:** https://fashion-mnist-cnn-api.onrender.com

**Swagger Docs:** https://fashion-mnist-cnn-api.onrender.com/docs

---

## Project Overview

This project classifies grayscale clothing images into **10 Fashion-MNIST categories** using a Convolutional Neural Network (CNN). It demonstrates the complete machine learning workflow from model training to cloud deployment.

### Features

- PyTorch CNN model
- Batch Normalization & Dropout
- Early Stopping
- FastAPI REST API
- Streamlit interactive UI
- Render cloud deployment
- Streamlit Community Cloud deployment
- Top-3 prediction probabilities

---

## Dataset

**Fashion-MNIST**

- 70,000 grayscale images
- Image size: 28×28
- 10 clothing classes

### Classes

| Label | Class       |
| ----- | ----------- |
| 0     | T-shirt/Top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle Boot  |

---

## CNN Architecture

```text
Input (1×28×28)
      │
Conv2D (1→16)
      │
BatchNorm
      │
ReLU
      │
MaxPool
      │
Conv2D (16→32)
      │
BatchNorm
      │
ReLU
      │
MaxPool
      │
Flatten (1568)
      │
Linear (1568→128)
      │
ReLU
      │
Dropout
      │
Linear (128→10)
      │
Softmax Prediction
```

---

## Project Structure

```text
Fashion_MNIST_CNN_Classifier/
│
├── app/
│   └── streamlit_app.py
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── src/
│   ├── __init__.py
│   └── model.py
│
├── models/
│   └── fashion_cnn_best.pth
│
├── notebooks/
│   └── Fashion_MNIST_CNN.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Model Training

| Parameter        |            Value |
| ---------------- | ---------------: |
| Optimizer        |             Adam |
| Loss Function    | CrossEntropyLoss |
| Epochs           |               20 |
| Batch Size       |               64 |
| Validation Split |              10% |
| Early Stopping   |     Patience = 3 |
| Input Size       |            28×28 |

---

## Deployment Architecture

```text
          User
            │
            ▼
 Streamlit Community Cloud
            │
      HTTP Request
            │
            ▼
   FastAPI (Render Cloud)
            │
            ▼
   PyTorch CNN Model (.pth)
            │
            ▼
      JSON Prediction
            │
            ▼
      Streamlit Result
```

---

## API Endpoint

### POST `/predict`

Upload a clothing image and receive the predicted class.

### Example Response

```json
{
  "prediction": "Coat",
  "confidence": 98.42,
  "top3": [
    {
      "class": "Coat",
      "confidence": 98.42
    },
    {
      "class": "Pullover",
      "confidence": 0.91
    },
    {
      "class": "Shirt",
      "confidence": 0.32
    }
  ]
}
```

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- FastAPI
- Uvicorn
- Streamlit
- Pillow
- Requests
- Matplotlib
- Scikit-learn

---

## Running Locally

### Install

```bash
pip install -r requirements.txt
```

### Start FastAPI

```bash
uvicorn api.main:app --reload
```

### Start Streamlit

```bash
streamlit run app/streamlit_app.py
```

---

## What I Learned

- Convolutional Neural Networks (CNNs)
- Image preprocessing
- Batch Normalization
- Dropout regularization
- Early Stopping
- Model serialization with PyTorch
- Building REST APIs using FastAPI
- Connecting Streamlit with FastAPI
- Deploying AI applications to Render
- Deploying Streamlit Community Cloud

---

## Author

**Robel Gebregziabher**

Information Technology Student | AI Engineer

GitHub: `robelgher16-ai`
