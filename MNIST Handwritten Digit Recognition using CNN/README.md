# MNIST Handwritten Digit Recognition using CNN

A production-style Deep Learning project built with **PyTorch** that recognizes handwritten digits using a custom Convolutional Neural Network (CNN).

## Business Problem

Banks, postal services, and document processing systems need to automatically recognize handwritten digits from scanned documents. This project builds an end-to-end AI solution for that task.

## Dataset

- **Name:** MNIST
- **Training:** 60,000 images
- **Testing:** 10,000 images
- **Classes:** 10 (0–9)
- **Image Size:** 28×28 grayscale

## Project Architecture

```text
src/
├── config.py
├── dataset.py
├── transforms.py
├── model.py
├── engine.py
├── train.py
├── evaluate.py
├── inference.py
├── gradcam.py
├── logger.py
├── visualize.py
└── utils.py
```

## CNN Architecture

- Conv2D
- ReLU
- MaxPool
- Conv2D
- ReLU
- MaxPool
- Fully Connected
- Dropout
- Softmax

## Technologies

- Python
- PyTorch
- FastAPI
- OpenCV
- Pandas
- Matplotlib
- Pytest

## Training Configuration

| Parameter      |  Value |
| -------------- | -----: |
| Epochs         |     20 |
| Batch Size     |     64 |
| Learning Rate  |  0.001 |
| Optimizer      |   Adam |
| Scheduler      | StepLR |
| Early Stopping |      3 |

## Results

The project automatically generates:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Loss Curve
- Accuracy Curve
- Grad-CAM Visualization

## Run Training

```bash
python -m src.train
```

## Evaluate

```bash
python -m src.evaluate
```

## Inference

```bash
python -m src.inference
```

## Grad-CAM

```bash
python -m src.gradcam
```

## FastAPI

```bash
uvicorn app.api:app --reload
```

Open:

`http://127.0.0.1:8000/docs`

## Project Status

- Data Pipeline
- CNN Training
- Evaluation
- Error Analysis
- Grad-CAM
- FastAPI Deployment
- Unit Testing

## Author

**Robel Gebregziabher**

AI Engineering & Computer Vision Portfolio

## Quick Start

Install dependencies

```bash
pip install -r requirements.txt
```

Train

```bash
python -m src.train
```

Evaluate

```bash
python -m src.evaluate
```

API

```bash
uvicorn app.api:app --reload
```
