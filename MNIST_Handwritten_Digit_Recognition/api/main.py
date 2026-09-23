from pathlib import Path
from io import BytesIO

import numpy as np
import torch
import torch.nn as nn

from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms

from fastapi import FastAPI, UploadFile, File


# ==========================================
# DEVICE
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# CNN MODEL
# ==========================================

class DigitCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ==========================================
# LOAD MODEL
# ==========================================

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "best_digit_cnn.pth"

model = DigitCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


# ==========================================
# FASTAPI
# ==========================================

app = FastAPI(
    title="MNIST Digit Recognition API",
    version="1.0.0"
)


# ==========================================
# PREPROCESSING
# ==========================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def prepare_image(image: Image.Image):

    image = image.convert("L")
    image = ImageOps.autocontrast(image)

    arr = np.array(image)

    if arr.mean() > 127:
        image = ImageOps.invert(image)

    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    image = image.filter(ImageFilter.GaussianBlur(0.2))

    tensor = transform(image).unsqueeze(0)
    return tensor


# ==========================================
# ROOT ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {
        "message": "MNIST Digit Recognition API is running",
        "model": "DigitCNN"
    }


# ==========================================
# PREDICT ENDPOINT
# ==========================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image = Image.open(BytesIO(await file.read()))

    tensor = prepare_image(image).to(device)

    with torch.inference_mode():

        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)

        predicted_digit = probabilities.argmax(dim=1).item()
        confidence = probabilities.max(dim=1).values.item()

    return {
        "predicted_digit": predicted_digit,
        "confidence": round(confidence * 100, 2)
    }