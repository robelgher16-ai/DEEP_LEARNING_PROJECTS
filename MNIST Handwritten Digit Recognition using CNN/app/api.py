from io import BytesIO

import torch
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from torchvision import transforms

from src.model import DigitCNN
from src.config import DEVICE, CHECKPOINT_DIR

# ---------------------------------------------------
# FastAPI App
# ---------------------------------------------------

app = FastAPI(
    title="MNIST Digit Recognition API",
    version="1.0"
)

# ---------------------------------------------------
# Image Transform
# ---------------------------------------------------

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ---------------------------------------------------
# Load Model (Only Once)
# ---------------------------------------------------

model = DigitCNN().to(DEVICE)

model.load_state_dict(
    torch.load(
        CHECKPOINT_DIR / "best_model.pth",
        map_location=DEVICE
    )
)

model.eval()

# ---------------------------------------------------
# Home Endpoint
# ---------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "MNIST CNN API is running"
    }

# ---------------------------------------------------
# Prediction Endpoint
# ---------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image = Image.open(BytesIO(image_bytes))

    image = transform(image)

    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, prediction = torch.max(probabilities, 1)

    return {
        "predicted_digit": int(prediction.item()),
        "confidence": round(float(confidence.item()), 4)
    }