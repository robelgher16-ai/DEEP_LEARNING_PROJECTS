# ==========================================
# Fashion MNIST CNN API
# FastAPI Backend
# ==========================================

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import io
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from src.model import FashionCNN

# ==========================================
# APP
# ==========================================

app = FastAPI(
    title="Fashion MNIST CNN API",
    version="1.0.0",
    description="Deep Learning Image Classification API"
)

# ==========================================
# CLASSES
# ==========================================

class_names = [
    "T-shirt/Top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"
]

# ==========================================
# MODEL
# ==========================================

device = torch.device("cpu")

model = FashionCNN()

MODEL_PATH = ROOT_DIR / "models" / "fashion_cnn_best.pth"

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

model.eval()

# ==========================================
# TRANSFORM
# ==========================================

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ==========================================
# HOME
# ==========================================

@app.get("/")
def home():
    return {
        "message": "Fashion MNIST CNN API is running",
        "model": "FashionCNN"
    }

# ==========================================
# PREDICT
# ==========================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = image.resize((28, 28))

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():

        outputs = model(tensor)

        probs = F.softmax(outputs, dim=1)

        confidence, prediction = torch.max(probs, 1)

        top_prob, top_class = torch.topk(probs, 3)

    results = []

    for i in range(3):

        results.append({
            "class": class_names[top_class[0][i]],
            "confidence": round(float(top_prob[0][i]) * 100, 2)
        })

    return JSONResponse({
        "prediction": class_names[prediction.item()],
        "confidence": round(confidence.item() * 100, 2),
        "top3": results
    })