from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

import torch
from torchvision import transforms

from .model import DigitCNN
from .config import DEVICE, CHECKPOINT_DIR

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# -----------------------------
# Global Variables
# -----------------------------
gradients = None
activations = None


# -----------------------------
# Hooks
# -----------------------------
def forward_hook(module, input, output):
    global activations
    activations = output


def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]


# -----------------------------
# Load Model
# -----------------------------
model = DigitCNN().to(DEVICE)

model.load_state_dict(
    torch.load(
        CHECKPOINT_DIR / "best_model.pth",
        map_location=DEVICE
    )
)

model.eval()

# Register Hooks
model.features[3].register_forward_hook(forward_hook)
model.features[3].register_full_backward_hook(backward_hook)
# -----------------------------
# Generate Grad-CAM
# -----------------------------
def generate_gradcam(image_path):

    image = Image.open(image_path)

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    output = model(img_tensor)

    prediction = output.argmax(dim=1)

    model.zero_grad()

    output[0, prediction].backward()

    weights = gradients.mean(dim=[0, 2, 3])

    feature_maps = activations.squeeze()

    for i in range(len(weights)):
        feature_maps[i] *= weights[i]

    heatmap = feature_maps.mean(dim=0).cpu().detach().numpy()

    heatmap = np.maximum(heatmap, 0)

    heatmap = heatmap / heatmap.max()

    heatmap = cv2.resize(heatmap, (28, 28))

    plt.figure(figsize=(6, 6))

    plt.imshow(image, cmap="gray")

    plt.imshow(heatmap, cmap="jet", alpha=0.5)

    plt.title(f"Predicted Digit : {prediction.item()}")

    plt.axis("off")

    plt.show()


if __name__ == "__main__":

    generate_gradcam("sample.png")