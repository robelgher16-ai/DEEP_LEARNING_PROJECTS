from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .config import DEVICE, CHECKPOINT_DIR
from .model import DigitCNN


transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def predict(image_path):

    model = DigitCNN().to(DEVICE)

    model.load_state_dict(
        torch.load(
            CHECKPOINT_DIR / "best_model.pth",
            map_location=DEVICE
        )
    )

    model.eval()

    image = Image.open(image_path)

    image = transform(image)

    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, prediction = torch.max(probabilities, 1)

    return prediction.item(), confidence.item()


if __name__ == "__main__":

    image = "sample.png"

    digit, confidence = predict(image)

    print("=" * 40)
    print(f"Predicted Digit : {digit}")
    print(f"Confidence      : {confidence:.2%}")
    print("=" * 40)