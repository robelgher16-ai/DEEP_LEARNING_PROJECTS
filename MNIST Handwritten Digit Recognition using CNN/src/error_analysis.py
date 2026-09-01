import matplotlib.pyplot as plt
import torch

from .config import DEVICE, CHECKPOINT_DIR, OUTPUT_DIR
from .dataset import get_dataloaders
from .model import DigitCNN


def visualize_errors():

    _, _, test_loader = get_dataloaders()

    model = DigitCNN().to(DEVICE)

    model.load_state_dict(
        torch.load(
            CHECKPOINT_DIR / "best_model.pth",
            map_location=DEVICE
        )
    )

    model.eval()

    wrong_images = []
    wrong_true = []
    wrong_pred = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            for i in range(len(labels)):

                if predicted[i].cpu() != labels[i]:

                    wrong_images.append(images[i].cpu())

                    wrong_true.append(labels[i].item())

                    wrong_pred.append(predicted[i].cpu().item())

                if len(wrong_images) == 16:
                    break

            if len(wrong_images) == 16:
                break

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))

    for i, ax in enumerate(axes.flat):

        ax.imshow(wrong_images[i].squeeze(), cmap="gray")

        ax.set_title(
            f"T:{wrong_true[i]}  P:{wrong_pred[i]}",
            fontsize=9
        )

        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "wrong_predictions.png",
        dpi=300
    )

    plt.show()


if __name__ == "__main__":
    visualize_errors()