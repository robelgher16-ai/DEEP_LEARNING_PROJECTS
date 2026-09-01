import matplotlib.pyplot as plt
import torch
from collections import Counter

from .dataset import get_dataloaders


def show_random_samples():
    train_loader, _, _ = get_dataloaders()

    images, labels = next(iter(train_loader))

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    axes = axes.flatten()

    for i in range(10):
        img = images[i].squeeze()
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"Label: {labels[i].item()}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def class_distribution():
    train_loader, _, _ = get_dataloaders()

    all_labels = []

    for _, labels in train_loader:
        all_labels.extend(labels.tolist())

    counts = Counter(all_labels)

    plt.figure(figsize=(8, 4))
    plt.bar(counts.keys(), counts.values())
    plt.title("MNIST Class Distribution")
    plt.xlabel("Digit")
    plt.ylabel("Number of Images")
    plt.xticks(range(10))
    plt.show()


def tensor_information():
    train_loader, _, _ = get_dataloaders()

    images, labels = next(iter(train_loader))

    print("=" * 40)
    print("Tensor Information")
    print("=" * 40)
    print(f"Images Shape : {images.shape}")
    print(f"Labels Shape : {labels.shape}")
    print(f"Image dtype  : {images.dtype}")
    print(f"Min Pixel    : {images.min():.3f}")
    print(f"Max Pixel    : {images.max():.3f}")
    print("=" * 40)


if __name__ == "__main__":
    tensor_information()
    show_random_samples()
    class_distribution()
