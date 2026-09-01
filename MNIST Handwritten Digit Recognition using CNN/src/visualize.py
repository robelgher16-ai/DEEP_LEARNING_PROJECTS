import pandas as pd
import matplotlib.pyplot as plt

from .config import OUTPUT_DIR


def plot_learning_curves():

    log_file = OUTPUT_DIR / "training_log.csv"

    df = pd.read_csv(log_file)

    # Loss Curve
    plt.figure(figsize=(8,5))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Validation Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=300)
    plt.close()

    # Accuracy Curve
    plt.figure(figsize=(8,5))
    plt.plot(df["epoch"], df["train_accuracy"], label="Train Accuracy")
    plt.plot(df["epoch"], df["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "accuracy_curve.png", dpi=300)
    plt.close()

    print("Learning curves saved successfully!")