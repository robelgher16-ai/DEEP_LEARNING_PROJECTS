import csv
from pathlib import Path

from .config import OUTPUT_DIR


def create_log_file():

    OUTPUT_DIR.mkdir(exist_ok=True)

    log_file = OUTPUT_DIR / "training_log.csv"

    with open(log_file, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "learning_rate"
        ])

    return log_file


def log_epoch(
    log_file,
    epoch,
    train_loss,
    train_acc,
    val_loss,
    val_acc,
    lr
):

    with open(log_file, "a", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            lr
        ])