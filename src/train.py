import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from .utils import set_seed
from .logger import create_log_file, log_epoch
from .visualize import plot_learning_curves


from .config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    CHECKPOINT_DIR,
    STEP_SIZE,
    GAMMA,
    PATIENCE,
    SEED
)

from .dataset import get_dataloaders
from .model import DigitCNN
from .engine import train_one_epoch, validate


def main():
    set_seed(SEED)
    log_file = create_log_file()
    # Data
    train_loader, val_loader, test_loader = get_dataloaders()

    # Model
    model = DigitCNN().to(DEVICE)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

    # Learning Rate Scheduler
    scheduler = StepLR(
        optimizer,
        step_size=STEP_SIZE,
        gamma=GAMMA
    )

    # Tracking
    best_accuracy = 0
    patience_counter = 0

    print("=" * 55)
    print("MNIST CNN TRAINING STARTED")
    print("=" * 55)

    # Training Loop
    for epoch in range(EPOCHS):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            DEVICE
        )

        # Current Learning Rate
        current_lr = optimizer.param_groups[0]["lr"]
        
        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | "
            f"LR: {current_lr:.6f}"
        )
        log_epoch(
                    log_file,
                    epoch + 1,
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc,
                    current_lr
                )

        # Save Best Model
        if val_acc > best_accuracy:

            best_accuracy = val_acc
            patience_counter = 0

            torch.save(
                model.state_dict(),
                CHECKPOINT_DIR / "best_model.pth"
            )

            print("Best model saved!")

        else:

            patience_counter += 1

            print(
                f"No improvement ({patience_counter}/{PATIENCE})"
            )

        # Early Stopping
        if patience_counter >= PATIENCE:

            print("=" * 55)
            print("EARLY STOPPING TRIGGERED")
            print("=" * 55)

            break

        # Update Learning Rate
        scheduler.step()


    print("=" * 55)
    print("TRAINING FINISHED")
    print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
    print("=" * 55)
    plot_learning_curves()


if __name__ == "__main__":
    main()