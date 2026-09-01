import torch

def save_checkpoint(
    model,
    optimizer,
    epoch,
    best_acc,
    history,
    filepath
):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_accuracy": best_acc,
        "history": history
    }

    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer):

    checkpoint = torch.load(filepath)

    model.load_state_dict(
        checkpoint["model_state"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state"]
    )

    return (
        checkpoint["epoch"],
        checkpoint["best_accuracy"],
        checkpoint["history"]
    )