from pathlib import Path

import torch

from src.model import DigitCNN
from src.config import DEVICE, CHECKPOINT_DIR


def test_checkpoint_exists():

    checkpoint = CHECKPOINT_DIR / "best_model.pth"

    assert checkpoint.exists()


def test_checkpoint_loads():

    model = DigitCNN().to(DEVICE)

    checkpoint = CHECKPOINT_DIR / "best_model.pth"

    state_dict = torch.load(
        checkpoint,
        map_location=DEVICE
    )

    model.load_state_dict(state_dict)

    model.eval()