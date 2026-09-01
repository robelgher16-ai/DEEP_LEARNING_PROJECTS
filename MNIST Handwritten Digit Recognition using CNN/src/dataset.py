import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

from .config import DATA_DIR, BATCH_SIZE, VALID_SPLIT, SEED
from .transforms import train_transform, test_transform
def get_dataloaders():

    # Training dataset
    full_train = MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=train_transform
    )

    # Test dataset
    test_dataset = MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=test_transform
    )

    # Train / Validation split
    valid_size = int(len(full_train) * VALID_SPLIT)
    train_size = len(full_train) - valid_size

    generator = torch.Generator().manual_seed(SEED)

    train_dataset, valid_dataset = random_split(
        full_train,
        [train_size, valid_size],
        generator=generator
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, valid_loader, test_loader