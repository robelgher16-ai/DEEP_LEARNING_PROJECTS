from src.dataset import get_dataloaders


def test_dataloader():

    train_loader, val_loader, test_loader = get_dataloaders()

    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0


def test_batch_shape():

    train_loader, _, _ = get_dataloaders()

    images, labels = next(iter(train_loader))

    assert images.ndim == 4
    assert labels.ndim == 1

    assert images.shape[0] == labels.shape[0]