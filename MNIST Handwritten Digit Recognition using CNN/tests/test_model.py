import torch

from src.model import DigitCNN
from src.config import DEVICE


def test_model_output_shape():

    model = DigitCNN().to(DEVICE)

    model.eval()

    x = torch.randn(4, 1, 28, 28).to(DEVICE)

    with torch.no_grad():

        output = model(x)

    assert output.shape == (4, 10)


def test_model_has_parameters():

    model = DigitCNN()

    parameters = list(model.parameters())

    assert len(parameters) > 0