import torch

from src.model import DigitCNN

model = DigitCNN()

x = torch.randn(64, 1, 28, 28)

output = model(x)

print("Input Shape :", x.shape)
print("Output Shape:", output.shape)