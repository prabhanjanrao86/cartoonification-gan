import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.discriminator import Discriminator

model = Discriminator()

dummy_input = torch.randn(1, 3, 256, 256)
output = model(dummy_input)

print("Input shape :", dummy_input.shape)
print("Output shape:", output.shape)
