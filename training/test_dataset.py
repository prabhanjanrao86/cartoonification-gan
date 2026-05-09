import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.dataset import ImageDataset

real_dataset = ImageDataset("data/real")
cartoon_dataset = ImageDataset("data/cartoon")

print("Real images:", len(real_dataset))
print("Cartoon images:", len(cartoon_dataset))

sample = real_dataset[0]
print("Sample shape:", sample.shape)
print("Min value:", sample.min().item())
print("Max value:", sample.max().item())
