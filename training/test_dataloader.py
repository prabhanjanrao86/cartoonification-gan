import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch.utils.data import DataLoader
from utils.dataset import ImageDataset

real_dataset = ImageDataset("data/real")
cartoon_dataset = ImageDataset("data/cartoon")

real_loader = DataLoader(real_dataset, batch_size=2, shuffle=True)
cartoon_loader = DataLoader(cartoon_dataset, batch_size=2, shuffle=True)

real_batch = next(iter(real_loader))
cartoon_batch = next(iter(cartoon_loader))

print("Real batch shape:", real_batch.shape)
print("Cartoon batch shape:", cartoon_batch.shape)
