import os
import cv2
import torch
from torch.utils.data import Dataset

from config import Config


class ImageDataset(Dataset):
    """
    Custom Dataset for loading images from a folder.
    Converts images to normalized tensors in range [-1, 1].
    """

    def __init__(self, folder_path):
        self.cfg = Config()
        self.folder_path = folder_path

        # Filter valid image files
        self.files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if len(self.files) == 0:
            raise ValueError(f"No valid images found in {folder_path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = os.path.join(self.folder_path, self.files[index])

        # Read image
        img = cv2.imread(img_path)

        if img is None:
            # Handle corrupted images safely
            print(f"Skipping unreadable image: {img_path}")
            return self.__getitem__((index + 1) % len(self.files))

        # Resize
        img = cv2.resize(img, (self.cfg.IMAGE_SIZE, self.cfg.IMAGE_SIZE))

        # Convert BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to tensor
        img = torch.from_numpy(img).float()

        # Change shape: HWC → CHW
        img = img.permute(2, 0, 1)

        # Normalize to [-1, 1]
        img = (img / 127.5) - 1.0

        return img