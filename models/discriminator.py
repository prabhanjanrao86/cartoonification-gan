import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator used in CycleGAN.

    Instead of classifying the entire image as real/fake,
    it classifies small patches → improves texture quality.
    """

    def __init__(self, in_channels=3):
        super().__init__()

        self.model = nn.Sequential(
            # Input: (in_channels x 256 x 256)
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # (64 x 128 x 128)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # (128 x 64 x 64)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # (256 x 32 x 32)
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Output: Patch map (not single value)
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)