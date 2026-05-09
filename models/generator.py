import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual Block used in CycleGAN Generator.
    Helps preserve spatial information using skip connections.
    """

    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)  # Skip connection


class Generator(nn.Module):
    """
    CycleGAN Generator:
    Encoder → Residual Blocks → Decoder
    """

    def __init__(self, in_channels=3, out_channels=3, num_residuals=6):
        super().__init__()

        # ======================
        # Encoder (Downsampling)
        # ======================
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # ======================
        # Residual Blocks
        # ======================
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(num_residuals)]
        )

        # ======================
        # Decoder (Upsampling)
        # ======================
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                128, 64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, out_channels, kernel_size=7, padding=3),
            nn.Tanh()  # Output range [-1, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.res_blocks(x)
        x = self.decoder(x)
        return x