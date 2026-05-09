import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """
    Least Squares GAN (LSGAN) Loss.

    Uses MSE instead of BCE for more stable training.
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, prediction, target_is_real):
        """
        Args:
            prediction: Discriminator output
            target_is_real (bool): True for real, False for fake
        """
        target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
        return self.loss(prediction, target)


class CycleLoss(nn.Module):
    """
    Cycle Consistency Loss.

    Ensures that translated images can be reconstructed back
    to the original domain.
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, real, reconstructed):
        return self.loss(reconstructed, real)


class IdentityLoss(nn.Module):
    """
    Identity Loss.

    Encourages the generator to preserve color and structure
    when input image is already in target domain.
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, real, same):
        return self.loss(same, real)