import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.losses import GANLoss, CycleLoss, IdentityLoss

gan_loss = GANLoss()
cycle_loss = CycleLoss()
identity_loss = IdentityLoss()

pred_fake = torch.randn(1, 1, 30, 30)
pred_real = torch.randn(1, 1, 30, 30)

real_img = torch.randn(1, 3, 256, 256)
recon_img = torch.randn(1, 3, 256, 256)

print("GAN loss (fake):", gan_loss(pred_fake, False).item())
print("GAN loss (real):", gan_loss(pred_real, True).item())
print("Cycle loss:", cycle_loss(real_img, recon_img).item())
print("Identity loss:", identity_loss(real_img, recon_img).item())
