# ==============================
# Imports & Path Setup
# ==============================
import os
import sys
import torch
from torch.utils.data import DataLoader

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# Local imports
from config import Config
from utils.dataset import ImageDataset
from models.generator import Generator
from models.discriminator import Discriminator
from utils.losses import GANLoss, CycleLoss, IdentityLoss


# ==============================
# Setup
# ==============================
def setup():
    cfg = Config()

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")

    return cfg, device


# ==============================
# Data Loaders
# ==============================
def get_dataloaders(cfg):
    real_dataset = ImageDataset(cfg.TRAIN_A)
    cartoon_dataset = ImageDataset(cfg.TRAIN_B)

    real_loader = DataLoader(
        real_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True
    )

    cartoon_loader = DataLoader(
        cartoon_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True
    )

    return real_loader, cartoon_loader


# ==============================
# Model Initialization
# ==============================
def initialize_models(device):
    G_real2cartoon = Generator().to(device)
    G_cartoon2real = Generator().to(device)

    D_real = Discriminator().to(device)
    D_cartoon = Discriminator().to(device)

    return G_real2cartoon, G_cartoon2real, D_real, D_cartoon


# ==============================
# Loss Functions
# ==============================
def get_losses():
    return GANLoss(), CycleLoss(), IdentityLoss()


# ==============================
# Optimizers
# ==============================
def get_optimizers(cfg, G_r2c, G_c2r, D_real, D_cartoon):
    optimizer_G = torch.optim.Adam(
        list(G_r2c.parameters()) + list(G_c2r.parameters()),
        lr=cfg.LR,
        betas=(0.5, 0.999)
    )

    optimizer_D_real = torch.optim.Adam(
        D_real.parameters(),
        lr=cfg.LR,
        betas=(0.5, 0.999)
    )

    optimizer_D_cartoon = torch.optim.Adam(
        D_cartoon.parameters(),
        lr=cfg.LR,
        betas=(0.5, 0.999)
    )

    return optimizer_G, optimizer_D_real, optimizer_D_cartoon


# ==============================
# Training Loop
# ==============================
def train():
    cfg, device = setup()

    real_loader, cartoon_loader = get_dataloaders(cfg)

    G_r2c, G_c2r, D_real, D_cartoon = initialize_models(device)

    gan_loss, cycle_loss, identity_loss = get_losses()

    optimizer_G, optimizer_D_real, optimizer_D_cartoon = get_optimizers(
        cfg, G_r2c, G_c2r, D_real, D_cartoon
    )

    print("🚀 Training started...")

    for epoch in range(cfg.EPOCHS):

        for real_img, cartoon_img in zip(real_loader, cartoon_loader):

            real_img = real_img.to(device)
            cartoon_img = cartoon_img.to(device)

            # ======================
            # Train Generators
            # ======================
            optimizer_G.zero_grad()

            fake_cartoon = G_r2c(real_img)
            reconstructed_real = G_c2r(fake_cartoon)

            fake_real = G_c2r(cartoon_img)
            reconstructed_cartoon = G_r2c(fake_real)

            loss_gan = (
                gan_loss(D_cartoon(fake_cartoon), True) +
                gan_loss(D_real(fake_real), True)
            )

            loss_cycle_total = (
                cycle_loss(real_img, reconstructed_real) +
                cycle_loss(cartoon_img, reconstructed_cartoon)
            )

            loss_identity_total = (
                identity_loss(real_img, G_c2r(real_img)) +
                identity_loss(cartoon_img, G_r2c(cartoon_img))
            )

            loss_G = (
                loss_gan +
                cfg.LAMBDA_CYCLE * loss_cycle_total +
                cfg.LAMBDA_IDENTITY * loss_identity_total
            )

            loss_G.backward()
            optimizer_G.step()

            # ======================
            # Train Discriminator (Real)
            # ======================
            optimizer_D_real.zero_grad()

            loss_D_real = (
                gan_loss(D_real(real_img), True) +
                gan_loss(D_real(fake_real.detach()), False)
            )

            loss_D_real.backward()
            optimizer_D_real.step()

            # ======================
            # Train Discriminator (Cartoon)
            # ======================
            optimizer_D_cartoon.zero_grad()

            loss_D_cartoon = (
                gan_loss(D_cartoon(cartoon_img), True) +
                gan_loss(D_cartoon(fake_cartoon.detach()), False)
            )

            loss_D_cartoon.backward()
            optimizer_D_cartoon.step()

        # ======================
        # Logging
        # ======================
        print(
            f"[Epoch {epoch+1}/{cfg.EPOCHS}] "
            f"G: {loss_G.item():.4f} | "
            f"D_real: {loss_D_real.item():.4f} | "
            f"D_cartoon: {loss_D_cartoon.item():.4f}"
        )

    # ==============================
    # Save Model
    # ==============================
    os.makedirs("outputs/checkpoints", exist_ok=True)

    torch.save(
        G_r2c.state_dict(),
        "outputs/checkpoints/G_real2cartoon.pth"
    )

    print("✅ Model saved successfully!")


# ==============================
# Entry Point
# ==============================
if __name__ == "__main__":
    train()