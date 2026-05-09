import os
import sys
import cv2
import torch
import numpy as np

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from models.generator import Generator

# ========================
# Paths
# ========================
INPUT_DIR = "data/test_real"
OUTPUT_DIR = "outputs/cartoon_results"
MODEL_PATH = "outputs/checkpoints/G_real2cartoon.pth"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================
# Device
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# Load Model
# ========================
model = Generator().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Disable gradients (extra safety)
for param in model.parameters():
    param.requires_grad = False

# ========================
# Preprocess
# ========================
def preprocess(img):
    # Center crop (face focus)
    h, w, _ = img.shape
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    img = img[start_y:start_y + min_dim, start_x:start_x + min_dim]

    # Resize
    img = cv2.resize(img, (256, 256))

    # Convert color
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize to [-1, 1]
    img = torch.from_numpy(img).float()
    img = img.permute(2, 0, 1)
    img = (img / 127.5) - 1.0

    return img.unsqueeze(0)

# ========================
# Postprocess
# ========================
def postprocess(tensor):
    img = tensor.squeeze(0).cpu().detach()
    img = img.permute(1, 2, 0).numpy()

    # Convert back to [0, 255]
    img = (img + 1) * 127.5
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Convert to BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Smooth (cartoon effect)
    img = cv2.bilateralFilter(img, 9, 75, 75)

    # Sharpen (restore edges)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)

    return img

# ========================
# Inference Loop
# ========================
for file in os.listdir(INPUT_DIR):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):

        img_path = os.path.join(INPUT_DIR, file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ Skipping invalid image: {file}")
            continue

        # Preprocess
        input_tensor = preprocess(img).to(device)

        # Generate cartoon
        with torch.no_grad():
            output_tensor = model(input_tensor)

        # Postprocess
        cartoon_img = postprocess(output_tensor)

        # Save
        output_path = os.path.join(OUTPUT_DIR, file)
        cv2.imwrite(output_path, cartoon_img)

        print(f"Cartoon generated: {output_path}")

print("All images processed successfully!")