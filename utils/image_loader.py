import os
import cv2

def load_images(folder_path):
    images = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)

            if img is None:
                print(f"⚠️ Skipping unreadable file: {file}")
                continue

            img = cv2.resize(img, (256, 256))
            images.append(img)

    return images

if __name__ == "__main__":
    real_images = load_images("data/real")
    cartoon_images = load_images("data/cartoon")

    print(f"Loaded {len(real_images)} real images")
    print(f"Loaded {len(cartoon_images)} cartoon images")
