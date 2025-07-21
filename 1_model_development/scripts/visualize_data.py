# visualize_data.py
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
import os

# Add the project root to the Python path to find Modules
BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.append(str(BASE))

from Modules.Module_1.dataset import build_loader, Task

# --- Configuration ---
IMG_SIZE = 256
BATCH_SIZE = 8
DATA_SPLIT = "train"
OUTPUT_DIR = BASE / "output_images"


# --- Main Visualization Logic ---
def visualize():
    """
    Loads a batch of data and saves the images with their bounding box
    annotations to the 'output_images' directory.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"--- Output images will be saved to: {OUTPUT_DIR.resolve()} ---")

    print("Building data loader...")
    loader = build_loader(
        Task.IMDG_DET,
        split=DATA_SPLIT,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=0
    )

    print("Fetching one batch of data...")
    images, targets = next(iter(loader))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    print(f"\nProcessing and saving {len(images)} images...")

    for i in range(images.shape[0]):
        img_tensor = images[i]
        img = img_tensor.cpu().numpy().transpose(1, 2, 0)
        img = std * img + mean
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        target_boxes = targets[i]

        for box in target_boxes:
            class_id, cx, cy, w, h = box
            img_h, img_w, _ = img.shape
            x1 = int((cx - w / 2) * img_w)
            y1 = int((cy - h / 2) * img_h)
            x2 = int((cx + w / 2) * img_w)
            y2 = int((cy + h / 2) * img_h)

            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            label = f"Class: {int(class_id)}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the visualized image to a file
        output_path = OUTPUT_DIR / f"visualization_batch_0_img_{i}.png"
        cv2.imwrite(str(output_path), img)

    print(f"\nSuccessfully saved {len(images)} visualized images to the '{OUTPUT_DIR.name}' directory.")
    print("Please check the folder to confirm your data is loading correctly.")


if __name__ == "__main__":
    visualize()
