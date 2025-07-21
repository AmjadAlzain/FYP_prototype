# FILE: Modules/Module_2/train_container_detector.py
"""
Trains a lightweight, SINGLE-CLASS object detector whose only job
is to find the main container in an image.
"""
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.append(str(SCRIPTS_ROOT))

# Importing from the unified dataset.py file
from Modules.Module_1.dataset import DamageDetDataset, detection_collate_fn
from Modules.Module_2.tinynas_detection_model import create_model

# --- Hyperparameters ---
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 20
IMG_SIZE = 128
NUM_CLASSES = 1  # CRITICAL: We are only detecting one class: "container"
WIDTH_MULTIPLIER = 0.5  # A smaller model is sufficient for this simple task

SAVE_DIR_MODELS = PROJECT_ROOT / "1_model_development" / "models"
SAVE_DIR_PLOTS = PROJECT_ROOT / "1_model_development" / "benchmarks"
BEST_MODEL_NAME = "container_detector_best.pth"


# --- A Wrapper for the Dataset to Filter for Only the "Container" Class ---
class ContainerOnlyDataset(DamageDetDataset):
    def __init__(self, split: str, img_size: int):
        super().__init__(split, img_size)
        # The 'container' class has an ID of 0 in the original dataset
        self.container_class_id = 0

    def __getitem__(self, idx):
        img, original_targets = super().__getitem__(idx)
        if original_targets.numel() > 0:
            container_targets = original_targets[original_targets[:, 0] == self.container_class_id]
            if container_targets.numel() > 0:
                container_targets[:, 0] = 0  # Remap class ID to 0 for our single-class model
            return img, container_targets
        return img, torch.empty(0, 5)


def build_detector_loader(split, img_size, batch_size, num_workers):
    """Builds a DataLoader for the container detection task."""
    dataset = ContainerOnlyDataset(split=split, img_size=img_size)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=(split == "train"),
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0), collate_fn=detection_collate_fn
    )


class DetectorLoss(nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.lambda_coord, self.lambda_noobj = lambda_coord, lambda_noobj
        self.box_loss_fn = nn.SmoothL1Loss(reduction='sum')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, predictions, targets):
        device = predictions.device
        batch_size, S, _, _ = predictions.shape
        obj_mask = torch.zeros(batch_size, S, S, dtype=torch.bool, device=device)
        t_box = torch.zeros(batch_size, S, S, 4, device=device)
        for b, target in enumerate(targets):
            if target.numel() > 0:
                gti = (target[:, 1:3] * S).long()
                gt_y, gt_x = torch.clamp(gti[:, 1], 0, S - 1), torch.clamp(gti[:, 0], 0, S - 1)
                obj_mask[b, gt_y, gt_x] = True
                t_box[b, gt_y, gt_x, :2] = target[:, 1:3] * S - gti.float()
                t_box[b, gt_y, gt_x, 2:] = target[:, 3:5]

        noobj_mask = ~obj_mask
        loss_noobj = self.bce_loss(predictions[noobj_mask][..., 4], torch.zeros_like(predictions[noobj_mask][..., 4]))

        if not obj_mask.any():
            return (self.lambda_noobj * loss_noobj) / batch_size

        p_obj = predictions[obj_mask]
        loss_obj = self.bce_loss(p_obj[..., 4], torch.ones_like(p_obj[..., 4]))
        loss_box = self.box_loss_fn(p_obj[..., :4], t_box[obj_mask])

        return (self.lambda_coord * loss_box + loss_obj + self.lambda_noobj * loss_noobj) / batch_size


def train_one_epoch(model, loader, opt, crit, device):
    model.train()
    total_loss = 0.0
    loop = tqdm(loader, leave=False, desc="Training Detector")
    for imgs, targs in loop:
        imgs, targs = imgs.to(device), [t.to(device) for t in targs]
        preds = model(imgs)
        loss = crit(preds, targs)
        if torch.isnan(loss) or torch.isinf(loss): continue
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return total_loss / len(loader) if len(loader) > 0 else 0.0


@torch.no_grad()
def validate(model, loader, crit, device):
    model.eval()
    total_loss = 0.0
    for imgs, targs in loader:
        imgs, targs = imgs.to(device), [t.to(device) for t in targs]
        preds = model(imgs)
        loss = crit(preds, targs)
        if not (torch.isnan(loss) or torch.isinf(loss)):
            total_loss += loss.item()
    return total_loss / len(loader) if len(loader) > 0 else 0.0


def plot_curves(train_losses, val_losses, filename="container_detector_loss_curves.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Detector Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(SAVE_DIR_PLOTS / filename)
    plt.close()
    print(f"Loss curves saved to {SAVE_DIR_PLOTS / filename}")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Training Container Detector (Model 1) ---")
    print(f"Using device: {device}")

    model = create_model(NUM_CLASSES, width_mult=args.width_mult).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters.")
    train_loader = build_detector_loader('train', args.img_size, args.batch_size, num_workers=args.num_workers)
    val_loader = build_detector_loader('val', args.img_size, args.batch_size, num_workers=args.num_workers)
    criterion = DetectorLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)
    best_val_loss = float('inf')

    train_losses, val_losses = [], []
    for ep in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        print(
            f"Epoch {ep + 1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {time.time() - t0:.1f}s")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_DIR_MODELS / BEST_MODEL_NAME)
            print(f"** New best container detector saved (Val Loss: {best_val_loss:.4f})")

    plot_curves(train_losses, val_losses)
    print(f"\n--- Container Detector training complete. Best validation loss: {best_val_loss:.4f} ---")


# This is the corrected argument parser block
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a single-class container detector.")
    parser.add_argument("--width_mult", type=float, default=WIDTH_MULTIPLIER, help="Width multiplier")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--img_size", type=int, default=IMG_SIZE, help="Image size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    args = parser.parse_args()
    main(args)
