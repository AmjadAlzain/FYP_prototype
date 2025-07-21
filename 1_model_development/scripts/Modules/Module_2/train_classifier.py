# FILE: Modules/Module_2/train_classifier.py
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import numpy as np
import copy
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.append(str(SCRIPTS_ROOT))
BASE = PROJECT_ROOT

from Modules.Module_1.dataset import build_classifier_loader, NUM_CLASSIFIER_CLASSES
from Modules.Module_2.tinynas_detection_model import TinyNASFeatureExtractor

# --- Hyperparameters ---
# Phase 1: FP32 Training
LEARNING_RATE = 5e-4
BATCH_SIZE = 64
EPOCHS_FP32 = 50
# Phase 2: QAT Fine-tuning
LR_QAT = 1e-5
EPOCHS_QAT = 20

IMG_SIZE = 128
WIDTH_MULTIPLIER = 1.0

SAVE_DIR_MODELS = BASE / "1_model_development" / "models"
SAVE_DIR_PLOTS = BASE / "1_model_development" / "benchmarks"
BEST_FP32_NAME = "feature_extractor_fp32_best.pth"
FINAL_QUANTIZED_NAME = "feature_extractor_quantized.pth"


class ClassifierWrapper(nn.Module):
    def __init__(self, width_mult, num_classes, img_size=128):
        super().__init__()
        self.feature_extractor = TinyNASFeatureExtractor(width_mult=width_mult)

        # Ensure the feature extractor is on the same device as the wrapper will be
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        with torch.no_grad():
            self.feature_extractor.eval()
            dummy_input = torch.randn(1, 3, img_size, img_size, device=device)
            # Temporarily disable quant/dequant stubs for this dummy pass
            self.feature_extractor.quant.enabled = False
            self.feature_extractor.dequant.enabled = False
            dummy_output = self.feature_extractor(dummy_input)
            self.feature_extractor.quant.enabled = True
            self.feature_extractor.dequant.enabled = True
            backbone_out_features = dummy_output.view(dummy_output.size(0), -1).size(1)

        self.projection_head = nn.Sequential(
            nn.Linear(backbone_out_features, 256),
            nn.ReLU()
        )
        self.head = nn.Linear(256, num_classes)

    def forward(self, x, return_features=False):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        features = self.projection_head(x)
        if return_features:
            return features
        output = self.head(features)
        return output

    def fuse_model(self):
        for m in self.modules():
            if type(m) == nn.Sequential:
                patterns_to_fuse = []
                for i in range(len(m) - 2):
                    if type(m[i]) == nn.Conv2d and type(m[i + 1]) == nn.BatchNorm2d and type(m[i + 2]) == nn.ReLU:
                        patterns_to_fuse.append([str(i), str(i + 1), str(i + 2)])
                if patterns_to_fuse:
                    torch.quantization.fuse_modules(m, patterns_to_fuse, inplace=True)


def train_one_epoch(model, loader, opt, crit, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    loop = tqdm(loader, leave=False, desc="Training")
    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)
        opt.zero_grad()
        outputs = model(imgs)
        loss = crit(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return total_loss / len(loader), 100 * correct / total


@torch.no_grad()
def validate(model, loader, crit, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = crit(outputs, labels)
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return total_loss / len(loader), 100 * correct / total


def plot_curves(history, filename="classifier_curves.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Classifier Training Metrics')
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()
    ax2.plot(history['train_acc'], label='Training Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Accuracy Curves')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    ax2.legend()
    plt.savefig(SAVE_DIR_PLOTS / filename)
    plt.close()
    print(f"Training curves saved to {SAVE_DIR_PLOTS / filename}")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = ClassifierWrapper(
        width_mult=args.width_mult,
        num_classes=NUM_CLASSIFIER_CLASSES,
        img_size=args.img_size
    ).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters.")

    # --- UPDATED: Correctly unpack the two return values ---
    train_dataset, train_loader = build_classifier_loader('train', args.img_size, args.batch_size,
                                                          num_workers=args.num_workers)
    val_dataset, val_loader = build_classifier_loader('val', args.img_size, args.batch_size,
                                                      num_workers=args.num_workers)

    if not train_loader: return print("Training loader is empty.")

    # --- UPDATED: Use the new train_dataset variable ---
    class_counts = np.bincount(train_dataset.targets, minlength=NUM_CLASSIFIER_CLASSES)
    class_weights = 1.0 / torch.tensor(class_counts + 1e-6, dtype=torch.float)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSIFIER_CLASSES
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    print(f"Using class weights: {class_weights.cpu().numpy()}")

    # --- PHASE 1: FP32 Training ---
    print("\n--- Phase 1: Starting FP32 Training ---")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_fp32)
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for ep in range(args.epochs_fp32):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        history['train_loss'].append(train_loss);
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc);
        history['val_acc'].append(val_acc)
        print(
            f"Epoch {ep + 1}/{args.epochs_fp32} | Train Loss: {train_loss:.3f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss:.3f}, Acc: {val_acc:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_DIR_MODELS / BEST_FP32_NAME)
            print(f"** New best FP32 model saved (Val Acc: {best_val_acc:.2f}%)")

    plot_curves(history, "classifier_fp32_curves.png")
    print(f"\n--- FP32 Training complete. Best validation accuracy: {best_val_acc:.2f}% ---")

    # --- PHASE 2: QAT Fine-tuning ---
    print("\n--- Phase 2: Applying Quantization-Aware Training ---")
    model.load_state_dict(torch.load(SAVE_DIR_MODELS / BEST_FP32_NAME))
    model.eval()
    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    model.train()
    torch.quantization.prepare_qat(model, inplace=True)

    optimizer = optim.Adam(model.parameters(), lr=LR_QAT)
    print(f"Fine-tuning with QAT for {args.epochs_qat} epochs...")
    for ep in range(args.epochs_qat):
        _, _ = train_one_epoch(model, train_loader, optimizer, criterion, device)
        _, val_acc = validate(model, val_loader, criterion, device)
        print(f"QAT Epoch {ep + 1}/{args.epochs_qat} | Val Acc: {val_acc:.2f}%")

    model.to('cpu')
    model.eval()
    torch.quantization.convert(model, inplace=True)
    print("Conversion complete.")
    torch.save(model.state_dict(), SAVE_DIR_MODELS / FINAL_QUANTIZED_NAME)
    print(f"** Final quantized model saved to {SAVE_DIR_MODELS / FINAL_QUANTIZED_NAME}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Feature Extractor with Transfer Learning and QAT")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs_fp32", type=int, default=EPOCHS_FP32)
    parser.add_argument("--epochs_qat", type=int, default=EPOCHS_QAT)
    parser.add_argument("--width_mult", type=float, default=WIDTH_MULTIPLIER)
    args = parser.parse_args()
    main(args)
