# FILE: Module_2_5/fine_tune_extractor.py
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import numpy as np

# --- Path Setup ---
SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.append(str(SCRIPTS_ROOT))
PROJECT_ROOT = SCRIPTS_ROOT.parents[1]

from Modules.Module_1.dataset import build_classifier_loader, NUM_CLASSIFIER_CLASSES
from Modules.Module_2.train_classifier import ClassifierWrapper, train_one_epoch, validate

# --- Configuration ---
BASE_MODEL_PATH = PROJECT_ROOT / "1_model_development" / "models" / "feature_extractor_fp32_best.pth"
FINETUNED_MODEL_PATH = PROJECT_ROOT / "1_model_development" / "models" / "damage_extractor_qas.pth"
IMG_SIZE = 128
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("--- Module 2.5: Fine-tuning Feature Extractor for Damage Classification ---")
    print(f"Using device: {DEVICE}")

    # 1. Load the pre-trained FP32 model
    model = ClassifierWrapper(width_mult=1.0, num_classes=NUM_CLASSIFIER_CLASSES, img_size=IMG_SIZE)
    model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    print(f"Loaded pre-trained model from {BASE_MODEL_PATH.name}")

    # 2. Freeze the feature extractor backbone
    for param in model.feature_extractor.parameters():
        param.requires_grad = False
    print("Froze feature extractor backbone. Only the classification head will be trained.")

    # 3. Prepare the model for QAT with learned scale/zero-point (QAS)
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm', version=1)
    print("Applying QAT/QAS configuration...")
    model.train() # Set to train mode before preparing for QAT
    torch.quantization.prepare_qat(model, inplace=True)

    # 4. Load data
    train_dataset, train_loader = build_classifier_loader('train', IMG_SIZE, BATCH_SIZE, num_workers=0)
    val_dataset, val_loader = build_classifier_loader('val', IMG_SIZE, BATCH_SIZE, num_workers=0)
    if not train_loader or not val_loader:
        print("Could not load data. Exiting.")
        return
        
    class_weights = 1.0 / torch.tensor(np.bincount(train_dataset.targets), dtype=torch.float)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSIFIER_CLASSES
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    # 5. Fine-tune the model
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    best_val_acc = 0.0

    print(f"Starting fine-tuning for {EPOCHS} epochs...")
    for ep in range(EPOCHS):
        model.train()
        model.feature_extractor.eval()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        print(f"Epoch {ep + 1}/{EPOCHS} | Train Loss: {train_loss:.3f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss:.3f}, Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save the QAT model state dict
            torch.save(model.state_dict(), FINETUNED_MODEL_PATH)
            print(f"** New best QAS model saved (Val Acc: {best_val_acc:.2f}%)")

    # 6. Convert and save the final model
    print("\nConverting and saving final model...")
    model.load_state_dict(torch.load(FINETUNED_MODEL_PATH))
    final_model = model.to('cpu')
    final_model.eval()
    torch.quantization.convert(final_model, inplace=True)
    torch.save(final_model.state_dict(), FINETUNED_MODEL_PATH)
    print(f"--- Fine-tuning complete. Final specialized model saved to {FINETUNED_MODEL_PATH.name} ---")

if __name__ == "__main__":
    main()
