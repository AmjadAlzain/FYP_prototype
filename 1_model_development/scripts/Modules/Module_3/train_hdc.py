"""
Module 3: Consolidated HDC Training with torchhd
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import cv2
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms

# --- Path Setup ---
SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.append(str(SCRIPTS_ROOT))
PROJECT_ROOT = SCRIPTS_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Modules.Module_1.dataset import NUM_CLASSIFIER_CLASSES
from Modules.Module_3.embhd.embhd_py import EmbHDModel, EmbHDVectorType, EmbHDPrototypes

# --- Configuration ---
HDC_DIMENSIONS = 2048
FEATURES_DIR = PROJECT_ROOT / "1_model_development" / "features"
MODELS_DIR = PROJECT_ROOT / "1_model_development" / "models"
BENCHMARKS_DIR = PROJECT_ROOT / "1_model_development" / "benchmarks"
TRAIN_EPOCHS = 50
FINETUNE_EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 0.01
FINETUNE_LR = 0.001

# Container damage classes
DAMAGE_CLASSES = ['axis', 'concave', 'dentado', 'perforation', 'no_damage']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(DAMAGE_CLASSES)}

class TorchHDCModel(nn.Module):
    """PyTorch HDC model"""
    
    def __init__(self, input_dim, hd_dim, num_classes):
        super(TorchHDCModel, self).__init__()
        self.projection = nn.Linear(input_dim, hd_dim, bias=False)
        self.class_prototypes = nn.Parameter(torch.randn(num_classes, hd_dim))
        nn.init.normal_(self.projection.weight, mean=0, std=1/np.sqrt(input_dim))
        nn.init.normal_(self.class_prototypes, mean=0, std=1)
        
    def encode(self, x):
        return torch.sign(self.projection(x))
    
    def forward(self, x):
        encoded = self.encode(x)
        encoded_norm = nn.functional.normalize(encoded, p=2, dim=1)
        prototypes_norm = nn.functional.normalize(self.class_prototypes, p=2, dim=1)
        similarities = torch.mm(encoded_norm, prototypes_norm.t())
        return similarities

def main(args):
    print("=" * 60)
    print("Module 3: Consolidated HDC Training with Fine-tuning")
    print("=" * 60)
    
    model_name = f"module3_torchhdc_model_{HDC_DIMENSIONS}_{args.weight_strategy}.pth"
    HDC_MODEL_PATH = MODELS_DIR / model_name
    BENCHMARK_DIR = BENCHMARKS_DIR / f"hdc_evaluation_{HDC_DIMENSIONS}_{args.weight_strategy}"
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"\n📁 Loading pre-extracted features from {FEATURES_DIR}...")
    
    try:
        train_features = np.load(FEATURES_DIR / "train_features.npy")
        train_labels = np.load(FEATURES_DIR / "train_labels.npy")
        val_features = np.load(FEATURES_DIR / "val_features.npy")
        val_labels = np.load(FEATURES_DIR / "val_labels.npy")
    except FileNotFoundError as e:
        print(f"❌ Error loading features: {e}")
        return
    
    FEATURE_SIZE = train_features.shape[1]
    print(f"✅ Loaded features:")
    print(f"  Feature size: {FEATURE_SIZE}")
    print(f"  Training samples: {len(train_labels)}")
    print(f"  Validation samples: {len(val_labels)}")
    
    X_train = torch.from_numpy(train_features).float()
    y_train = torch.from_numpy(train_labels).long()
    X_val = torch.from_numpy(val_features).float().to(device)
    y_val = torch.from_numpy(val_labels).long().to(device)
    
    sample_weights = None
    if args.weight_strategy == 'balanced':
        print("Calculating class weights ('balanced')...")
        class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        sample_weights = torch.from_numpy(np.array([class_weights[label] for label in train_labels])).float()
    
    if sample_weights is not None:
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    else:
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"\n🧠 Initializing TorchHDCModel ({HDC_DIMENSIONS}D)...")
    model = TorchHDCModel(FEATURE_SIZE, HDC_DIMENSIONS, NUM_CLASSIFIER_CLASSES).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\n🚀 Training for {args.epochs} epochs...")
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        for batch_features, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            _, predicted = torch.max(val_outputs.data, 1)
            val_acc = (predicted == y_val).float().mean().item() * 100
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"** New best model saved (Val Acc: {best_val_acc:.2f}%)")
            torch.save(model.state_dict(), HDC_MODEL_PATH)

        print(f"Epoch {epoch + 1:3d}/{args.epochs} | Val Acc: {val_acc:6.2f}%")

    print(f"\n🔥 Fine-tuning for {FINETUNE_EPOCHS} epochs...")
    optimizer = optim.AdamW(model.parameters(), lr=FINETUNE_LR)
    for epoch in range(FINETUNE_EPOCHS):
        model.train()
        for batch_features, batch_labels in tqdm(train_loader, desc=f"Finetune Epoch {epoch+1}/{FINETUNE_EPOCHS}"):
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            _, predicted = torch.max(val_outputs.data, 1)
            val_acc = (predicted == y_val).float().mean().item() * 100
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"** New best fine-tuned model saved (Val Acc: {best_val_acc:.2f}%)")
            torch.save(model.state_dict(), HDC_MODEL_PATH)

        print(f"Finetune Epoch {epoch + 1:3d}/{FINETUNE_EPOCHS} | Val Acc: {val_acc:6.2f}%")

    print(f"\n✅ Training completed!")
    print(f"🎯 Best validation accuracy: {best_val_acc:.2f}%")
    print(f"📁 Model saved to: {HDC_MODEL_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HDC Model with torchhd")
    parser.add_argument("--weight-strategy", type=str, default="balanced", choices=["none", "balanced"],
                        help="The class weighting strategy to use.")
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS,
                       help="Number of training epochs")
    args = parser.parse_args()
    main(args)
