"""
Module 3: Consolidated HDC Evaluation with torchhd
"""

import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

# --- Path Setup ---
SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.append(str(SCRIPTS_ROOT))
PROJECT_ROOT = SCRIPTS_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Modules.Module_3.train_hdc import TorchHDCModel

# --- Configuration ---
HDC_DIMENSIONS = 10000
FEATURES_DIR = PROJECT_ROOT / "1_model_development" / "features"
MODELS_DIR = PROJECT_ROOT / "1_model_development" / "models"
BENCHMARKS_DIR = PROJECT_ROOT / "1_model_development" / "benchmarks"

# Container damage classes
DAMAGE_CLASSES = ['axis', 'concave', 'dentado', 'perforation', 'no_damage']
NUM_CLASSES = len(DAMAGE_CLASSES)

def main(args):
    print("=" * 60)
    print("Module 3: Consolidated HDC Evaluation")
    print("=" * 60)

    # Determine model name and benchmark dir based on weight strategy
    model_name = f"module3_torchhdc_model_{HDC_DIMENSIONS}_{args.weight_strategy}.pth"
    HDC_MODEL_PATH = MODELS_DIR / model_name
    BENCHMARK_DIR = BENCHMARKS_DIR / f"hdc_evaluation_{HDC_DIMENSIONS}_{args.weight_strategy}"
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load pre-extracted features
    print(f"\n📁 Loading pre-extracted features from {FEATURES_DIR}...")
    
    try:
        test_features = np.load(FEATURES_DIR / "test_features.npy")
        test_labels = np.load(FEATURES_DIR / "test_labels.npy")
    except FileNotFoundError as e:
        print(f"❌ Error loading features: {e}")
        print("Please run feature extraction first")
        return
    
    FEATURE_SIZE = test_features.shape[1]
    print(f"✅ Loaded features:")
    print(f"  Feature size: {FEATURE_SIZE}")
    print(f"  Test samples: {len(test_labels)}")
    
    # Convert to tensors
    X_test = torch.from_numpy(test_features).float().to(device)
    y_test = torch.from_numpy(test_labels).long().to(device)
    
    # Load torchhd model
    print(f"\n🧠 Loading TorchHDCModel from {HDC_MODEL_PATH}...")
    model = TorchHDCModel(FEATURE_SIZE, HDC_DIMENSIONS, NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(HDC_MODEL_PATH))
    model.eval()

    # Evaluation
    print("\n📊 Evaluating model...")
    with torch.no_grad():
        outputs = model(X_test)
        _, y_pred = torch.max(outputs.data, 1)
    
    y_pred = y_pred.cpu().numpy()
    y_true = y_test.cpu().numpy()

    # --- Metrics ---
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=DAMAGE_CLASSES, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=DAMAGE_CLASSES))

    print("\n📊 Confusion Matrix:")
    print(cm)

    # --- Save Results ---
    print(f"\n💾 Saving results to {BENCHMARK_DIR}...")
    with open(BENCHMARK_DIR / "classification_report.json", "w") as f:
        json.dump(report, f, indent=4)
    
    cm_df = pd.DataFrame(cm, index=DAMAGE_CLASSES, columns=DAMAGE_CLASSES)
    cm_df.to_csv(BENCHMARK_DIR / "confusion_matrix.csv")

    print("\n✅ Evaluation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HDC Model")
    parser.add_argument("--weight-strategy", type=str, default="balanced", choices=["none", "balanced"],
                        help="The class weighting strategy to use.")
    args = parser.parse_args()
    main(args)
