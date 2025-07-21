# FILE: Modules/Module_2/evaluate_classifier.py
import sys
from pathlib import Path
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Path Setup ---
# Explicitly add the 'scripts' directory to the path to ensure modules are found
SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.append(str(SCRIPTS_ROOT))
BASE = SCRIPTS_ROOT.parents[1]

from Modules.Module_1.dataset import build_classifier_loader
from Modules.Module_2.tinynas_detection_model import TinyNASFeatureExtractor

# --- Default evaluation settings ---
BENCHMARK_DIR = BASE / "1_model_development" / "benchmarks"
EVAL_OUTPUT_DIR = BENCHMARK_DIR / "classifier_evaluation"
DEFAULT_FP32_MODEL_PATH = BASE / "1_model_development" / "models" / "feature_extractor_fp32_best.pth"
DEFAULT_QUANT_MODEL_PATH = BASE / "1_model_development" / "models" / "feature_extractor_quantized.pth"
DEFAULT_IMG_SIZE = 128
DEFAULT_WIDTH_MULT = 1.0
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Must use the identical ClassifierWrapper class to reconstruct the model
class ClassifierWrapper(nn.Module):
    def __init__(self, width_mult, num_classes, img_size=128):
        super().__init__()
        self.feature_extractor = TinyNASFeatureExtractor(width_mult=width_mult)

        # Determine feature dimension dynamically from the backbone
        with torch.no_grad():
            self.feature_extractor.eval()
            dummy_input = torch.randn(1, 3, img_size, img_size)
            # Temporarily disable stubs for this dummy pass
            if hasattr(self.feature_extractor, 'quant'):
                self.feature_extractor.quant.enabled = False
                self.feature_extractor.dequant.enabled = False

            dummy_output = self.feature_extractor(dummy_input)

            if hasattr(self.feature_extractor, 'quant'):
                self.feature_extractor.quant.enabled = True
                self.feature_extractor.dequant.enabled = True

            backbone_out_features = dummy_output.view(dummy_output.size(0), -1).size(1)

        self.projection_head = nn.Sequential(
            nn.Linear(backbone_out_features, 256),
            nn.ReLU()
        )
        self.head = nn.Linear(256, num_classes)

    def forward(self, x, return_features=False):
        # CORRECTED: This defines the correct data flow
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        features = self.projection_head(x)  # Create the 256-dim feature vector

        if return_features:
            return features

        output = self.head(features)  # Pass the feature vector to the final classifier
        return output


def save_sample_images(model, loader, class_names, device, output_dir, num_images=20):
    """Saves a grid of sample images with their predicted and actual labels."""
    model.eval()
    # Get a single batch of images
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Limit to num_images and move to CPU for plotting
    images = images[:num_images].cpu()
    labels = labels[:num_images].cpu()
    preds = preds[:num_images].cpu()

    # Create a plot grid
    rows = (num_images + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows))
    fig.suptitle('Sample Predictions vs. Actuals', fontsize=20)
    axes = axes.flatten()

    for i, (img_tensor, label, pred) in enumerate(zip(images, labels, preds)):
        img = img_tensor.numpy().transpose((1, 2, 0))
        # Denormalize for display: assumes standard ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        ax = axes[i]
        ax.imshow(img)
        pred_name = class_names[pred]
        label_name = class_names[label]
        title_color = "green" if pred == label else "red"
        ax.set_title(f"Pred: {pred_name}\nActual: {label_name}", color=title_color)
        ax.axis('off')

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_dir / 'sample_predictions.png')
    plt.close()
    print(f"Saved sample prediction images to '{output_dir / 'sample_predictions.png'}'")


def evaluate(args):
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"FATAL: Model file not found at {model_path}")
        return

    print("--- Starting Classifier Model Evaluation on TEST SET ---")
    print(f"Loading model state from: {model_path.name}")

    test_dataset, test_loader = build_classifier_loader('test', args.img_size, BATCH_SIZE, num_workers=args.num_workers)
    if not test_loader: return print("Test loader is empty.")

    num_classes = len(test_dataset.classes)
    class_names = test_dataset.classes
    print(f"Dynamically found {num_classes} classes: {class_names}")

    model = ClassifierWrapper(width_mult=args.width_mult, num_classes=num_classes, img_size=args.img_size)

    if args.quantized:
        print("Evaluating QUANTIZED (INT8) model on CPU...")
        eval_device = torch.device("cpu")
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        model.eval()
        # Note: Fusion is not strictly needed for inference but good practice for consistency
        # In a real deployment, you'd load a model already fused.
        torch.quantization.prepare(model, inplace=True)
        torch.quantization.convert(model, inplace=True)
        model.load_state_dict(torch.load(model_path, map_location=eval_device))
    else:
        print("Evaluating standard (FP32) model...")
        eval_device = DEVICE
        model.load_state_dict(torch.load(model_path, map_location=eval_device))

    model.to(eval_device)
    model.eval()

    # --- Save a batch of sample images with predictions ---
    print("\nSaving a sample of visual predictions...")
    save_sample_images(model, test_loader, class_names, eval_device, EVAL_OUTPUT_DIR)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
            imgs, labels = imgs.to(eval_device), labels.to(eval_device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if not all_labels: return print("No labels found in the test set.")

    print("\n--- Evaluation Complete ---")
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Overall Test Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    print(report)
    print("Confusion Matrix:")
    conf_matrix = confusion_matrix(all_labels, all_preds)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
    print(conf_matrix_df)

    # --- Save evaluation artifacts ---
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save classification report to a text file
    (EVAL_OUTPUT_DIR / 'classification_report.txt').write_text(report)

    # Save confusion matrix as a heatmap image
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(EVAL_OUTPUT_DIR / 'confusion_matrix.png')
    plt.close()

    print(f"\nSaved classification report and confusion matrix to '{EVAL_OUTPUT_DIR}'")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Feature Extractor Classifier.")
    parser.add_argument("--model_path", type=str, help="Path to the trained model (.pth) file.")
    parser.add_argument("--img_size", type=int, default=DEFAULT_IMG_SIZE, help="Image size model was trained on.")
    parser.add_argument("--width_mult", type=float, default=DEFAULT_WIDTH_MULT, help="Width multiplier of the model.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--quantized", action='store_true', help="Set this flag to evaluate the final quantized model.")
    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = DEFAULT_QUANT_MODEL_PATH if args.quantized else DEFAULT_FP32_MODEL_PATH

    evaluate(args)


if __name__ == "__main__":
    main()
