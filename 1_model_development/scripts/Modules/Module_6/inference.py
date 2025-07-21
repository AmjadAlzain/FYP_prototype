# Module_6/inference.py
import sys, torch, time, torchhd
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
if str(BASE) not in sys.path: sys.path.append(str(BASE))

from Modules.Module_1.dataset import build_loader, Task
from Modules.Module_2.detection_model import DetectionModel

# --- Configuration ---
IMG_SIZE, NUM_CLASSES, HDC_DIMS, FEAT_SIZE = 64, 28, 10000, 1280
QUANTIZED_BACKBONE_PATH = BASE / "1_model_development" / "models" / "module4_backbone_quantized.pth"
PROJECTION_PATH = BASE / "1_model_development" / "models" / "module3_hdc_projection.pth"
HDC_MODEL_PATH = BASE / "1_model_development" / "models" / "module3_hdc_model.pth"
DEVICE = torch.device("cpu")

CLASS_ID_TO_NAME = {0: "text", 1: "exp", 2: "exp", 3: "exp", 4: "exp", 5: "fgas", 6: "fgas", 7: "nfgas", 8: "nfgas",
                    9: "tgas", 10: "fliq", 11: "fliq", 12: "fsol", 13: "scom", 14: "dwet", 15: "dwet", 16: "oxy",
                    17: "orgp", 18: "orgp", 19: "tox", 20: "inf", 21: "rad", 22: "rad", 23: "rad", 24: "rad",
                    25: "corr", 26: "misc", 27: "cont"}


def main():
    print("--- Module 6: End-to-End System Inference Test ---")

    # Load Models
    backbone = DetectionModel(num_classes=NUM_CLASSES)
    quantized_backbone = torch.quantization.quantize_dynamic(backbone, {torch.nn.Conv2d}, dtype=torch.qint8)
    quantized_backbone.load_state_dict(torch.load(QUANTIZED_BACKBONE_PATH, map_location=DEVICE))
    quantized_backbone.eval()

    # CORRECTED: The torch-hd library uses the string 'BSC' for bipolar vectors.
    hdc_projection = torchhd.embeddings.Random(FEAT_SIZE, HDC_DIMS, vsa="BSC")
    hdc_projection.load_state_dict(torch.load(PROJECTION_PATH, map_location=DEVICE))
    hdc_projection.eval()

    hdc_model = torchhd.models.Centroid(HDC_DIMS, NUM_CLASSES)
    hdc_model.load_state_dict(torch.load(HDC_MODEL_PATH, map_location=DEVICE))
    hdc_model.eval()

    # Load Data
    val_loader = build_loader(Task.IMDG_DET, "val", IMG_SIZE, 1, 0)
    image, target = next(iter(val_loader))
    true_class_id = int(target[0][0, 0].item()) if len(target[0]) > 0 else -1

    # Inference
    start_time = time.time()
    with torch.no_grad():
        features = quantized_backbone.extract_features(image.to(DEVICE))
        encoded = hdc_projection(features)
        scores = hdc_model(encoded)
        prediction = torch.argmax(scores, dim=-1).item()

    print("\n--- Inference Complete! ---")
    print(f"True Label: {CLASS_ID_TO_NAME.get(true_class_id, 'N/A')}")
    print(f"Predicted Label: {CLASS_ID_TO_NAME.get(prediction, 'N/A')}")
    print(f"Inference Time: {(time.time() - start_time) * 1000:.2f} ms")


if __name__ == "__main__": main()
