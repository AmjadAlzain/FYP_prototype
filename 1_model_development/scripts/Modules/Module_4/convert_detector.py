# FILE: Module4/convert_detector.py
import torch
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare
import os
import shutil
import sys
from pathlib import Path

# --- Add project base to path to allow module imports ---
PROJECT_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.append(str(SCRIPTS_ROOT))

# --- Import only the necessary model definition ---
from Modules.Module_2.tinynas_detection_model import create_model as create_detector_model

# --- Configuration ---
PYTORCH_MODEL_PATH = PROJECT_ROOT / "1_model_development" / "models" / "container_detector_best.pth"
OUTPUT_TFLITE_MODEL = PROJECT_ROOT / "2_firmware_esp32" / "components" / "model_data" / "detector_model.tflite"

# --- Main Conversion Logic ---
print("--- Starting Container Detector Conversion ---")

# 1. Load PyTorch Model
print(f"Loading PyTorch model from {PYTORCH_MODEL_PATH}...")
pytorch_model = create_detector_model(num_classes=1, width_mult=0.5)
pytorch_model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location=torch.device('cpu')))
pytorch_model.eval()

# 2. Export to ONNX
dummy_input = torch.randn(1, 3, 128, 128)
onnx_path = BASE / "detector.onnx"
print("Exporting to ONNX format...")
torch.onnx.export(
    pytorch_model,
    dummy_input,
    str(onnx_path),
    input_names=['input'],
    output_names=['output'],
    opset_version=12
)

# 3. Convert ONNX to TensorFlow SavedModel
print("Converting ONNX to TensorFlow SavedModel...")
onnx_model = onnx.load(str(onnx_path))
tf_rep = prepare(onnx_model)
tf_model_path = "detector_tf"
tf_rep.export_graph(tf_model_path)

# 4. Convert TensorFlow to TFLite
print("Converting TensorFlow SavedModel to TFLite...")
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# 5. Save the TFLite model
OUTPUT_TFLITE_MODEL.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_TFLITE_MODEL, "wb") as f:
    f.write(tflite_model)
print(f"Successfully created TFLite model at {OUTPUT_TFLITE_MODEL}")

# 6. Cleanup
os.remove(onnx_path)
if os.path.exists(tf_model_path):
    shutil.rmtree(tf_model_path)

print("--- Detector Conversion Complete! ---")
