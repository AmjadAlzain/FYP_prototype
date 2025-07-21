# FILE: Module4/run_conversion.py
# A single, self-contained script to convert both models.
# This script has no external project dependencies to avoid import conflicts.

import torch
import torch.nn as nn
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare
import os
import shutil
import sys
import numpy as np
from pathlib import Path

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.append(str(SCRIPTS_ROOT))
# ==============================================================================
# SECTION 1: COPIED CLASS DEFINITIONS
# We copy the class definitions here to avoid importing conflicting files.
# ==============================================================================

# --- From: tinynas_detection_model.py ---
class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        self.use_res_connect = (stride == 1 and in_ch == out_ch)
        hidden_dim = int(round(in_ch * expand_ratio))
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_ch, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            ])
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class TinyNASFeatureExtractor(nn.Module):
    def __init__(self, width_mult: float = 1.0):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        input_ch = int(16 * width_mult)
        interverted_residual_setting = [
            [1, 16, 1, 1], [4, 24, 2, 2], [4, 40, 3, 2],
            [4, 80, 4, 2], [4, 96, 3, 1], [4, 128, 1, 1],
        ]
        features = [
            nn.Conv2d(3, input_ch, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_ch),
            nn.ReLU(inplace=True)
        ]
        for t, c, n, s in interverted_residual_setting:
            output_ch = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_ch, output_ch, stride, expand_ratio=t))
                input_ch = output_ch
        self.features = nn.Sequential(*features)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.features(x)
        x = self.dequant(x)
        return x

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class TinyNASDetectionModel(nn.Module):
    def __init__(self, num_classes: int, width_mult: float = 0.5):
        super().__init__()
        input_ch = int(16 * width_mult)
        last_ch = int(128 * width_mult)
        self.features = nn.Sequential(
            nn.Conv2d(3, input_ch, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_ch),
            nn.ReLU(inplace=True),
            InvertedResidual(input_ch, int(16 * width_mult), 1, 1),
            InvertedResidual(int(16 * width_mult), int(24 * width_mult), 2, 4),
            InvertedResidual(int(24 * width_mult), int(24 * width_mult), 1, 4),
            InvertedResidual(int(24 * width_mult), int(40 * width_mult), 2, 4),
            InvertedResidual(int(40 * width_mult), int(80 * width_mult), 2, 4),
            InvertedResidual(int(80 * width_mult), int(80 * width_mult), 1, 4),
            InvertedResidual(int(80 * width_mult), int(96 * width_mult), 1, 4),
            nn.Conv2d(int(96 * width_mult), last_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_ch),
            nn.ReLU(inplace=True),
        )
        out_ch = 5 + num_classes
        self.detection_head = nn.Sequential(
            DepthwiseSeparableConv(last_ch, last_ch, 3, 1, 1),
            nn.Conv2d(last_ch, out_ch, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        pred = self.detection_head(feat)
        return pred.permute(0, 2, 3, 1)

# --- From: train_classifier.py ---
class ClassifierWrapper(nn.Module):
    def __init__(self, width_mult, num_classes, img_size=128):
        super().__init__()
        self.feature_extractor = TinyNASFeatureExtractor(width_mult=width_mult)
        with torch.no_grad():
            self.feature_extractor.eval()
            dummy_input = torch.randn(1, 3, img_size, img_size)
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

class FeatureExtractorOnly(torch.nn.Module):
    def __init__(self, full_model):
        super().__init__()
        self.model = full_model
    def forward(self, x):
        return self.model(x, return_features=True)

# ==============================================================================
# SECTION 2: MAIN CONVERSION LOGIC
# ==============================================================================
def convert(model_name, pytorch_model, dummy_input, output_path, is_quantized=False):
    print(f"\n--- Starting Conversion for: {model_name} ---")
    pytorch_model.eval()

    onnx_path = PROJECT_ROOT / f"{model_name}.onnx"
    print("1. Exporting to ONNX format...")
    torch.onnx.export(
        pytorch_model, dummy_input, str(onnx_path),
        input_names=['input'], output_names=['output'], opset_version=12
    )

    print("2. Converting ONNX to TensorFlow SavedModel...")
    onnx_model = onnx.load(str(onnx_path))
    tf_rep = prepare(onnx_model)
    tf_model_path = f"{model_name}_tf"
    tf_rep.export_graph(tf_model_path)

    print("3. Converting TensorFlow SavedModel to TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if is_quantized:
        print("   Applying INT8 quantization...")
        def representative_dataset_gen():
            for _ in range(100):
                yield [np.random.rand(*dummy_input.shape).astype(np.float32)]
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"Successfully created TFLite model at {output_path}")
    os.remove(onnx_path)
    if os.path.exists(tf_model_path):
        shutil.rmtree(tf_model_path)
    print(f"--- {model_name} Conversion Complete! ---")

if __name__ == "__main__":
    # --- Convert Detector ---
    detector_model = TinyNASDetectionModel(num_classes=1, width_mult=0.5)
    detector_model.load_state_dict(torch.load(PROJECT_ROOT / "1_model_development" / "models" / "container_detector_best.pth", map_location=torch.device('cpu')))
    convert(
        model_name="detector",
        pytorch_model=detector_model,
        dummy_input=torch.randn(1, 3, 128, 128),
        output_path=PROJECT_ROOT / "2_firmware_esp32" / "components" / "model_data" / "detector_model.tflite",
        is_quantized=False
    )

    # --- Convert Feature Extractor (using FP32 model with Post-Training Quantization) ---
    print("\nLoading FP32 feature extractor and applying Post-Training Quantization...")
    # 1. Create a model instance and load the FP32 weights
    fp32_model = ClassifierWrapper(width_mult=1.0, num_classes=5, img_size=128)
    fp32_model.load_state_dict(torch.load(PROJECT_ROOT / "1_model_development" / "models" / "feature_extractor_fp32_best.pth", map_location=torch.device('cpu')))
    
    # 2. Wrap the model to only export the feature extraction part
    feature_extractor = FeatureExtractorOnly(fp32_model)

    # 3. Convert to TFLite with INT8 quantization
    convert(
        model_name="feature_extractor",
        pytorch_model=feature_extractor,
        dummy_input=torch.randn(1, 3, 128, 128),
        output_path=PROJECT_ROOT / "2_firmware_esp32" / "components" / "model_data" / "feature_extractor_model.tflite",
        is_quantized=True
    )
