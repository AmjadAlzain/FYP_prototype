# FILE: Module4/finalize_and_convert.py
# A dedicated script to correctly load the QAT-trained feature extractor
# and convert it to a final, deployable TFLite model.

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
def convert_to_tflite(model_name, pytorch_model, dummy_input, output_path):
    print(f"\n--- Starting TFLite Conversion for: {model_name} ---")
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
    # --- Definitive Feature Extractor Conversion ---
    print("\nLoading and converting feature extractor...")
    
    # 1. Create a model instance and prepare it for QAT
    model = ClassifierWrapper(width_mult=1.0, num_classes=5, img_size=128)
    model.train()
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model, inplace=True)
    
    # 2. Load the state dict from the QAT-trained model
    qat_state_dict = torch.load(PROJECT_ROOT / "1_model_development" / "models" / "feature_extractor_quantized.pth", map_location=torch.device('cpu'))
    model.load_state_dict(qat_state_dict)
    
    # 3. Convert the QAT model to a finalized quantized model for inference
    model.to('cpu')
    model.eval()
    quantized_model = torch.quantization.convert(model, inplace=False)
    
    # 4. Wrap the model to only export the feature extraction part
    feature_extractor = FeatureExtractorOnly(quantized_model)

    # 5. Convert to TFLite
    convert_to_tflite(
        model_name="feature_extractor",
        pytorch_model=feature_extractor,
        dummy_input=torch.randn(1, 3, 128, 128),
        output_path=PROJECT_ROOT / "storage" / "feature_extractor_model.tflite"
    )
