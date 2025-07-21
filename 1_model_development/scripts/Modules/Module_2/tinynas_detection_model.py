# FILE: Modules/Module_2/tinynas_detection_model.py
"""
Unified model script containing architectures for both the single-class
object detector and the multi-class damage patch classifier.
Uses nn.ReLU for full QAT compatibility.
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.append(str(SCRIPTS_ROOT))

#==============================================================================
# SECTION 1: SHARED BUILDING BLOCKS
#==============================================================================

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
        self.relu = nn.ReLU(inplace=True) # This was already correct

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        self.use_res_connect = (stride == 1 and in_ch == out_ch)
        hidden_dim = int(round(in_ch * expand_ratio))
        self.skip_add = nn.quantized.FloatFunctional()
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_ch, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),  # UPDATED: Changed to nn.ReLU
            ])
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),  # UPDATED: Changed to nn.ReLU
            nn.Conv2d(hidden_dim, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)

#==============================================================================
# SECTION 2: ARCHITECTURE FOR THE CONTAINER DETECTOR (MODEL 1)
#==============================================================================

class TinyNASDetectionModel(nn.Module):
    def __init__(self, num_classes: int, width_mult: float = 0.5):
        super().__init__()
        input_ch = int(16 * width_mult)
        last_ch = int(128 * width_mult)
        self.features = nn.Sequential(
            nn.Conv2d(3, input_ch, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_ch),
            nn.ReLU(inplace=True),  # UPDATED: Changed to nn.ReLU
            InvertedResidual(input_ch, int(16 * width_mult), 1, 1),
            InvertedResidual(int(16 * width_mult), int(24 * width_mult), 2, 4),
            InvertedResidual(int(24 * width_mult), int(24 * width_mult), 1, 4),
            InvertedResidual(int(24 * width_mult), int(40 * width_mult), 2, 4),
            InvertedResidual(int(40 * width_mult), int(80 * width_mult), 2, 4),
            InvertedResidual(int(80 * width_mult), int(80 * width_mult), 1, 4),
            InvertedResidual(int(80 * width_mult), int(96 * width_mult), 1, 4),
            nn.Conv2d(int(96 * width_mult), last_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_ch),
            nn.ReLU(inplace=True),  # UPDATED: Changed to nn.ReLU
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

def create_model(num_classes: int, width_mult: float = 0.5) -> TinyNASDetectionModel:
    return TinyNASDetectionModel(num_classes=num_classes, width_mult=width_mult)

#==============================================================================
# SECTION 3: ARCHITECTURE FOR THE CLASSIFIER / FEATURE EXTRACTOR (MODEL 2)
#==============================================================================

class TinyNASFeatureExtractor(nn.Module):
    def __init__(self, width_mult: float = 1.0):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        input_ch = int(16 * width_mult)
        interverted_residual_setting = [
            [1, 16, 1, 1], [4, 24, 2, 2], [4, 40, 3, 2],
            [4, 80, 4, 2], [4, 96, 3, 1], [4, 128, 1, 1],
        ]
        features = [
            nn.Conv2d(3, input_ch, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_ch),
            nn.ReLU(inplace=True) # UPDATED: Changed to nn.ReLU
        ]
        for t, c, n, s in interverted_residual_setting:
            output_ch = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_ch, output_ch, stride, expand_ratio=t))
                input_ch = output_ch
        self.features = nn.Sequential(*features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.features(x)
        x = self.dequant(x)
        return x
