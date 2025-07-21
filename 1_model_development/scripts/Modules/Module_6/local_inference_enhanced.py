# Module6/local_inference_enhanced.py
"""local_inference_enhanced.py – Enhanced Local Inference Engine (v2.4)
=====================================================================
An all‑in‑one PyTorch/TinyNAS inference backend that provides:

* **Robust dynamic path discovery** – no more PYTHONPATH errors.
* **Geometry identical to `test_end_to_end.py`** – boxes line up.
* **`using_gpu` flag** – for GUI status bars without attribute errors.
* **Clean API** – `create_inference_engine()` returns a ready instance.

This replaces *all* earlier versions (v2.0–v2.3).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# 🛣️  Dynamic repo path discovery – ensures Module_2 imports work everywhere
# -----------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
MODULES_DIR: Path | None = None
for parent in THIS_FILE.parents:
    if (parent / "Module_2").is_dir():  # sibling folder to Module_6
        MODULES_DIR = parent
        break

if MODULES_DIR is None:
    # Fallback: assume the structure is <repo>/scripts/Modules/Module_6/<this_file>
    # so parent.parent is the Modules directory
    MODULES_DIR = THIS_FILE.parent.parent

# Prepend the Modules directory *and* its parent (repo/scripts) to sys.path
MODULES_PARENT = MODULES_DIR.parent
for p in (MODULES_DIR, MODULES_PARENT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# -----------------------------------------------------------------------------
# 📂  Project root & model paths (relative to repo layout)
# -----------------------------------------------------------------------------
# The expected layout is:
#   <repo_root>/1_model_development/scripts/Modules/Module_6/<this_file>
#   <repo_root>/1_model_development/models/<model_files>
PROJECT_ROOT = MODULES_PARENT.parent  # step out of scripts/Modules

DETECTOR_MODEL_PATH  = PROJECT_ROOT / "1_model_development" / "models" / "container_detector_best.pth"
EXTRACTOR_MODEL_PATH = PROJECT_ROOT / "1_model_development" / "models" / "feature_extractor_fp32_best.pth"
HDC_MODEL_PATH       = PROJECT_ROOT / "1_model_development" / "models" / "module3_pytorch_hdc_model.pth"

# -----------------------------------------------------------------------------
# 🧩  TinyNAS / classifier imports – succeed after path prepending
# -----------------------------------------------------------------------------
# 📂  Project root & model paths (relative to repo layout)
# -----------------------------------------------------------------------------
# MODULES_DIR layout example:
#   <repo_root>/1_model_development/scripts/Modules
# We want PROJECT_ROOT = <repo_root>
PROJECT_ROOT = MODULES_DIR.parent.parent.parent  # step out of scripts/Modules/1_model_development

# Final model paths (single 1_model_development folder, no duplication)
DETECTOR_MODEL_PATH  = PROJECT_ROOT / "1_model_development" / "models" / "container_detector_best.pth"
EXTRACTOR_MODEL_PATH = PROJECT_ROOT / "1_model_development" / "models" / "feature_extractor_fp32_best.pth"
HDC_MODEL_PATH       = PROJECT_ROOT / "1_model_development" / "models" / "module3_pytorch_hdc_model.pth"

# -----------------------------------------------------------------------------
# 🧩  TinyNAS / classifier imports – succeed after path prepending
# -----------------------------------------------------------------------------
try:
    from Module_2.tinynas_detection_model import create_model as create_detector_model  # type: ignore
    from Module_2.train_classifier import ClassifierWrapper  # type: ignore
except ModuleNotFoundError:
    # Fallback for namespace "Modules.Module_2"
    from Modules.Module_2.tinynas_detection_model import create_model as create_detector_model  # type: ignore
    from Modules.Module_2.train_classifier import ClassifierWrapper  # type: ignore

# -----------------------------------------------------------------------------
# 📊  Dataclasses
# -----------------------------------------------------------------------------
@dataclass
class DetectionResult:
    x: int
    y: int
    w: int
    h: int
    box_confidence: float
    damage_type: str
    damage_confidence: float
    is_damaged: bool


# -----------------------------------------------------------------------------
# 🧠  Simple Torch HDC implementation
# -----------------------------------------------------------------------------
class TorchHDCModel(nn.Module):
    def __init__(self, input_dim: int, hd_dim: int, num_classes: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, hd_dim, bias=False)
        self.class_prototypes = nn.Parameter(torch.randn(num_classes, hd_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sign(self.projection(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, num_classes)
        enc = self.encode(x)
        enc = nn.functional.normalize(enc, p=2, dim=1)
        prot = nn.functional.normalize(self.class_prototypes, p=2, dim=1)
        return enc @ prot.t()

    @torch.no_grad()
    def predict_with_confidence(self, feat: np.ndarray | torch.Tensor) -> Tuple[int, float]:
        if isinstance(feat, np.ndarray):
            feat = torch.as_tensor(feat, dtype=torch.float32)
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)
        logits = self(feat.to(next(self.parameters()).device))
        probs = torch.softmax(logits, 1)
        conf, idx = probs.max(1)
        return int(idx.item()), float(conf.item())


# -----------------------------------------------------------------------------
# 🚀  Main inference engine
# -----------------------------------------------------------------------------
class EnhancedInferenceEngine:
    """Bundles detector, feature extractor, and HDC classifier."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.using_gpu = self.device.type == "cuda"  # for GUIs

        # Config
        self.img_size = 128  # must match TinyNAS training
        self.confidence_threshold = 0.30  # matches test_end_to_end.py
        self.iou_threshold = 0.45
        self.damage_classes = ['axis', 'concave', 'dentado', 'no_damage', 'perforation']

        # Models
        self.detector: Optional[nn.Module] = None
        self.extractor: Optional[nn.Module] = None
        self.hdc: Optional[TorchHDCModel] = None

        self._load_models()

    # --------------------------------------------------------------------- models
    def _load_models(self) -> None:
        for p in (DETECTOR_MODEL_PATH, EXTRACTOR_MODEL_PATH, HDC_MODEL_PATH):
            if not p.exists():
                raise FileNotFoundError(f"Model file missing: {p}")

        # Detector --------------------------------------------------
        self.detector = create_detector_model(num_classes=1, width_mult=0.5).to(self.device)
        self.detector.load_state_dict(torch.load(DETECTOR_MODEL_PATH, map_location=self.device), strict=False)
        self.detector.eval()

        # Feature extractor ----------------------------------------
        self.extractor = ClassifierWrapper(width_mult=1.0, num_classes=len(self.damage_classes)).to(self.device)
        self.extractor.load_state_dict(torch.load(EXTRACTOR_MODEL_PATH, map_location=self.device), strict=False)
        self.extractor.eval()

        # HDC classifier ------------------------------------------
        ckpt = torch.load(HDC_MODEL_PATH, map_location=self.device)
        cfg = ckpt['config']
        self.hdc = TorchHDCModel(cfg['input_dim'], cfg['hd_dim'], cfg['num_classes']).to(self.device)
        self.hdc.load_state_dict(ckpt['model_state_dict'], strict=False)
        self.hdc.eval()

    # ----------------------------------------------------------- helpers (private)
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, thresh: float) -> List[int]:
        if len(boxes) == 0:
            return []
        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep: List[int] = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[np.where(iou <= thresh)[0] + 1]
        return keep

    # ----------------------------------------------------------- main public API
    def process_full_image(self, img_rgb: np.ndarray) -> Dict:
        """Run end‑to‑end detection & classification on a full RGB frame."""
        orig_h, orig_w, _ = img_rgb.shape
        t0 = time.time()

        # Stage 1 – detection ------------------------------------------------------------------
        img_resized = cv2.resize(img_rgb, (self.img_size, self.img_size))
        tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().div(255).unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self.detector(tensor).cpu().numpy()[0]  # (S, S, 5)

        # Convert predictions to absolute boxes
        preds[..., :2] = self._sigmoid(preds[..., :2])
        preds[..., 4]   = self._sigmoid(preds[..., 4])
        S = preds.shape[0]

        all_boxes, all_scores = [], []
        for r in range(S):
            for c in range(S):
                conf = preds[r, c, 4]
                if conf < self.confidence_threshold:
                    continue
                x_off, y_off, w_norm, h_norm = preds[r, c, :4]
                x_center_norm = (c + x_off) / S
                y_center_norm = (r + y_off) / S
                x1 = int((x_center_norm - w_norm / 2) * orig_w)
                y1 = int((y_center_norm - h_norm / 2) * orig_h)
                x2 = int((x_center_norm + w_norm / 2) * orig_w)
                y2 = int((y_center_norm + h_norm / 2) * orig_h)
                if x2 <= x1 or y2 <= y1:
                    continue
                all_boxes.append([max(0, x1), max(0, y1), min(orig_w, x2), min(orig_h, y2)])
                all_scores.append(conf)

        if not all_boxes:
            return {'detections': [], 'processing_time': time.time() - t0}

        boxes   = np.array(all_boxes, dtype=np.float32)
        scores  = np.array(all_scores, dtype=np.float32)
        keep    = self._nms(boxes, scores, self.iou_threshold)

        # Stage 2 – damage classification ------------------------------------------------------
        detections: List[DetectionResult] = []
        for idx in keep:
            x1, y1, x2, y2 = boxes[idx].astype(int)
            crop = img_rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_resized = cv2.resize(crop, (self.img_size, self.img_size))
            crop_tensor = torch.from_numpy(crop_resized).permute(2, 0, 1).float().div(255).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.extractor(crop_tensor, return_features=True).cpu().numpy()[0]
            cls_idx, dmg_conf = self.hdc.predict_with_confidence(features)
            dmg_type = self.damage_classes[cls_idx]
            detections.append(
                DetectionResult(
                    x=int(x1), y=int(y1), w=int(x2 - x1), h=int(y2 - y1),
                    box_confidence=float(scores[idx]),
                    damage_type=dmg_type,
                    damage_confidence=dmg_conf,
                    is_damaged=dmg_type != 'no_damage',
                )
            )

        return {
            'detections': detections,
            'processing_time': time.time() - t0,
        }


# -----------------------------------------------------------------------------
# 🔌  Factory helper for external modules (GUI, Streamlit, tests)
# -----------------------------------------------------------------------------

def create_inference_engine() -> EnhancedInferenceEngine:
    """Factory so import sites need not know the class name."""
    return EnhancedInferenceEngine()
