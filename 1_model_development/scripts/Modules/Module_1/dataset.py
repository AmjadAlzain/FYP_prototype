# FILE: Modules/Module_1/dataset.py
"""
Unified dataset script to handle loading for both the object detector (Model 1)
and the patch classifier (Model 2).
"""
import sys
from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np

import albumentations as A
try:
    from albumentations import (
        Compose, HorizontalFlip, Resize, PadIfNeeded, Normalize, BboxParams,
        RandomBrightnessContrast, Affine
    )
    from albumentations.pytorch import ToTensorV2
except ImportError:
    raise ImportError("pip install -U albumentations")

BASE = Path(__file__).resolve().parents[3]
if str(BASE) not in sys.path:
    sys.path.append(str(BASE))

# ==============================================================================
# SECTION 1: LOGIC FOR THE PATCH CLASSIFIER (MODEL 2)
# ==============================================================================
# For the Classifier (Model 2)
CLASSIFIER_CLASS_NAMES = ['axis', 'concave', 'dentado', 'perforation', 'no_damage']
NUM_CLASSIFIER_CLASSES = len(CLASSIFIER_CLASS_NAMES)

# For the Detector (Model 1)
DETECTOR_CLASS_NAMES = ['container']
NUM_DETECTOR_CLASSES = len(DETECTOR_CLASS_NAMES)

def build_classifier_loader(split: str, img_size: int, batch_size: int, num_workers: int = 4):
    """Builds a DataLoader for the patch classification task using the preprocessed dataset."""
    data_root = BASE.parent / "processed_damage_dataset"
    split_dir = data_root / split

    if not split_dir.exists():
        raise FileNotFoundError(f"Processed dataset not found at {split_dir}. Please run preprocess_damage_data.py first.")

    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    dataset = datasets.ImageFolder(root=split_dir, transform=transform)
    print(
        f"  [Classifier Loader] Found {len(dataset)} images in {len(dataset.classes)} classes for the '{split}' split.")
    if len(dataset) == 0: return None, None

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=(split == "train"),
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )

    return dataset, loader


# ==============================================================================
# SECTION 2: LOGIC FOR THE CONTAINER DETECTOR (MODEL 1)
# ==============================================================================
class _BaseDetDataset(Dataset):
    """Base class for detection datasets using YOLO-style .txt labels."""

    def __init__(self, split: str, img_size: int, img_path_template: str, label_path_template: str):
        d = BASE.parent # This now correctly points to the project root
        img_dir = d / img_path_template.format(split=split)
        label_dir = d / label_path_template.format(split=split)

        print(f"  [Detector Loader] Searching for images in: {img_dir}")
        self.images = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
        print(f"  [Detector Loader] Found {len(self.images)} images for the '{split}' split.")

        if not self.images:
            raise FileNotFoundError(f"Dataset error: No images found in {img_dir}")

        self.label_files = [label_dir / f'{p.stem}.txt' for p in self.images]
        self.aug = self._build_augment(img_size, train=(split == "train"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        p = self.images[idx]
        im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        lines = self.label_files[idx].read_text().strip().splitlines() if self.label_files[idx].exists() else []
        boxes, labels = [], []
        for L in lines:
            try:
                parts = list(map(float, L.strip().split()))
                if len(parts) >= 5:
                    labels.append(int(parts[0]))
                    boxes.append(parts[1:5])
            except (ValueError, IndexError):
                continue  # Skip malformed lines
        try:
            augmented = self.aug(image=im, bboxes=boxes, class_labels=labels)
            im, boxes, labels = augmented['image'], augmented['bboxes'], augmented['class_labels']
        except ValueError:
            return self.__getitem__((idx + 1) % len(self))
        targets = [[label] + list(box) for label, box in zip(labels, boxes)]
        return im, torch.tensor(targets, dtype=torch.float32)

    def _build_augment(self, size: int, train: bool):
        base_transforms = [
            Resize(height=size, width=size),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]
        if train:
            return Compose([
                HorizontalFlip(p=0.5),
                RandomBrightnessContrast(p=0.7),
                Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), rotate=(-10, 10), p=0.7),
                A.GaussianBlur(p=0.2),
                A.GaussNoise(p=0.2),
                *base_transforms
            ], bbox_params=BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.1))
        else:
            return Compose(base_transforms, bbox_params=BboxParams(format='yolo', label_fields=['class_labels']))


class DamageDetDataset(_BaseDetDataset):
    """The original dataset loader for damage detection, reading .txt files from bbannotation."""

    def __init__(self, split: str, img_size: int):
        # This now points to the processed YOLO-style annotations
        super().__init__(split, img_size, 'processed_detection_dataset/images/{split}', 'processed_detection_dataset/labels/{split}')


def detection_collate_fn(batch):
    """Correctly collates batches for object detection."""
    imgs, targets = zip(*batch)
    return torch.stack(imgs, 0), list(targets)
