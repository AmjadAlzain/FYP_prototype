import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import sys
from pathlib import Path
from collections import OrderedDict

# --- Path Setup ---
PROJECT_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_ROOT = Path(__file__).resolve().parents[2] # This should point to the 'scripts' directory
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.append(str(SCRIPTS_ROOT))

from Modules.Module_2.train_classifier import ClassifierWrapper, WIDTH_MULTIPLIER, IMG_SIZE
from Modules.Module_1.dataset import NUM_CLASSIFIER_CLASSES

# --- Configuration ---
DATA_DIR = PROJECT_ROOT / 'processed_damage_dataset'
MODEL_PATH = PROJECT_ROOT / '1_model_development' / 'models' / 'feature_extractor_fp32_best.pth'
FEATURES_SAVE_DIR = PROJECT_ROOT / '1_model_development' / 'features'
NUM_CLASSES = 5
BATCH_SIZE = 32

class DamageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = str(data_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {cls: i for i, cls in enumerate(sorted(os.listdir(self.data_dir)))}
        
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(self.data_dir, class_name)
            for img_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def extract_features(loader, model, device):
    features = []
    labels_list = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            # Call the model with return_features=True to get the 256-dim vector
            output = model(inputs, return_features=True)
            features.append(output.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            
    features = np.concatenate(features, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)
    return features, labels_list

def main():
    print("--- Starting Feature Extraction using FP32 Model ---")
    
    FEATURES_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ensured directory exists: {FEATURES_SAVE_DIR}")

    data_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Instantiating model using ClassifierWrapper...")
    # Use the correct parameters from the original training script
    model = ClassifierWrapper(width_mult=WIDTH_MULTIPLIER, num_classes=NUM_CLASSIFIER_CLASSES, img_size=IMG_SIZE)
    
    print(f"Loading state_dict from: {MODEL_PATH}")
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    
    # Use the full model to get the correct feature vectors
    model.to(device)
    model.eval()

    for split in ['train', 'val', 'test']:
        print(f"Processing '{split}' split...")
        split_dir = os.path.join(DATA_DIR, split)
        if not os.path.exists(split_dir):
            print(f"Warning: Directory not found for split '{split}'. Skipping.")
            continue
            
        dataset = DamageDataset(data_dir=split_dir, transform=data_transforms)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        # Pass the full model to the extraction function
        features, labels = extract_features(loader, model, device)
        
        feature_path = os.path.join(FEATURES_SAVE_DIR, f'{split}_features.npy')
        label_path = os.path.join(FEATURES_SAVE_DIR, f'{split}_labels.npy')
        
        np.save(feature_path, features)
        np.save(label_path, labels)
        print(f"Saved {split} features to {feature_path}")
        print(f"Saved {split} labels to {label_path}")

    print("--- Feature Extraction Complete ---")

if __name__ == '__main__':
    main()
