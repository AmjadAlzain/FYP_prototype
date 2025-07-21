import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import sys
from pathlib import Path

# Navigate from script_dir -> Module_3 -> Modules -> scripts -> 1_model_development -> project_root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
MODULES_ROOT = Path(__file__).resolve().parents[0]
sys.path.append(str(MODULES_ROOT.parent))

from collections import OrderedDict
from models import build_quantized_feature_extractor
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch
import os

# Configuration
DATA_DIR = PROJECT_ROOT / 'processed_damage_dataset'
MODEL_PATH = PROJECT_ROOT / '1_model_development' / 'models' / 'damage_extractor_qas.pth'
FEATURES_SAVE_DIR = PROJECT_ROOT / '1_model_development' / 'features' / 'specialized'
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
            # Get features
            output = model(inputs)
            # Flatten the features
            output = output.view(output.size(0), -1)
            features.append(output.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            
    features = np.concatenate(features, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)
    return features, labels_list

def main():
    print("Starting specialized feature extraction with localized models...")
    
    FEATURES_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ensured directory exists: {FEATURES_SAVE_DIR}")

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Instantiating model from local definition...")
    # Instantiate the model using the local definition from models.py
    model = build_quantized_feature_extractor(num_classes=NUM_CLASSES)
    
    print(f"Loading state_dict from: {MODEL_PATH}")
    # Load the pre-trained weights into the correctly defined structure
    state_dict = torch.load(MODEL_PATH, map_location=device)
    
    # The saved state_dict has a different structure, so we load with strict=False
    # This is expected as the saved model might be wrapped or have a slightly different setup
    model.load_state_dict(state_dict, strict=False)
    
    # Set the model to evaluation mode and send to device
    feature_extractor = model
    feature_extractor.eval()
    feature_extractor.to(device)
    
    # We only want the features, so we'll use the .features attribute of our model
    feature_extractor = model.features

    for split in ['train', 'val', 'test']:
        print(f"Processing '{split}' split...")
        split_dir = os.path.join(DATA_DIR, split)
        if not os.path.exists(split_dir):
            print(f"Warning: Directory not found for split '{split}'. Skipping.")
            continue
            
        dataset = DamageDataset(data_dir=split_dir, transform=data_transforms)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        features, labels = extract_features(loader, feature_extractor, device)
        
        feature_path = os.path.join(FEATURES_SAVE_DIR, f'{split}_features.npy')
        label_path = os.path.join(FEATURES_SAVE_DIR, f'{split}_labels.npy')
        
        np.save(feature_path, features)
        np.save(label_path, labels)
        print(f"Saved {split} features to {feature_path}")
        print(f"Saved {split} labels to {label_path}")

    print("Specialized feature extraction complete.")

if __name__ == '__main__':
    main()
