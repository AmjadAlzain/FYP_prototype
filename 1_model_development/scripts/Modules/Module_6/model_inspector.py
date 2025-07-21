"""
Model Inspector for ESP32-S3-EYE Container Detection
Inspect saved models to understand their structure for local inference
"""

import torch
import numpy as np
from pathlib import Path
import json

def inspect_pytorch_model(model_path: str):
    """Inspect PyTorch model structure"""
    print(f"\n=== Inspecting PyTorch Model: {model_path} ===")
    
    try:
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Check the structure
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        print(f"\nModel structure:")
        for key, tensor in state_dict.items():
            print(f"  {key}: {tensor.shape}")
        
        # Check for metadata
        if 'epoch' in checkpoint:
            print(f"\nTraining Info:")
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'loss' in checkpoint:
            print(f"  Loss: {checkpoint['loss']}")
        if 'accuracy' in checkpoint:
            print(f"  Accuracy: {checkpoint['accuracy']}")
            
        return state_dict
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def inspect_hdc_model(model_path: str):
    """Inspect HDC model structure"""
    print(f"\n=== Inspecting HDC Model: {model_path} ===")
    
    try:
        data = np.load(model_path)
        
        print(f"Available arrays: {list(data.keys())}")
        
        for key in data.keys():
            array = data[key]
            print(f"  {key}: shape={array.shape}, dtype={array.dtype}")
            if len(array.shape) <= 2 and array.size < 20:
                print(f"    values: {array}")
        
        return data
        
    except Exception as e:
        print(f"Error loading HDC model: {e}")
        return None

def inspect_all_models():
    """Inspect all available models"""
    models_dir = Path("../../models")
    
    print("🔍 ESP32-S3-EYE Model Inspector")
    print("=" * 50)
    
    # PyTorch models
    pytorch_models = [
        "feature_extractor_fp32_best.pth",
        "feature_extractor_hdc_ready.pth", 
        "feature_extractor_best.pth"
    ]
    
    for model_name in pytorch_models:
        model_path = models_dir / model_name
        if model_path.exists():
            inspect_pytorch_model(str(model_path))
    
    # HDC models
    hdc_models = [
        "module3_hdc_model_embhd.npz",
        "module3_hdc_model_embhd_2048.npz"
    ]
    
    for model_name in hdc_models:
        model_path = models_dir / model_name
        if model_path.exists():
            inspect_hdc_model(str(model_path))
    
    print("\n" + "=" * 50)
    print("Model inspection complete!")

if __name__ == "__main__":
    inspect_all_models()
