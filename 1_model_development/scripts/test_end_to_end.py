"""
End-to-End Test with PyTorch HDC Model
Complete pipeline test: Container Detection → Feature Extraction → Damage Classification
Uses the new PyTorch HDC model with 98.35% accuracy
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import json
import time

# --- Path Setup ---
SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.append(str(SCRIPTS_ROOT))
PROJECT_ROOT = SCRIPTS_ROOT.parents[0]

# --- Imports ---
from Modules.Module_1.dataset import DamageDetDataset
from Modules.Module_2.tinynas_detection_model import create_model as create_detector_model
from Modules.Module_2.train_classifier import ClassifierWrapper

# --- Configuration ---
DETECTOR_MODEL_PATH = PROJECT_ROOT / "1_model_development" / "models" / "container_detector_best.pth"
EXTRACTOR_MODEL_PATH = PROJECT_ROOT / "1_model_development" / "models" / "feature_extractor_fp32_best.pth"
HDC_MODEL_PATH = PROJECT_ROOT / "1_model_development" / "models" / "module3_pytorch_hdc_model.pth"
OUTPUT_DIR = PROJECT_ROOT / "output_images_final_pipeline"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.3

# Container damage classes
DAMAGE_CLASSES = ['axis', 'concave', 'dentado', 'no_damage' , 'perforation']
NUM_CLASSES = len(DAMAGE_CLASSES)
DAMAGE_COLORS = {
    'axis': (255, 100, 100),        # Light Blue
    'concave': (100, 255, 255),     # Light Cyan
    'dentado': (255, 100, 255),     # Light Magenta
    'no_damage': (100, 255, 100),   # Light Green
    'perforation': (255, 255, 100)  # Light Yellow
       
}

class TorchHDCModel(nn.Module):
    """PyTorch HDC model for end-to-end inference"""
    
    def __init__(self, input_dim, hd_dim, num_classes):
        super(TorchHDCModel, self).__init__()
        self.projection = nn.Linear(input_dim, hd_dim, bias=False)
        self.class_prototypes = nn.Parameter(torch.randn(num_classes, hd_dim))
        
    def encode(self, x):
        """Encode features to hyperdimensional space"""
        hd_features = self.projection(x)
        return torch.sign(hd_features)
    
    def forward(self, x):
        """Forward pass"""
        encoded = self.encode(x)
        encoded_norm = nn.functional.normalize(encoded, p=2, dim=1)
        prototypes_norm = nn.functional.normalize(self.class_prototypes, p=2, dim=1)
        similarities = torch.mm(encoded_norm, prototypes_norm.t())
        return similarities
    
    def predict_with_confidence(self, features):
        """Make prediction with confidence scores"""
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        
        features = features.to(next(self.parameters()).device)
        
        with torch.no_grad():
            outputs = self.forward(features)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            confidences = torch.max(probabilities, dim=1)[0]
            
        return predictions.cpu().numpy()[0], confidences.cpu().numpy()[0], probabilities.cpu().numpy()[0]

def load_end_to_end_models():
    """Load all models for end-to-end testing"""
    print("=" * 70)
    print("LOADING END-TO-END PIPELINE MODELS")
    print("=" * 70)
    
    start_time = time.time()
    
    # Load container detector
    print("🚢 Loading container detector...")
    detector = create_detector_model(num_classes=1, width_mult=0.5).to(DEVICE)
    detector.load_state_dict(torch.load(DETECTOR_MODEL_PATH, map_location=DEVICE))
    detector.eval()
    print(f"   ✅ Container detector loaded")

    # Load feature extractor
    print("🔍 Loading TinyNAS feature extractor...")
    feature_extractor = ClassifierWrapper(width_mult=1.0, num_classes=NUM_CLASSES)
    feature_extractor.load_state_dict(torch.load(EXTRACTOR_MODEL_PATH, map_location=DEVICE))
    feature_extractor.to(DEVICE)
    feature_extractor.eval()
    print(f"   ✅ Feature extractor loaded")

    # Load PyTorch HDC model
    print("🧠 Loading PyTorch HDC damage classifier...")
    if not HDC_MODEL_PATH.exists():
        print(f"   ❌ HDC model not found at: {HDC_MODEL_PATH}")
        return None
    
    checkpoint = torch.load(HDC_MODEL_PATH, map_location=DEVICE)
    config = checkpoint['config']
    
    hdc_model = TorchHDCModel(
        config['input_dim'], 
        config['hd_dim'], 
        config['num_classes']
    ).to(DEVICE)
    
    hdc_model.load_state_dict(checkpoint['model_state_dict'])
    hdc_model.eval()
    
    best_val_acc = checkpoint.get('best_val_acc', 0)
    print(f"   ✅ PyTorch HDC model loaded")
    print(f"   📊 Validation accuracy: {best_val_acc:.2f}%")
    print(f"   🔧 Input: {config['input_dim']}D → HD: {config['hd_dim']}D → Classes: {config['num_classes']}")
    
    loading_time = time.time() - start_time
    print(f"\n⏱️  Total loading time: {loading_time:.2f}s")
    print(f"🖥️  Using device: {DEVICE}")
    
    return detector, feature_extractor, hdc_model, checkpoint

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

def process_end_to_end(detector, feature_extractor, hdc_model, image_path, output_path, save_details=True):
    """Process image through complete end-to-end pipeline"""
    
    # Load image
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return None
    
    original_h, original_w, _ = img_bgr.shape
    
    # Prepare image for detection
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(DEVICE)

    # Stage 1: Container Detection
    detection_start = time.time()
    with torch.no_grad():
        detection_preds = detector(img_tensor).cpu().numpy()[0]
    detection_time = time.time() - detection_start
    
    # Post-process detection
    detection_preds[..., :2] = sigmoid(detection_preds[..., :2])
    detection_preds[..., 4] = sigmoid(detection_preds[..., 4])
    S = detection_preds.shape[0]
    
    # Find best container
    best_container_score = 0
    container_box = None
    
    for row in range(S):
        for col in range(S):
            confidence = detection_preds[row, col, 4]
            if confidence > best_container_score and confidence > CONFIDENCE_THRESHOLD:
                best_container_score = confidence
                x_offset, y_offset, w, h = detection_preds[row, col, :4]
                x_abs, y_abs = (col + x_offset) / S, (row + y_offset) / S
                x1 = int((x_abs - w / 2) * original_w)
                y1 = int((y_abs - h / 2) * original_h)
                x2 = int((x_abs + w / 2) * original_w)
                y2 = int((y_abs + h / 2) * original_h)
                
                # Clamp to image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(original_w, x2), min(original_h, y2)
                container_box = (x1, y1, x2, y2)

    # Stage 2: Feature Extraction & Damage Classification
    damage_class = 'no_damage'
    damage_confidence = 0.0
    damage_probabilities = None
    feature_extraction_time = 0.0
    hdc_classification_time = 0.0
    
    if container_box and best_container_score > CONFIDENCE_THRESHOLD:
        x1, y1, x2, y2 = container_box
        
        # Crop container region
        container_crop = img_bgr[y1:y2, x1:x2]
        
        if container_crop.size > 0:
            # Prepare container crop for feature extraction
            container_rgb = cv2.cvtColor(container_crop, cv2.COLOR_BGR2RGB)
            container_resized = cv2.resize(container_rgb, (IMG_SIZE, IMG_SIZE))
            container_tensor = torch.from_numpy(container_resized).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(DEVICE)

            # Stage 2a: Feature Extraction
            feature_start = time.time()
            with torch.no_grad():
                features = feature_extractor(container_tensor, return_features=True).cpu().numpy()
            feature_extraction_time = time.time() - feature_start
            
            # Stage 2b: HDC Classification
            hdc_start = time.time()
            predicted_class_idx, damage_confidence, damage_probabilities = hdc_model.predict_with_confidence(features[0])
            hdc_classification_time = time.time() - hdc_start
            
            damage_class = DAMAGE_CLASSES[predicted_class_idx]

    # Visualization
    result_image = img_bgr.copy()
    
    if container_box:
        x1, y1, x2, y2 = container_box
        color = DAMAGE_COLORS[damage_class]
        
        # Draw container box
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 3)
        
        # Create label background
        label_height = 60
        cv2.rectangle(result_image, (x1, y1 - label_height), (x1 + 350, y1), color, -1)
        
        # Add text labels
        cv2.putText(result_image, f"Container: {best_container_score:.3f}", 
                   (x1 + 5, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(result_image, f"Damage: {damage_class} ({damage_confidence:.3f})", 
                   (x1 + 5, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Add timing info
        total_time = detection_time + feature_extraction_time + hdc_classification_time
        cv2.putText(result_image, f"Total: {total_time*1000:.1f}ms", 
                   (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    else:
        # No container detected
        cv2.putText(result_image, "No container detected", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Save result image
    if save_details:
        cv2.imwrite(str(output_path), result_image)

    # Return detailed results
    total_processing_time = detection_time + feature_extraction_time + hdc_classification_time
    
    return {
        'image_path': str(image_path),
        'container_detected': container_box is not None,
        'container_confidence': float(best_container_score),
        'container_box': container_box,
        'damage_class': damage_class,
        'damage_confidence': float(damage_confidence),
        'damage_probabilities': damage_probabilities.tolist() if damage_probabilities is not None else None,
        'timing': {
            'detection_ms': detection_time * 1000,
            'feature_extraction_ms': feature_extraction_time * 1000,
            'hdc_classification_ms': hdc_classification_time * 1000,
            'total_ms': total_processing_time * 1000
        }
    }

def main():
    """Main end-to-end testing function"""
    print("=" * 70)
    print("🚀 END-TO-END PIPELINE TEST WITH PYTORCH HDC MODEL")
    print("=" * 70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Load models
    models_result = load_end_to_end_models()
    if models_result is None:
        print("❌ Failed to load models")
        return
    
    detector, feature_extractor, hdc_model, checkpoint = models_result
    
    # Load test dataset
    print(f"\n📁 Loading test dataset...")
    try:
        test_dataset = DamageDetDataset(split='test', img_size=IMG_SIZE)
        print(f"✅ Test dataset loaded: {len(test_dataset)} images")
    except Exception as e:
        print(f"❌ Error loading test dataset: {e}")
        return
    
    # Process test images
    num_test_images = min(100, len(test_dataset))  # Test on 100 images
    print(f"\n🔄 Processing {num_test_images} test images through end-to-end pipeline...")
    
    all_results = []
    total_start_time = time.time()
    
    for i in tqdm(range(num_test_images), desc="End-to-End Processing"):
        img_path = test_dataset.images[i]
        output_filename = OUTPUT_DIR / f"end_to_end_{i+1:03d}_{Path(img_path).name}"
        
        result = process_end_to_end(detector, feature_extractor, hdc_model, img_path, output_filename)
        if result:
            all_results.append(result)
    
    total_processing_time = time.time() - total_start_time
    
    # Analysis and reporting
    print(f"\n📊 END-TO-END PIPELINE RESULTS")
    print("=" * 50)
    
    # Basic statistics
    containers_detected = sum(1 for r in all_results if r['container_detected'])
    damage_cases = sum(1 for r in all_results if r['container_detected'] and r['damage_class'] != 'no_damage')
    
    print(f"📈 Processing Statistics:")
    print(f"   Total images processed: {len(all_results)}")
    print(f"   Containers detected: {containers_detected}/{len(all_results)} ({containers_detected/len(all_results)*100:.1f}%)")
    print(f"   Damage detected: {damage_cases}/{containers_detected if containers_detected > 0 else 1} ({damage_cases/max(containers_detected, 1)*100:.1f}%)")
    
    # Timing analysis
    if all_results:
        avg_detection = np.mean([r['timing']['detection_ms'] for r in all_results])
        avg_feature = np.mean([r['timing']['feature_extraction_ms'] for r in all_results if r['container_detected']])
        avg_hdc = np.mean([r['timing']['hdc_classification_ms'] for r in all_results if r['container_detected']])
        avg_total = np.mean([r['timing']['total_ms'] for r in all_results])
        
        print(f"\n⏱️  Timing Analysis:")
        print(f"   Detection:          {avg_detection:.1f} ms")
        print(f"   Feature extraction: {avg_feature:.1f} ms")
        print(f"   HDC classification: {avg_hdc:.1f} ms")
        print(f"   Average total:      {avg_total:.1f} ms")
        print(f"   Throughput:         {1000/avg_total:.1f} FPS")
        
    # Damage class distribution
    damage_distribution = {}
    for result in all_results:
        if result['container_detected']:
            damage_class = result['damage_class']
            damage_distribution[damage_class] = damage_distribution.get(damage_class, 0) + 1
    
    print(f"\n🔍 Damage Class Distribution:")
    for damage_class in DAMAGE_CLASSES:
        count = damage_distribution.get(damage_class, 0)
        percentage = (count / containers_detected) * 100 if containers_detected > 0 else 0
        print(f"   {damage_class:12s}: {count:3d} cases ({percentage:5.1f}%)")
    
    # High-confidence predictions
    high_conf_threshold = 0.8
    high_conf_predictions = [r for r in all_results if r['container_detected'] and r['damage_confidence'] > high_conf_threshold]
    
    print(f"\n🎯 High-Confidence Predictions (>{high_conf_threshold}):")
    print(f"   {len(high_conf_predictions)}/{containers_detected} predictions ({len(high_conf_predictions)/max(containers_detected, 1)*100:.1f}%)")
    
    # Sample results
    print(f"\n📋 Sample Results (First 15 images):")
    print("-" * 80)
    print(f"{'#':<3} {'Container':<10} {'C.Conf':<7} {'Damage':<12} {'D.Conf':<7} {'Time(ms)':<9} {'Image'}")
    print("-" * 80)
    
    for i, result in enumerate(all_results[:15]):
        container_status = "✓" if result['container_detected'] else "✗"
        container_conf = f"{result['container_confidence']:.3f}" if result['container_detected'] else "N/A"
        damage_class = result['damage_class'] if result['container_detected'] else "N/A"
        damage_conf = f"{result['damage_confidence']:.3f}" if result['container_detected'] else "N/A"
        timing = f"{result['timing']['total_ms']:.1f}" if result['container_detected'] else "N/A"
        image_name = Path(result['image_path']).name[:15]
        
        print(f"{i+1:<3} {container_status:<10} {container_conf:<7} {damage_class:<12} {damage_conf:<7} {timing:<9} {image_name}")
    
    # Save detailed results
    results_file = OUTPUT_DIR / "end_to_end_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'model_info': {
                'detector': str(DETECTOR_MODEL_PATH),
                'feature_extractor': str(EXTRACTOR_MODEL_PATH),
                'hdc_model': str(HDC_MODEL_PATH),
                'hdc_validation_accuracy': checkpoint.get('best_val_acc', 0),
                'device': str(DEVICE)
            },
            'processing_stats': {
                'total_images': len(all_results),
                'containers_detected': containers_detected,
                'damage_detected': damage_cases,
                'total_processing_time_sec': total_processing_time,
                'average_time_per_image_ms': avg_total if all_results else 0
            },
            'damage_distribution': damage_distribution,
            'detailed_results': all_results
        }, f, indent=2)
    
    print(f"\n✅ End-to-end testing completed!")
    print(f"📁 Output images: {OUTPUT_DIR}")
    print(f"📊 Detailed results: {results_file}")
    print(f"🧠 HDC Model accuracy: {checkpoint.get('best_val_acc', 0):.2f}%")
    print(f"⚡ Average processing time: {avg_total:.1f}ms per image" if all_results else "")

if __name__ == "__main__":
    main()
