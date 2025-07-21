"""
Local Inference Engine for ESP32-S3-EYE Container Detection
Loads trained TinyNAS + HDC models for desktop inference
"""

import os
import numpy as np
import torch
import torch.nn as nn
import cv2
import time
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TinyNASFeatureExtractor(nn.Module):
    """TinyNAS feature extractor matching ESP32 implementation"""
    
    def __init__(self, width_mult=1.0, num_classes=5):
        super().__init__()
        
        # Define the TinyNAS architecture (simplified MobileNet-like)
        # This should match the architecture from Module 2
        input_channel = self._make_divisible(32 * width_mult, 8)
        
        self.features = nn.Sequential(
            # Initial conv layer
            nn.Conv2d(3, input_channel, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),
            
            # Simplified inverted residual blocks
            self._inverted_residual_block(input_channel, 16, 1, 1, width_mult),
            self._inverted_residual_block(16, 24, 2, 1, width_mult),
            self._inverted_residual_block(24, 32, 2, 2, width_mult),
            self._inverted_residual_block(32, 64, 2, 3, width_mult),
            self._inverted_residual_block(64, 96, 1, 3, width_mult),
            self._inverted_residual_block(96, 160, 2, 3, width_mult),
            self._inverted_residual_block(160, 320, 1, 1, width_mult),
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final feature dimension
        final_channel = self._make_divisible(320 * width_mult, 8)
        
        # Projection head for 256D features
        self.projection = nn.Sequential(
            nn.Linear(final_channel, 512),
            nn.ReLU6(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU6(inplace=True)
        )
        
        # Classification head (original)
        self.classifier = nn.Linear(256, num_classes)
        
    def _make_divisible(self, v, divisor, min_value=None):
        """Ensure channel number is divisible by divisor"""
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v
    
    def _inverted_residual_block(self, in_channels, out_channels, stride, expand_ratio, width_mult):
        """Create inverted residual block (simplified MobileNetV2 block)"""
        in_channels = self._make_divisible(in_channels * width_mult, 8)
        out_channels = self._make_divisible(out_channels * width_mult, 8)
        hidden_dim = self._make_divisible(in_channels * expand_ratio, 8)
        
        layers = []
        
        # Expand
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim if expand_ratio != 1 else in_channels, 
                     hidden_dim if expand_ratio != 1 else in_channels, 
                     3, stride, 1, groups=hidden_dim if expand_ratio != 1 else in_channels, bias=False),
            nn.BatchNorm2d(hidden_dim if expand_ratio != 1 else in_channels),
            nn.ReLU6(inplace=True)
        ])
        
        # Project
        layers.extend([
            nn.Conv2d(hidden_dim if expand_ratio != 1 else in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        return nn.Sequential(*layers)
        
    def forward(self, x, return_features=False):
        # Extract features through backbone
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Get 256D features
        features = self.projection(x)
        
        if return_features:
            return features
        
        # Classification
        out = self.classifier(features)
        return out

class HDCClassifier:
    """Hyperdimensional Computing classifier"""
    
    def __init__(self, feature_dim=256, hd_dim=2048, num_classes=5):
        self.feature_dim = feature_dim
        self.hd_dim = hd_dim
        self.num_classes = num_classes
        
        # Will be loaded from saved model
        self.projection_matrix = None
        self.class_prototypes = None
        self.class_names = ['axis', 'concave', 'dentado', 'perforation', 'no_damage']
        
    def load_from_npz(self, model_path: str):
        """Load HDC model from NPZ file"""
        try:
            data = np.load(model_path)
            
            # Load projection matrix and prototypes
            if 'projection_matrix' in data:
                self.projection_matrix = data['projection_matrix']
            elif 'projection' in data:
                self.projection_matrix = data['projection']
            else:
                raise KeyError("Projection matrix not found in model file")
                
            if 'class_prototypes' in data:
                self.class_prototypes = data['class_prototypes']
            elif 'prototypes' in data:
                self.class_prototypes = data['prototypes']
            else:
                raise KeyError("Class prototypes not found in model file")
                
            logger.info(f"HDC model loaded: projection {self.projection_matrix.shape}, prototypes {self.class_prototypes.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading HDC model: {e}")
            return False
    
    def encode_features(self, features: np.ndarray) -> np.ndarray:
        """Encode features to hyperdimensional space"""
        if self.projection_matrix is None:
            raise RuntimeError("HDC model not loaded")
            
        # Project to HD space
        projected = np.dot(features, self.projection_matrix.T)
        
        # Binarize to bipolar {-1, +1}
        hypervector = np.where(projected > 0, 1, -1).astype(np.int8)
        
        return hypervector
    
    def classify(self, features: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """Classify features using HDC"""
        if self.class_prototypes is None:
            raise RuntimeError("HDC model not loaded")
            
        # Encode to hyperdimensional space
        hypervector = self.encode_features(features)
        
        # Compute similarities with all class prototypes
        similarities = np.dot(hypervector, self.class_prototypes.T) / self.hd_dim
        
        # Find best match
        predicted_class = np.argmax(similarities)
        confidence = (similarities[predicted_class] + 1) / 2  # Convert to [0, 1]
        
        return predicted_class, confidence, similarities

class LocalInferenceEngine:
    """Main inference engine for desktop processing"""
    
    def __init__(self, models_dir: str = "../../../models"):
        self.models_dir = Path(models_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Models
        self.feature_extractor = None
        self.hdc_classifier = None
        
        # Class information
        self.class_names = ['axis', 'concave', 'dentado', 'perforation', 'no_damage']
        # Simple color scheme: Red for any damage, Green for no damage
        self.box_color_damaged = (0, 0, 255)   # Red for any damage
        self.box_color_healthy = (0, 255, 0)   # Green for no damage
        
        logger.info(f"Inference engine initialized on {self.device}")
        
    def load_models(self) -> bool:
        """Load trained models"""
        try:
            # Load TinyNAS feature extractor
            feature_model_path = self.models_dir / "feature_extractor_fp32_best.pth"
            if not feature_model_path.exists():
                # Try alternative names
                alternatives = [
                    "feature_extractor_best.pth",
                    "feature_extractor_hdc_ready.pth"
                ]
                for alt in alternatives:
                    alt_path = self.models_dir / alt
                    if alt_path.exists():
                        feature_model_path = alt_path
                        break
                else:
                    logger.error(f"Feature extractor model not found in {self.models_dir}")
                    return False
            
            # Initialize and load TinyNAS model
            self.feature_extractor = TinyNASFeatureExtractor(width_mult=1.0, num_classes=5)
            state_dict = torch.load(feature_model_path, map_location=self.device)
            
            # Handle potential state dict format issues
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
                
            self.feature_extractor.load_state_dict(state_dict, strict=False)
            self.feature_extractor.to(self.device)
            self.feature_extractor.eval()
            
            logger.info(f"TinyNAS model loaded from {feature_model_path}")
            
            # Load HDC classifier
            hdc_model_path = self.models_dir / "module3_hdc_model_embhd.npz"
            if not hdc_model_path.exists():
                # Try alternative
                alt_hdc = self.models_dir / "module3_hdc_model_embhd_2048.npz"
                if alt_hdc.exists():
                    hdc_model_path = alt_hdc
                else:
                    logger.error(f"HDC model not found in {self.models_dir}")
                    return False
            
            self.hdc_classifier = HDCClassifier(feature_dim=256, hd_dim=2048, num_classes=5)
            if not self.hdc_classifier.load_from_npz(str(hdc_model_path)):
                return False
                
            logger.info("Models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int] = (64, 64)) -> torch.Tensor:
        """Preprocess image to match ESP32 pipeline"""
        # Ensure image is RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            # BGR to RGB if needed
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
        else:
            raise ValueError("Expected RGB image")
        
        # Center crop to square
        h, w = image.shape[:2]
        min_dim = min(h, w)
        
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        cropped = image[start_y:start_y + min_dim, start_x:start_x + min_dim]
        
        # Resize to target size
        resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert to tensor and normalize
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
        
        # Normalize to match training (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def detect_full_containers(self, image: np.ndarray, confidence_threshold: float = 0.7) -> List[Dict]:
        """
        NEW APPROACH: Detect full containers and classify damage status
        Returns container-level detections with green/red boxes
        """
        container_detections = []
        
        if self.feature_extractor is None or self.hdc_classifier is None:
            logger.warning("Models not loaded")
            return container_detections
        
        h, w = image.shape[:2]
        
        # Use sliding window to analyze container regions
        patch_size = 64
        stride = 32
        damage_votes = {}  # Track damage votes per region
        
        with torch.no_grad():
            # Analyze patches to find container regions
            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    patch = image[y:y + patch_size, x:x + patch_size]
                    
                    # Preprocess and extract features
                    input_tensor = self.preprocess_image(patch).to(self.device)
                    features = self.feature_extractor(input_tensor, return_features=True)
                    features_np = features.cpu().numpy().flatten()
                    
                    # Classify with HDC
                    class_id, confidence, similarities = self.hdc_classifier.classify(features_np)
                    
                    if confidence > confidence_threshold:
                        # Group patches into container regions
                        region_key = (x // 128, y // 128)  # 128x128 container regions
                        
                        if region_key not in damage_votes:
                            damage_votes[region_key] = {
                                'patches': [],
                                'damage_types': [],
                                'confidences': [],
                                'bounds': [x, y, x + patch_size, y + patch_size]
                            }
                        
                        # Update region bounds
                        damage_votes[region_key]['bounds'][0] = min(damage_votes[region_key]['bounds'][0], x)
                        damage_votes[region_key]['bounds'][1] = min(damage_votes[region_key]['bounds'][1], y)
                        damage_votes[region_key]['bounds'][2] = max(damage_votes[region_key]['bounds'][2], x + patch_size)
                        damage_votes[region_key]['bounds'][3] = max(damage_votes[region_key]['bounds'][3], y + patch_size)
                        
                        damage_votes[region_key]['patches'].append((x, y))
                        damage_votes[region_key]['damage_types'].append(self.class_names[class_id])
                        damage_votes[region_key]['confidences'].append(confidence)
        
        # Convert regions to container detections
        for region_key, region_data in damage_votes.items():
            if len(region_data['patches']) >= 3:  # Minimum patches for valid container
                
                # Determine container damage status
                damage_types = region_data['damage_types']
                confidences = region_data['confidences']
                
                # Vote for damage type (exclude no_damage)
                damage_only = [(dt, conf) for dt, conf in zip(damage_types, confidences) if dt != 'no_damage']
                
                if damage_only:
                    # Container is damaged - find most confident damage type
                    best_damage = max(damage_only, key=lambda x: x[1])
                    damage_type = best_damage[0]
                    is_damaged = True
                    container_confidence = best_damage[1]
                else:
                    # Container has no damage
                    damage_type = 'no_damage'
                    is_damaged = False
                    container_confidence = max(confidences)
                
                # Create container detection
                bounds = region_data['bounds']
                container_w = bounds[2] - bounds[0]
                container_h = bounds[3] - bounds[1]
                
                # Expand bounds to show full container
                margin = 20
                container_x = max(0, bounds[0] - margin)
                container_y = max(0, bounds[1] - margin)
                container_w = min(w - container_x, container_w + 2 * margin)
                container_h = min(h - container_y, container_h + 2 * margin)
                
                container_detection = {
                    'x': container_x,
                    'y': container_y,
                    'w': container_w,
                    'h': container_h,
                    'damage_type': damage_type,
                    'is_damaged': is_damaged,
                    'confidence': container_confidence,
                    'num_patches': len(region_data['patches'])
                }
                
                container_detections.append(container_detection)
        
        return container_detections
    
    def process_single_image(self, image: np.ndarray, confidence_threshold: float = 0.7) -> Dict:
        """
        Process a single image and return container detection results
        New approach: Detect full containers, classify as damaged/undamaged
        
        Args:
            image: Input image as numpy array
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            Dictionary with detection results for full containers
        """
        start_time = time.time()
        
        if self.feature_extractor is None or self.hdc_classifier is None:
            return {
                'detections': [],
                'num_detections': 0,
                'class_counts': {name: 0 for name in self.class_names},
                'damage_detected': False,
                'processing_time': 0.0,
                'error': 'Models not loaded'
            }
        
        try:
            # Detect containers using new approach
            container_detections = self.detect_full_containers(image, confidence_threshold)
            
            # Count classes
            class_counts = {name: 0 for name in self.class_names}
            for detection in container_detections:
                class_counts[detection['damage_type']] += 1
            
            # Check if any damage is detected
            damage_detected = any(d['is_damaged'] for d in container_detections)
            
            processing_time = time.time() - start_time
            
            return {
                'detections': container_detections,
                'num_detections': len(container_detections),
                'class_counts': class_counts,
                'damage_detected': damage_detected,
                'processing_time': processing_time
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                'detections': [],
                'num_detections': 0,
                'class_counts': {name: 0 for name in self.class_names},
                'damage_detected': False,
                'processing_time': processing_time,
                'error': str(e)
            }
    
    def filter_detections(self, detections: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
        """Apply non-maximum suppression to filter overlapping detections"""
        if not detections:
            return []
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        filtered = []
        for detection in detections:
            # Check overlap with already selected detections
            overlap = False
            for selected in filtered:
                if self.calculate_iou(detection, selected) > iou_threshold:
                    overlap = True
                    break
            
            if not overlap:
                filtered.append(detection)
        
        return filtered
    
    def calculate_iou(self, det1: Dict, det2: Dict) -> float:
        """Calculate Intersection over Union (IoU) between two detections"""
        x1_1, y1_1, w1, h1 = det1['x'], det1['y'], det1['w'], det1['h']
        x1_2, y1_2, w2, h2 = det2['x'], det2['y'], det2['w'], det2['h']
        
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        NEW APPROACH: Draw container-level detections with green/red boxes
        - Green box: No damage detected
        - Red box: Damage detected with damage type text above
        """
        result_image = image.copy()
        
        for detection in detections:
            x, y, w, h = detection['x'], detection['y'], detection['w'], detection['h']
            damage_type = detection['damage_type']
            is_damaged = detection['is_damaged']
            confidence = detection['confidence']
            
            # Choose color based on damage status
            if is_damaged:
                box_color = (0, 0, 255)  # Red for damaged containers
                text_color = (255, 255, 255)  # White text
            else:
                box_color = (0, 255, 0)  # Green for undamaged containers
                text_color = (0, 0, 0)    # Black text
            
            # Draw thick container bounding box
            box_thickness = 3
            cv2.rectangle(result_image, (x, y), (x + w, y + h), box_color, box_thickness)
            
            # Prepare label text
            if is_damaged:
                # Show damage type for damaged containers
                if damage_type == 'axis':
                    label = "AXIS DAMAGE"
                elif damage_type == 'concave':
                    label = "CONCAVE DAMAGE"
                elif damage_type == 'dentado':
                    label = "DENTADO DAMAGE"
                elif damage_type == 'perforation':
                    label = "PERFORATION DAMAGE"
                else:
                    label = f"{damage_type.upper()} DAMAGE"
            else:
                label = "NO DAMAGE"
            
            # Add confidence to label
            label_with_conf = f"{label} ({confidence:.0%})"
            
            # Calculate text size and position
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            (label_w, label_h), baseline = cv2.getTextSize(label_with_conf, font, font_scale, font_thickness)
            
            # Position text above the container box
            text_x = x
            text_y = max(label_h + 10, y - 10)  # Ensure text doesn't go off screen
            
            # Draw text background rectangle
            padding = 5
            bg_x1 = text_x - padding
            bg_y1 = text_y - label_h - padding
            bg_x2 = text_x + label_w + padding
            bg_y2 = text_y + baseline + padding
            
            # Semi-transparent background
            overlay = result_image.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), box_color, -1)
            alpha = 0.8
            result_image = cv2.addWeighted(overlay, alpha, result_image, 1 - alpha, 0)
            
            # Draw text
            cv2.putText(result_image, label_with_conf, (text_x, text_y), 
                       font, font_scale, text_color, font_thickness)
            
            # Optional: Add container ID or count info
            container_info = f"Container #{len([d for d in detections if d == detection]) + 1}"
            info_y = y + h + 20
            if info_y < result_image.shape[0] - 10:  # Check if within image bounds
                cv2.putText(result_image, container_info, (x, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
        
        return result_image

# Test function
def test_inference_engine():
    """Test the inference engine with a sample image"""
    engine = LocalInferenceEngine()
    
    if not engine.load_models():
        print("Failed to load models")
        return
    
    # Create a test image
    test_image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    
    # Process image
    result = engine.process_single_image(test_image)
    print(f"Test completed: {result['num_detections']} detections found")
    print(f"Class counts: {result['class_counts']}")

if __name__ == "__main__":
    test_inference_engine()
