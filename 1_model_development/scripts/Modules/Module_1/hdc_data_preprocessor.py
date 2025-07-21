"""
HDC Data Preprocessor for Container Damage Classification
Extracts whole container images from SeaFront dataset and organizes by damage type
Creates prototype/hdc_dataset/ for Module 3 HDC classifier training
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import shutil
from collections import defaultdict, Counter
import json
from tqdm import tqdm
import random

# Add project root to path
BASE = Path(__file__).resolve().parents[4]
if str(BASE) not in sys.path:
    sys.path.append(str(BASE))

# Dataset paths
SEAFRONT_ROOT = BASE / "SeaFront_v1_0_0" / "SeaFront_v1_0_0"
SEAFRONT_TEST = BASE / "SeaFront_v1_0_0_TEST"
OUTPUT_ROOT = BASE / "prototype" / "hdc_dataset"

# Class definitions from SeaFront dataset
CLASS_NAMES = ['container', 'axis', 'concave', 'dentado', 'perforation']
DAMAGE_CLASSES = ['axis', 'concave', 'dentado', 'perforation', 'no_damage']

# Damage severity priority (higher number = more severe)
DAMAGE_PRIORITY = {
    'perforation': 4,
    'axis': 3, 
    'dentado': 2,
    'concave': 1,
    'no_damage': 0
}

class HDCDataPreprocessor:
    """Preprocessor to create HDC-ready dataset from SeaFront container images"""
    
    def __init__(self, output_size=256, min_container_size=50):
        self.output_size = output_size
        self.min_container_size = min_container_size
        self.stats = defaultdict(Counter)
        
        print(f"🚀 HDC Data Preprocessor initialized")
        print(f"📏 Output size: {output_size}x{output_size}")
        print(f"📐 Minimum container size: {min_container_size}px")
        
    def create_output_structure(self):
        """Create output directory structure"""
        print(f"\n📁 Creating output structure at: {OUTPUT_ROOT}")
        
        # Remove existing directory if it exists
        if OUTPUT_ROOT.exists():
            print(f"🗑️ Removing existing directory...")
            shutil.rmtree(OUTPUT_ROOT)
        
        # Create directory structure
        for split in ['train', 'val', 'test']:
            for damage_class in DAMAGE_CLASSES:
                class_dir = OUTPUT_ROOT / split / damage_class
                class_dir.mkdir(parents=True, exist_ok=True)
                
        print(f"✅ Created directory structure for {len(DAMAGE_CLASSES)} classes")
        
    def parse_yolo_annotation(self, annotation_file):
        """Parse YOLO format annotation file"""
        annotations = []
        
        if not annotation_file.exists():
            return annotations
            
        try:
            with open(annotation_file, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    annotations.append({
                        'class_id': class_id,
                        'class_name': CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else 'unknown',
                        'x_center': x_center,
                        'y_center': y_center, 
                        'width': width,
                        'height': height
                    })
                    
        except Exception as e:
            print(f"❌ Error parsing {annotation_file}: {e}")
            
        return annotations
    
    def yolo_to_pixel_coords(self, annotation, img_width, img_height):
        """Convert YOLO coordinates to pixel coordinates"""
        x_center = annotation['x_center'] * img_width
        y_center = annotation['y_center'] * img_height
        width = annotation['width'] * img_width
        height = annotation['height'] * img_height
        
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        
        # Clamp to image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width, x2)
        y2 = min(img_height, y2)
        
        return x1, y1, x2, y2
    
    def classify_container_damage(self, container_bbox, damage_annotations, img_width, img_height):
        """Classify container based on damage types within its bounding box"""
        cx1, cy1, cx2, cy2 = self.yolo_to_pixel_coords(container_bbox, img_width, img_height)
        
        damages_in_container = []
        
        for damage in damage_annotations:
            # Skip if it's a container annotation
            if damage['class_name'] == 'container':
                continue
                
            # Get damage bounding box
            dx1, dy1, dx2, dy2 = self.yolo_to_pixel_coords(damage, img_width, img_height)
            
            # Check if damage overlaps with container (IoU > threshold)
            overlap_x1 = max(cx1, dx1)
            overlap_y1 = max(cy1, dy1)
            overlap_x2 = min(cx2, dx2)
            overlap_y2 = min(cy2, dy2)
            
            if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                damage_area = (dx2 - dx1) * (dy2 - dy1)
                
                # If significant overlap (>50% of damage is within container)
                if damage_area > 0 and overlap_area / damage_area > 0.5:
                    damages_in_container.append(damage['class_name'])
        
        # Determine container class based on damage priority
        if not damages_in_container:
            return 'no_damage'
        
        # Return most severe damage type
        most_severe = max(damages_in_container, key=lambda x: DAMAGE_PRIORITY.get(x, 0))
        return most_severe
    
    def extract_container_image(self, image, container_bbox, img_width, img_height):
        """Extract container region from image"""
        x1, y1, x2, y2 = self.yolo_to_pixel_coords(container_bbox, img_width, img_height)
        
        # Check minimum size
        if (x2 - x1) < self.min_container_size or (y2 - y1) < self.min_container_size:
            return None
            
        # Extract container region
        container_image = image[y1:y2, x1:x2]
        
        if container_image.size == 0:
            return None
        
        # Resize to standard size while maintaining aspect ratio
        h, w = container_image.shape[:2]
        
        # Calculate scaling to fit within output_size
        scale = min(self.output_size / w, self.output_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize container
        container_resized = cv2.resize(container_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create padded image
        padded_image = np.zeros((self.output_size, self.output_size, 3), dtype=np.uint8)
        
        # Center the container in the padded image
        start_x = (self.output_size - new_w) // 2
        start_y = (self.output_size - new_h) // 2
        
        padded_image[start_y:start_y + new_h, start_x:start_x + new_w] = container_resized
        
        return padded_image
    
    def process_split(self, split_name, source_root):
        """Process a dataset split (train/val/test)"""
        print(f"\n🔄 Processing {split_name} split from {source_root}")
        
        images_dir = source_root / "images" / split_name
        annotations_dir = source_root / "bbannotation" / split_name
        
        if not images_dir.exists():
            print(f"❌ Images directory not found: {images_dir}")
            return
            
        if not annotations_dir.exists():
            print(f"❌ Annotations directory not found: {annotations_dir}")
            return
        
        # Get all image files
        image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
        print(f"📸 Found {len(image_files)} images")
        
        container_count = 0
        processed_images = 0
        
        # Process each image
        for img_path in tqdm(image_files, desc=f"Processing {split_name}"):
            try:
                # Load image
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img_height, img_width = image.shape[:2]
                
                # Load corresponding annotation
                annotation_file = annotations_dir / f"{img_path.stem}.txt"
                annotations = self.parse_yolo_annotation(annotation_file)
                
                if not annotations:
                    continue
                
                # Separate containers and damages
                containers = [ann for ann in annotations if ann['class_name'] == 'container']
                damages = [ann for ann in annotations if ann['class_name'] != 'container']
                
                # Process each container in the image
                for i, container in enumerate(containers):
                    # Classify container damage
                    damage_class = self.classify_container_damage(container, damages, img_width, img_height)
                    
                    # Extract container image
                    container_image = self.extract_container_image(image, container, img_width, img_height)
                    
                    if container_image is not None:
                        # Save container image
                        output_dir = OUTPUT_ROOT / split_name / damage_class
                        output_filename = f"{img_path.stem}_container_{i:02d}.png"
                        output_path = output_dir / output_filename
                        
                        # Convert back to BGR for saving
                        container_bgr = cv2.cvtColor(container_image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(output_path), container_bgr)
                        
                        # Update statistics
                        self.stats[split_name][damage_class] += 1
                        container_count += 1
                
                processed_images += 1
                
            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")
                continue
        
        print(f"✅ Processed {processed_images} images, extracted {container_count} containers")
        
    def print_statistics(self):
        """Print dataset statistics"""
        print(f"\n📊 HDC Dataset Statistics")
        print("=" * 50)
        
        total_containers = 0
        
        for split in ['train', 'val', 'test']:
            split_total = sum(self.stats[split].values())
            total_containers += split_total
            
            print(f"\n{split.upper()} Split ({split_total} containers):")
            for damage_class in DAMAGE_CLASSES:
                count = self.stats[split][damage_class]
                percentage = (count / split_total * 100) if split_total > 0 else 0
                print(f"  {damage_class:12s}: {count:4d} ({percentage:5.1f}%)")
        
        print(f"\nTOTAL CONTAINERS: {total_containers}")
        
        # Class distribution across all splits
        print(f"\nOverall Class Distribution:")
        total_by_class = defaultdict(int)
        for split in ['train', 'val', 'test']:
            for damage_class in DAMAGE_CLASSES:
                total_by_class[damage_class] += self.stats[split][damage_class]
        
        for damage_class in DAMAGE_CLASSES:
            count = total_by_class[damage_class]
            percentage = (count / total_containers * 100) if total_containers > 0 else 0
            print(f"  {damage_class:12s}: {count:4d} ({percentage:5.1f}%)")
    
    def save_statistics(self):
        """Save statistics to JSON file"""
        stats_file = OUTPUT_ROOT / "dataset_statistics.json"
        
        # Convert stats to regular dict for JSON serialization
        stats_dict = {}
        for split in self.stats:
            stats_dict[split] = dict(self.stats[split])
        
        with open(stats_file, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        
        print(f"💾 Statistics saved to: {stats_file}")
    
    def create_dataset_info(self):
        """Create dataset info file"""
        info = {
            "name": "HDC Container Damage Dataset",
            "description": "Whole container images organized by damage type for HDC classification",
            "source": "SeaFront v1.0.0 Synthetic Dataset",
            "classes": DAMAGE_CLASSES,
            "image_size": f"{self.output_size}x{self.output_size}",
            "format": "PNG images organized in class folders",
            "splits": ["train", "val", "test"],
            "damage_priority": DAMAGE_PRIORITY,
            "preprocessing": {
                "extraction": "Whole container regions from YOLO bounding boxes",
                "sizing": f"Resized to {self.output_size}x{self.output_size} with padding",
                "classification": "Based on damage types within container bounds"
            }
        }
        
        info_file = OUTPUT_ROOT / "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"📋 Dataset info saved to: {info_file}")
    
    def run(self):
        """Run the complete HDC dataset creation process"""
        print("🚀 Starting HDC Dataset Creation")
        print("=" * 50)
        
        # Create output structure
        self.create_output_structure()
        
        # Process train/val splits
        if SEAFRONT_ROOT.exists():
            self.process_split('train', SEAFRONT_ROOT)
            self.process_split('val', SEAFRONT_ROOT)
        else:
            print(f"❌ Training data not found at: {SEAFRONT_ROOT}")
        
        # Process test split
        if SEAFRONT_TEST.exists():
            self.process_split('test', SEAFRONT_TEST)
        else:
            print(f"❌ Test data not found at: {SEAFRONT_TEST}")
        
        # Generate reports
        self.print_statistics()
        self.save_statistics()
        self.create_dataset_info()
        
        print(f"\n✅ HDC dataset creation completed!")
        print(f"📁 Output directory: {OUTPUT_ROOT}")

def main():
    """Main function"""
    print("HDC Data Preprocessor for Container Damage Classification")
    print("========================================================")
    
    # Create preprocessor
    preprocessor = HDCDataPreprocessor(
        output_size=256,  # Standard size for HDC input
        min_container_size=50  # Minimum container size to process
    )
    
    # Run preprocessing
    preprocessor.run()

if __name__ == "__main__":
    main()
