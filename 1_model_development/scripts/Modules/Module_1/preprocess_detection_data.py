# FILE: Modules/Module_1/preprocess_detection_data.py
import json
from pathlib import Path
import cv2
from tqdm import tqdm
import shutil
import albumentations as A
import numpy as np

BASE = Path(__file__).resolve().parents[3]
SF_ROOT_TRAIN_VAL = BASE.parent / "SeaFront_v1_0_0" / "SeaFront_v1_0_0"
SF_ROOT_TEST = BASE.parent / "SeaFront_v1_0_0_TEST"
OUTPUT_ROOT = BASE.parent / "processed_detection_dataset"
IMG_SIZE = 128

# --- Augmentation Pipeline ---
def get_augmentations():
    return A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.GaussNoise(p=0.2),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def preprocess_detection():
    print("--- Starting Detection Dataset Preprocessing ---")
    if OUTPUT_ROOT.exists():
        print(f"Removing existing processed dataset at: {OUTPUT_ROOT}")
        shutil.rmtree(OUTPUT_ROOT)
    print("Creating new directory for processed dataset.")
    
    augmentation_pipeline = get_augmentations()

    for split in ["train", "val", "test"]:
        is_test_split = (split == "test")
        root_dir = SF_ROOT_TEST if is_test_split else SF_ROOT_TRAIN_VAL
        json_path = root_dir / f"{split}.json"
        if not json_path.exists():
            # try the other root
            root_dir = SF_ROOT_TRAIN_VAL if is_test_split else SF_ROOT_TEST
            json_path = root_dir / f"{split}.json"
        if not json_path.exists():
            print(f"Warning: Annotation file not found at {json_path}")
            continue

        print(f"\nProcessing {split} split...")
        
        (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

        with open(json_path, 'r') as f:
            data = json.load(f)

        image_map = {img['id']: img for img in data['images']}
        ann_map = {}
        if 'annotations' in data:
            for ann in data['annotations']:
                img_id = ann['image_id']
                if ann['category_id'] == 0: # Only process 'container' class
                    if img_id not in ann_map: ann_map[img_id] = []
                    ann_map[img_id].append(ann)

        for image_id, image_info in tqdm(image_map.items(), desc=f"Processing {split} images"):
            if image_id not in ann_map:
                continue

            image_path_options = [
                root_dir / "images" / split / image_info['file_name'],
                root_dir / "images" / image_info['file_name']
            ]
            image_path = next((p for p in image_path_options if p.exists()), None)
            if not image_path: continue

            img = cv2.imread(str(image_path))
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img_h, img_w = img.shape[:2]

            bboxes = []
            class_labels = []
            for ann in ann_map[image_id]:
                bbox = ann['bbox']
                x, y, w, h = bbox
                
                # Convert COCO to YOLO format
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                width = w / img_w
                height = h / img_h
                
                bboxes.append([x_center, y_center, width, height])
                class_labels.append(0) # Class 0 for 'container'

            if not bboxes:
                continue

            try:
                augmented = augmentation_pipeline(image=img, bboxes=bboxes, class_labels=class_labels)
                augmented_img = augmented['image']
                augmented_bboxes = augmented['bboxes']

                # Save augmented image
                output_img_path = OUTPUT_ROOT / "images" / split / image_info['file_name']
                cv2.imwrite(str(output_img_path), cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR))

                # Save augmented labels
                output_label_path = OUTPUT_ROOT / "labels" / split / f"{Path(image_info['file_name']).stem}.txt"
                with open(output_label_path, 'w') as f:
                    for bbox in augmented_bboxes:
                        x_center, y_center, width, height = bbox
                        f.write(f"0 {x_center} {y_center} {width} {height}\n")

            except Exception as e:
                print(f"Could not augment image {image_info['file_name']}: {e}")


    print(f"--- Detection Preprocessing Complete. Dataset saved to: {OUTPUT_ROOT} ---")

if __name__ == '__main__':
    preprocess_detection()
