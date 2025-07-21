# FILE: Modules/Module_1/preprocess_damage_data.py
import json
from pathlib import Path
import cv2
from tqdm import tqdm
import random
import shutil
import albumentations as A

# --- Augmentation Pipeline ---
def get_augmentations():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.MotionBlur(blur_limit=7, p=0.5),
        A.GridDistortion(p=0.5),
        A.CoarseDropout(max_holes=8, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8, p=0.5)
    ])

BASE = Path(__file__).resolve().parents[3]
SF_ROOT_TRAIN_VAL = BASE.parent / "SeaFront_v1_0_0" / "SeaFront_v1_0_0"
SF_ROOT_TEST = BASE.parent / "SeaFront_v1_0_0_TEST"
OUTPUT_ROOT = BASE.parent / "processed_damage_dataset"

# FINAL CLASS DEFINITION: 4 damage types + 1 no_damage type
DAMAGE_CLASS_NAMES = ['axis', 'concave', 'dentado', 'perforation', 'no_damage']
# Mapping from the JSON category IDs. We will skip ID 0 ('container').
CATEGORY_ID_TO_NAME = {
    1: 'axis',
    2: 'concave',
    3: 'dentado',
    4: 'perforation',
}
CROP_SIZE = (64, 64)
NUM_NO_DAMAGE_SAMPLES_PER_IMAGE = 3


def iou_bbox(boxA, boxB):
    # ... (function is correct)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    return interArea


def preprocess():
    print("--- Starting Final Patch-Based Dataset Preprocessing ---")
    
    augmentation_pipeline = get_augmentations()

    if OUTPUT_ROOT.exists():
        print(f"Removing existing processed dataset at: {OUTPUT_ROOT}")
        shutil.rmtree(OUTPUT_ROOT)
    print("Creating new directory for processed dataset.")
    OUTPUT_ROOT.mkdir(parents=True)

    for split in ["train", "val", "test"]:
        is_test_split = (split == "test")
        root_dir = SF_ROOT_TEST if is_test_split else SF_ROOT_TRAIN_VAL
        json_path = root_dir / f"{split}.json"
        if not json_path.exists():
            print(f"Warning: Annotation file not found at {json_path}")
            continue

        print(f"\nProcessing {split} split...")
        with open(json_path, 'r') as f:
            data = json.load(f)

        image_map = {img['id']: img for img in data['images']}
        ann_map = {}
        if 'annotations' in data:
            for ann in data['annotations']:
                img_id = ann['image_id']
                if img_id not in ann_map: ann_map[img_id] = []
                ann_map[img_id].append(ann)

        for class_name in DAMAGE_CLASS_NAMES:
            (OUTPUT_ROOT / split / class_name).mkdir(parents=True, exist_ok=True)

        for image_id, image_info in tqdm(image_map.items(), desc=f"Cropping {split} images"):
            image_path_options = [
                root_dir / "images" / split / image_info['file_name'],
                root_dir / "images" / image_info['file_name']
            ]
            image_path = next((p for p in image_path_options if p.exists()), None)
            if not image_path: continue

            img = cv2.imread(str(image_path))
            if img is None: continue

            ground_truth_boxes = []
            if image_id in ann_map:
                for ann in ann_map[image_id]:
                    category_id = ann['category_id']
                    if category_id not in CATEGORY_ID_TO_NAME: continue

                    bbox = ann['bbox']
                    x, y, w, h = [int(v) for v in bbox]
                    ground_truth_boxes.append([x, y, x + w, y + h])
                    class_name = CATEGORY_ID_TO_NAME[category_id]

                    padding = 10
                    x1, y1 = max(0, x - padding), max(0, y - padding)
                    x2, y2 = min(img.shape[1], x + w + padding), min(img.shape[0], y + h + padding)

                    cropped_img = img[y1:y2, x1:x2]
                    if cropped_img.size == 0: continue
                    
                    # Apply augmentations
                    augmented = augmentation_pipeline(image=cropped_img)
                    augmented_img = augmented['image']

                    output_filename = f"{Path(image_info['file_name']).stem}_{ann['id']}.png"
                    output_path = OUTPUT_ROOT / split / class_name / output_filename
                    cv2.imwrite(str(output_path), augmented_img)

            # Generate "no_damage" samples only if there are no ground truth boxes
            if not ground_truth_boxes:
                attempts = 0
                samples_generated = 0
                while samples_generated < NUM_NO_DAMAGE_SAMPLES_PER_IMAGE and attempts < 20:
                    attempts += 1
                    if img.shape[1] <= CROP_SIZE[0] or img.shape[0] <= CROP_SIZE[1]: break
                    rand_x = random.randint(0, img.shape[1] - CROP_SIZE[0])
                    rand_y = random.randint(0, img.shape[0] - CROP_SIZE[1])
                    
                    # No need to check for overlap if there are no ground truth boxes
                    samples_generated += 1
                    no_damage_crop = img[rand_y:rand_y + CROP_SIZE[1], rand_x:rand_x + CROP_SIZE[0]]
                    
                    # Apply augmentations
                    augmented = augmentation_pipeline(image=no_damage_crop)
                    augmented_img = augmented['image']

                    output_filename = f"{Path(image_info['file_name']).stem}_nodamage_{samples_generated}.png"
                    output_path = OUTPUT_ROOT / split / "no_damage" / output_filename
                    cv2.imwrite(str(output_path), augmented_img)

    print(f"--- Preprocessing Complete. Dataset saved to: {OUTPUT_ROOT} ---")


if __name__ == '__main__':
    preprocess()
