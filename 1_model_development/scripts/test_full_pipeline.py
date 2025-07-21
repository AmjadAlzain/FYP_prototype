# # FILE: test_full_pipeline.py
# import sys
# import torch
# from pathlib import Path
# import numpy as np
# import cv2
# from tqdm import tqdm

# # --- Path Setup ---
# SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
# if str(SCRIPTS_ROOT) not in sys.path:
#     sys.path.append(str(SCRIPTS_ROOT))
# PROJECT_ROOT = SCRIPTS_ROOT.parents[0]

# # --- Imports ---
# from Modules.Module_1.dataset import DamageDetDataset
# from Modules.Module_2.tinynas_detection_model import create_model as create_detector_model
# from Modules.Module_2.train_classifier import ClassifierWrapper
# from Modules.Module_3.embhd.embhd_py import EmbHDModel

# # --- Configuration ---
# DETECTOR_MODEL_PATH = PROJECT_ROOT / "1_model_development" / "models" / "container_detector_best.pth"
# EXTRACTOR_MODEL_PATH = PROJECT_ROOT / "1_model_development" / "models" / "feature_extractor_fp32_best.pth"
# HDC_MODEL_PATH = PROJECT_ROOT / "1_model_development" / "models" / "module3_hdc_model_embhd_5000.npz" # Using the best model
# OUTPUT_DIR = PROJECT_ROOT / "output_images"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# IMG_SIZE = 128
# CONFIDENCE_THRESHOLD = 0.6
# DAMAGE_CLASSES = ['damage', 'no_damage', 'pothole', 'rutting', 'shoving']
# DAMAGE_COLORS = [(0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 0, 0)] # Red, Cyan, Magenta, Yellow, Black

# # --- Model Loading ---
# def load_models():
#     print("Loading models...")
#     detector = create_detector_model(num_classes=1, width_mult=0.5).to(DEVICE)
#     detector.load_state_dict(torch.load(DETECTOR_MODEL_PATH, map_location=DEVICE))
#     detector.eval()

#     feature_extractor_wrapper = ClassifierWrapper(width_mult=1.0, num_classes=len(DAMAGE_CLASSES))
#     feature_extractor_wrapper.load_state_dict(torch.load(EXTRACTOR_MODEL_PATH, map_location=DEVICE))
#     feature_extractor_wrapper.to(DEVICE) # Ensure the entire wrapper is on the device
#     feature_extractor_wrapper.eval()

#     hdc_model = EmbHDModel.load(HDC_MODEL_PATH)
#     print("Models loaded successfully.")
#     return detector, feature_extractor_wrapper, hdc_model

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def process_image(detector, feature_extractor, hdc_model, image_path, output_path):
#     img_bgr = cv2.imread(str(image_path))
#     if img_bgr is None: return
    
#     original_h, original_w, _ = img_bgr.shape
    
#     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#     img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
#     img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(DEVICE)

#     # 1. Detect Container
#     with torch.no_grad():
#         preds = detector(img_tensor).cpu().numpy()[0]
    
#     preds[..., :2] = sigmoid(preds[..., :2])
#     preds[..., 4] = sigmoid(preds[..., 4])
#     S = preds.shape[0]
    
#     best_score = 0
#     container_box = None
#     for row in range(S):
#         for col in range(S):
#             confidence = preds[row, col, 4]
#             if confidence > best_score and confidence > CONFIDENCE_THRESHOLD:
#                 best_score = confidence
#                 x_offset, y_offset, w, h = preds[row, col, :4]
#                 x_abs, y_abs = (col + x_offset) / S, (row + y_offset) / S
#                 x1 = int((x_abs - w / 2) * original_w)
#                 y1 = int((y_abs - h / 2) * original_h)
#                 x2 = int((x_abs + w / 2) * original_w)
#                 y2 = int((y_abs + h / 2) * original_h)
#                 container_box = (x1, y1, x2, y2)

#     # 2. Classify Damage within Container
#     overall_damage_detected = False
#     if container_box:
#         x1, y1, x2, y2 = container_box
#         # Crop the original full-res image for damage classification
#         container_crop_bgr = img_bgr[y1:y2, x1:x2]
#         if container_crop_bgr.size == 0:
#             # If crop is empty, just draw the box and skip classification
#             cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(img_bgr, f"Container ({best_score:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#             cv2.imwrite(str(output_path), img_bgr)
#             return

#         # Preprocess the cropped container image for the feature extractor
#         container_rgb = cv2.cvtColor(container_crop_bgr, cv2.COLOR_BGR2RGB)
#         container_resized = cv2.resize(container_rgb, (IMG_SIZE, IMG_SIZE))
#         container_tensor = torch.from_numpy(container_resized).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(DEVICE)

#         with torch.no_grad():
#             features = feature_extractor(container_tensor, return_features=True).cpu().numpy()
        
#         predicted_class_idx = hdc_model.predict(features[0])
#         predicted_class_name = DAMAGE_CLASSES[predicted_class_idx]

#         if predicted_class_name != 'no_damage':
#             overall_damage_detected = True
#             # Draw small box for the specific damage type
#             cv2.putText(img_bgr, f"Damage: {predicted_class_name}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, DAMAGE_COLORS[predicted_class_idx], 2)

#         # Draw main container box
#         color = (0, 0, 255) if overall_damage_detected else (0, 255, 0) # Red if damage, Green if not
#         cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(img_bgr, f"Container ({best_score:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#     cv2.imwrite(str(output_path), img_bgr)

# def main():
#     OUTPUT_DIR.mkdir(exist_ok=True)
#     detector, feature_extractor, hdc_model = load_models()
    
#     # Use the test set from the detection dataset
#     test_dataset = DamageDetDataset(split='test', img_size=IMG_SIZE)
    
#     num_images = min(20, len(test_dataset))
#     print(f"Testing pipeline on {num_images} images...")

#     for i in tqdm(range(num_images), desc="Processing Images"):
#         img_path = test_dataset.images[i] # Corrected attribute name
#         output_filename = OUTPUT_DIR / f"result_{Path(img_path).name}"
#         process_image(detector, feature_extractor, hdc_model, img_path, output_filename)

#     print(f"\nPipeline test complete. Output images saved to '{OUTPUT_DIR}'.")

# if __name__ == "__main__":
#     main()
