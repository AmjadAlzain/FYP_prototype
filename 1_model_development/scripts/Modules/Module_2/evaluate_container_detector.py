# FILE: Modules/Module_2/evaluate_container_detector.py
import sys
import torch
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from torch.utils.data import DataLoader
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2

# --- Path Setup ---
SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.append(str(SCRIPTS_ROOT))
BASE = SCRIPTS_ROOT.parents[1]

# Imports from your unified dataset.py
from Modules.Module_1.dataset import DamageDetDataset, detection_collate_fn
from Modules.Module_2.tinynas_detection_model import create_model

# --- Evaluation Settings ---
BENCHMARK_DIR = BASE / "1_model_development" / "benchmarks"
EVAL_OUTPUT_DIR = BENCHMARK_DIR / "detector_evaluation"
DEFAULT_MODEL_PATH = BASE / "1_model_development" / "models" / "container_detector_best.pth"
DEFAULT_IMG_SIZE = 128
DEFAULT_WIDTH_MULT = 0.5
BATCH_SIZE = 16
NUM_CLASSES = 1  # We are evaluating the single-class detector
CONFIDENCE_THRESHOLD = 0.5  # A higher threshold is better for a single class
IOU_THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Dataset Wrapper to filter for the 'container' class ---
class ContainerOnlyDataset(DamageDetDataset):
    def __init__(self, split: str, img_size: int):
        super().__init__(split, img_size)
        self.container_class_id = 0

    def __getitem__(self, idx):
        img, original_targets = super().__getitem__(idx)
        if original_targets.numel() > 0:
            container_targets = original_targets[original_targets[:, 0] == self.container_class_id]
            if container_targets.numel() > 0:
                container_targets[:, 0] = 0
            return img, container_targets
        return img, torch.empty(0, 5)


def build_detector_loader(split, img_size, batch_size, num_workers):
    dataset = ContainerOnlyDataset(split=split, img_size=img_size)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,  # No need to shuffle for evaluation
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0), collate_fn=detection_collate_fn
    )


def iou_wh(box1_wh, box2_wh):
    intersection = torch.min(box1_wh[..., 0], box2_wh[..., 0]) * torch.min(box1_wh[..., 1], box2_wh[..., 1])
    union = (box1_wh[..., 0] * box1_wh[..., 1] + box2_wh[..., 0] * box2_wh[..., 1] - intersection)
    return intersection / (union + 1e-6)


def save_detector_sample_images(model, loader, device, output_dir, num_images=20):
    """Saves a grid of sample images with predicted and actual bounding boxes."""
    model.eval()
    images, targets = next(iter(loader))
    images = images.to(device)

    with torch.no_grad():
        predictions = model(images)
        predictions[..., :2] = torch.sigmoid(predictions[..., :2])
        predictions[..., 4] = torch.sigmoid(predictions[..., 4])

    output_dir.mkdir(parents=True, exist_ok=True)
    images = images.cpu()
    S = predictions.shape[1]

    # Limit to num_images
    num_to_save = min(num_images, images.shape[0])

    for i in range(num_to_save):
        img_tensor = images[i]
        img_np = img_tensor.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        img_bgr = (img_np * 255).astype(np.uint8)[:, :, ::-1].copy() # Convert RGB to BGR for OpenCV

        # Draw ground truth boxes (in green)
        for t in targets[i]:
            x, y, w, h = t[1:]
            x1 = int((x - w / 2) * img_bgr.shape[1])
            y1 = int((y - h / 2) * img_bgr.shape[0])
            x2 = int((x + w / 2) * img_bgr.shape[1])
            y2 = int((y + h / 2) * img_bgr.shape[0])
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green

        # Draw predicted boxes (in red)
        preds = predictions[i]
        for row in range(S):
            for col in range(S):
                confidence = preds[row, col, 4]
                if confidence < CONFIDENCE_THRESHOLD: continue
                x_offset, y_offset, w, h = preds[row, col, :4]
                x_abs, y_abs = (col + x_offset) / S, (row + y_offset) / S
                x1 = int((x_abs - w / 2) * img_bgr.shape[1])
                y1 = int((y_abs - h / 2) * img_bgr.shape[0])
                x2 = int((x_abs + w / 2) * img_bgr.shape[1])
                y2 = int((y_abs + h / 2) * img_bgr.shape[0])
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red
                cv2.putText(img_bgr, f"{confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imwrite(str(output_dir / f"sample_{i}.png"), img_bgr)

    print(f"Saved {num_to_save} sample prediction images to '{output_dir}'")


def evaluate(args):
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"FATAL: Model file not found at {model_path}")
        return

    print("--- Starting Container Detector Evaluation ---")
    print(f"Loading model from: {model_path.name}")
    print(f"Using model parameters: width_mult={args.width_mult}, img_size={args.img_size}")

    model = create_model(NUM_CLASSES, width_mult=args.width_mult).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    test_loader = build_detector_loader('test', args.img_size, BATCH_SIZE, args.num_workers)
    if len(test_loader) == 0:
        return print("Test loader is empty.")

    # --- Save a batch of sample images with predictions ---
    print("\nSaving a sample of visual predictions...")
    save_detector_sample_images(model, test_loader, DEVICE, EVAL_OUTPUT_DIR)

    all_pred_boxes, all_true_boxes = [], []
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(test_loader, desc="Evaluating Detector")):
            images = images.to(DEVICE)
            predictions = model(images)
            predictions[..., :2] = torch.sigmoid(predictions[..., :2])
            predictions[..., 4] = torch.sigmoid(predictions[..., 4])
            S = predictions.shape[1]
            for i in range(images.shape[0]):
                preds = predictions[i]
                for row in range(S):
                    for col in range(S):
                        confidence = preds[row, col, 4]
                        if confidence < CONFIDENCE_THRESHOLD: continue
                        best_class = torch.argmax(preds[row, col, 5:], dim=0)  # Should always be 0
                        x_offset, y_offset, w, h = preds[row, col, :4]
                        x_abs, y_abs = (col + x_offset) / S, (row + y_offset) / S
                        pred_box = [batch_idx * BATCH_SIZE + i, int(best_class), float(confidence),
                                    float(x_abs), float(y_abs), float(w), float(h)]
                        all_pred_boxes.append(pred_box)
                for t in targets[i]:
                    all_true_boxes.append([batch_idx * BATCH_SIZE + i] + t.tolist())

    if not all_pred_boxes:
        print(f"\nNo predictions made above the confidence threshold. mAP is 0.")
        return

    print(f"\nCalculating mAP for 'container' class...")
    average_precisions = []
    epsilon = 1e-6
    detections = all_pred_boxes  # Only one class
    ground_truths = all_true_boxes

    amount_bboxes = Counter(gt[0] for gt in ground_truths)
    for key, val in amount_bboxes.items(): amount_bboxes[key] = torch.zeros(val)
    detections.sort(key=lambda x: x[2], reverse=True)
    TP, FP = torch.zeros(len(detections)), torch.zeros(len(detections))
    total_true_bboxes = len(ground_truths)

    for detection_idx, detection in enumerate(detections):
        ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
        best_iou, best_gt_idx = 0, -1
        if not ground_truth_img: continue
        for index, gt in enumerate(ground_truth_img):
            iou_val = iou_wh(torch.tensor(detection[5:]), torch.tensor(gt[4:]))
            if iou_val > best_iou: best_iou, best_gt_idx = iou_val, index
        if best_iou > IOU_THRESHOLD:
            if amount_bboxes[detection[0]][best_gt_idx] == 0:
                TP[detection_idx] = 1
                amount_bboxes[detection[0]][best_gt_idx] = 1
            else:
                FP[detection_idx] = 1
        else:
            FP[detection_idx] = 1

    TP_cumsum, FP_cumsum = torch.cumsum(TP, dim=0), torch.cumsum(FP, dim=0)
    recalls = TP_cumsum / (total_true_bboxes + epsilon)
    precisions = torch.cat((torch.tensor([1]), TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)))
    recalls = torch.cat((torch.tensor([0]), recalls))
    ap = torch.trapz(precisions, recalls)

    print(f"\n--- Evaluation Complete ---")
    print(f"Average Precision (AP) for 'container' class @{IOU_THRESHOLD} IoU: {ap:.4f}")


def main(args):
    evaluate(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the single-class container detector.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to the trained model (.pth) file.")
    parser.add_argument("--img_size", type=int, default=DEFAULT_IMG_SIZE, help="Image size the model was trained on.")
    parser.add_argument("--width_mult", type=float, default=DEFAULT_WIDTH_MULT,
                        help="Width multiplier of the trained model.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    args = parser.parse_args()
    main(args)
