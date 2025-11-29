"""
Test script for Mask R-CNN Panel Instance Segmentation

Supports both:
- 3-channel input (Gray + LSD + SDF)
- 1-channel input (Grayscale only)

Usage:
    # 3-channel (Gray + LSD + SDF)
    python test_maskrcnn.py \
        --input ./frame_dataset/test100_preprocessed \
        --weights ./panel_models/maskrcnn_3ch_best.pt \
        --input-type 3ch \
        --output ./results/maskrcnn

    # Grayscale only
    python test_maskrcnn.py \
        --input ./frame_dataset/test100_dataset \
        --weights ./panel_models/maskrcnn_gray_best.pt \
        --input-type gray \
        --output ./results/maskrcnn_gray
"""

import argparse
import glob
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from PIL import Image
import cv2
from tqdm import tqdm

from models.maskrcnn import create_maskrcnn
from models.maskrcnn_gray import create_maskrcnn_gray
from utils.instance_metrics import (
    compute_instance_metrics,
    extract_instance_masks,
    MetricsAggregator
)


# ============================================================================
# Evaluation Metrics
# ============================================================================
def compute_iou(pred_mask, gt_mask):
    """Compute IoU between two binary masks"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union


def compute_dice(pred_mask, gt_mask):
    """Compute Dice coefficient"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    total = pred_mask.sum() + gt_mask.sum()
    if total == 0:
        return 0.0
    return 2 * intersection / total


def match_instances(pred_masks, gt_masks, iou_threshold=0.5):
    """
    Match predicted instances to ground truth instances
    
    Returns:
        matched_pairs: List of (pred_idx, gt_idx, iou)
        unmatched_pred: List of unmatched pred indices
        unmatched_gt: List of unmatched gt indices
    """
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return [], list(range(len(pred_masks))), list(range(len(gt_masks)))
    
    # Compute IoU matrix
    iou_matrix = np.zeros((len(pred_masks), len(gt_masks)))
    for i, pred in enumerate(pred_masks):
        for j, gt in enumerate(gt_masks):
            iou_matrix[i, j] = compute_iou(pred, gt)
    
    matched_pairs = []
    used_preds = set()
    used_gts = set()
    
    # Greedy matching
    while True:
        if len(used_preds) == len(pred_masks) or len(used_gts) == len(gt_masks):
            break
        
        # Find best match
        max_iou = 0
        best_pair = None
        for i in range(len(pred_masks)):
            if i in used_preds:
                continue
            for j in range(len(gt_masks)):
                if j in used_gts:
                    continue
                if iou_matrix[i, j] > max_iou:
                    max_iou = iou_matrix[i, j]
                    best_pair = (i, j)
        
        if best_pair is None or max_iou < iou_threshold:
            break
        
        matched_pairs.append((best_pair[0], best_pair[1], max_iou))
        used_preds.add(best_pair[0])
        used_gts.add(best_pair[1])
    
    unmatched_pred = [i for i in range(len(pred_masks)) if i not in used_preds]
    unmatched_gt = [j for j in range(len(gt_masks)) if j not in used_gts]
    
    return matched_pairs, unmatched_pred, unmatched_gt


def compute_ap(pred_masks, pred_scores, gt_masks, iou_threshold=0.5):
    """Compute Average Precision at given IoU threshold"""
    if len(gt_masks) == 0:
        return 1.0 if len(pred_masks) == 0 else 0.0
    if len(pred_masks) == 0:
        return 0.0
    
    # Sort by confidence
    sorted_indices = np.argsort(pred_scores)[::-1]
    pred_masks = [pred_masks[i] for i in sorted_indices]
    
    tp = np.zeros(len(pred_masks))
    fp = np.zeros(len(pred_masks))
    used_gt = set()
    
    for i, pred in enumerate(pred_masks):
        best_iou = 0
        best_gt = -1
        
        for j, gt in enumerate(gt_masks):
            if j in used_gt:
                continue
            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt = j
        
        if best_iou >= iou_threshold and best_gt >= 0:
            tp[i] = 1
            used_gt.add(best_gt)
        else:
            fp[i] = 1
    
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / len(gt_masks)
    
    # Compute AP using 11-point interpolation
    ap = 0
    for t in np.linspace(0, 1, 11):
        p = precision[recall >= t]
        if len(p) > 0:
            ap += max(p) / 11
    
    return ap


# ============================================================================
# Visualization
# ============================================================================
def colorize_instances(masks, scores=None, score_threshold=0.5):
    """Create colorized instance visualization"""
    if len(masks) == 0:
        return None
    
    # Random colors for each instance
    np.random.seed(42)
    colors = np.random.randint(50, 255, (len(masks), 3))
    
    h, w = masks[0].shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i, mask in enumerate(masks):
        if scores is not None and scores[i] < score_threshold:
            continue
        canvas[mask > 0] = colors[i]
    
    return canvas


def overlay_instances(image, masks, scores=None, score_threshold=0.5, alpha=0.5):
    """Overlay instance masks on image"""
    if len(masks) == 0:
        return image
    
    instance_vis = colorize_instances(masks, scores, score_threshold)
    if instance_vis is None:
        return image
    
    # Convert grayscale to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    
    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, instance_vis, alpha, 0)
    
    return overlay


# ============================================================================
# Main Test
# ============================================================================
def test(args):
    """Main test function"""
    
    # Device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print("\n" + "=" * 60)
    print("Mask R-CNN Testing for Panel Instance Segmentation")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Input type: {args.input_type}")
    print(f"Weights: {args.weights}")
    print("=" * 60 + "\n")
    
    # Create model
    if args.input_type == 'gray':
        model = create_maskrcnn_gray(num_classes=2, pretrained=False)
    else:
        model = create_maskrcnn(num_classes=2, pretrained=False)
    
    # Load weights
    state_dict = torch.load(args.weights, map_location=device)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("✅ Model loaded")
    
    # Find images
    input_dir = Path(args.input)
    img_dir = input_dir / 'images' if (input_dir / 'images').exists() else input_dir
    lsd_dir = input_dir / 'lsd'
    sdf_dir = input_dir / 'sdf'
    mask_dir = input_dir / 'masks' if (input_dir / 'masks').exists() else None
    instance_dir = input_dir / 'instance_masks' if (input_dir / 'instance_masks').exists() else None
    
    img_paths = sorted(glob.glob(str(img_dir / "*.jpg"))) + \
                sorted(glob.glob(str(img_dir / "*.png")))
    
    if not img_paths:
        raise FileNotFoundError(f"No images found in {img_dir}")
    
    print(f"Found {len(img_paths)} images\n")
    
    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'masks').mkdir(exist_ok=True)
    (output_dir / 'overlay').mkdir(exist_ok=True)
    (output_dir / 'instances').mkdir(exist_ok=True)
    
    # Image size
    img_size = tuple(args.img_size)
    
    # Metrics
    all_results = []
    total_time = 0
    
    for img_path in tqdm(img_paths, desc="Processing"):
        stem = Path(img_path).stem
        
        # Load image
        img = Image.open(img_path).convert('L')
        orig_size = img.size[::-1]  # (H, W)
        img = img.resize((img_size[1], img_size[0]), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0
        
        # Prepare input
        if args.input_type == '3ch':
            lsd_path = lsd_dir / f"{stem}_lsd.png"
            sdf_path = sdf_dir / f"{stem}_sdf.png"
            
            if lsd_path.exists():
                lsd = Image.open(lsd_path).convert('L')
                lsd = lsd.resize((img_size[1], img_size[0]), Image.BILINEAR)
                lsd_np = np.array(lsd, dtype=np.float32) / 255.0
            else:
                lsd_np = np.zeros_like(img_np)
            
            if sdf_path.exists():
                sdf = Image.open(sdf_path).convert('L')
                sdf = sdf.resize((img_size[1], img_size[0]), Image.BILINEAR)
                sdf_np = np.array(sdf, dtype=np.float32) / 255.0
            else:
                sdf_np = np.zeros_like(img_np)
            
            input_tensor = torch.from_numpy(
                np.stack([img_np, lsd_np, sdf_np], axis=0)
            ).unsqueeze(0).to(device)
        else:
            input_tensor = torch.from_numpy(img_np[np.newaxis, np.newaxis, ...]).to(device)
        
        # Inference
        start_time = time.time()
        with torch.no_grad():
            outputs = model.predict(
                input_tensor, 
                score_threshold=args.score_threshold,
                mask_threshold=args.mask_threshold
            )
        inference_time = time.time() - start_time
        total_time += inference_time
        
        # Extract predictions
        pred_masks = outputs[0]['masks']  # (N, H, W)
        pred_scores = outputs[0]['scores']
        pred_boxes = outputs[0]['boxes']
        
        # Resize masks to original size
        if len(pred_masks) > 0:
            pred_masks_resized = []
            for mask in pred_masks:
                mask_resized = cv2.resize(
                    mask.astype(np.float32),
                    (orig_size[1], orig_size[0]),
                    interpolation=cv2.INTER_LINEAR
                ) > 0.5
                pred_masks_resized.append(mask_resized.astype(np.uint8))
            pred_masks = np.array(pred_masks_resized)
        
        # Create combined mask
        combined_mask = np.zeros(orig_size, dtype=np.uint8)
        for mask in pred_masks:
            combined_mask = np.maximum(combined_mask, mask * 255)
        
        # Create instance map
        instance_map = np.zeros(orig_size, dtype=np.uint16)
        for i, mask in enumerate(pred_masks):
            instance_map[mask > 0] = i + 1
        
        # Save results
        cv2.imwrite(str(output_dir / 'masks' / f"{stem}.png"), combined_mask)
        cv2.imwrite(str(output_dir / 'instances' / f"{stem}_instance.png"), instance_map)
        
        # Create overlay
        orig_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        overlay = overlay_instances(
            orig_img, 
            pred_masks, 
            pred_scores, 
            args.score_threshold
        )
        cv2.imwrite(str(output_dir / 'overlay' / f"{stem}_overlay.png"), overlay)
        
        # Evaluate if ground truth available
        result = {
            'image': stem,
            'num_instances': len(pred_masks),
            'inference_time': inference_time
        }
        
        # Check for instance masks (preferred)
        if instance_dir and (instance_dir / f"{stem}_instance.png").exists():
            gt_instance = cv2.imread(
                str(instance_dir / f"{stem}_instance.png"),
                cv2.IMREAD_UNCHANGED
            )
            
            # Extract GT instances
            gt_masks = extract_instance_masks(gt_instance)
            
            # Use unified metrics
            metrics = compute_instance_metrics(
                list(pred_masks), gt_masks, pred_scores, iou_threshold=0.5
            )
            
            result.update({
                'gt_instances': metrics['num_gt'],
                'matched': metrics['num_matched'],
                'mean_iou': metrics['mean_iou'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'AP@50': metrics['AP@50'],
                'AP@75': metrics['AP@75'],
                'mAP': metrics['mAP']
            })
        
        elif mask_dir and (mask_dir / f"{stem}.png").exists():
            # Binary mask comparison
            gt_mask = cv2.imread(str(mask_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
            gt_mask = (gt_mask > 127).astype(np.uint8)
            
            pred_binary = combined_mask > 127
            
            iou = compute_iou(pred_binary, gt_mask)
            dice = compute_dice(pred_binary, gt_mask)
            
            result.update({
                'iou': iou,
                'dice': dice
            })
        
        all_results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("Mask R-CNN Evaluation Results")
    print("=" * 60)
    
    avg_time = total_time / len(img_paths)
    print(f"Images evaluated: {len(all_results)}")
    print(f"Average inference time: {avg_time * 1000:.1f} ms")
    print(f"Average instances detected: {np.mean([r['num_instances'] for r in all_results]):.1f}")
    
    if 'mean_iou' in all_results[0]:
        print(f"Total predictions: {sum(r['num_instances'] for r in all_results)}")
        print(f"Total ground truth: {sum(r.get('gt_instances', 0) for r in all_results)}")
        print(f"Total matched: {sum(r.get('matched', 0) for r in all_results)}")
        print("-" * 60)
        print(f"Precision:  {np.mean([r['precision'] for r in all_results]):.4f}")
        print(f"Recall:     {np.mean([r['recall'] for r in all_results]):.4f}")
        print(f"F1 Score:   {np.mean([r['f1'] for r in all_results]):.4f}")
        print(f"Mean IoU:   {np.mean([r['mean_iou'] for r in all_results]):.4f}")
        print("-" * 60)
        print(f"AP@50:      {np.mean([r['AP@50'] for r in all_results]):.4f}")
        print(f"AP@75:      {np.mean([r['AP@75'] for r in all_results]):.4f}")
        print(f"mAP:        {np.mean([r['mAP'] for r in all_results]):.4f}")
    
    elif 'iou' in all_results[0]:
        print("-" * 60)
        print(f"Binary Segmentation Metrics:")
        print(f"  Mean IoU: {np.mean([r['iou'] for r in all_results]):.4f}")
        print(f"  Mean Dice: {np.mean([r['dice'] for r in all_results]):.4f}")
    
    # Save results
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save metrics summary (same format as Mask2Former)
    if 'mean_iou' in all_results[0]:
        metrics_keys = ['precision', 'recall', 'f1', 'mean_iou', 'AP@50', 'AP@75', 'mAP']
        metrics_summary = {}
        
        for key in metrics_keys:
            values = [r[key] for r in all_results if key in r]
            if values:
                metrics_summary[f'{key}_mean'] = float(np.mean(values))
                metrics_summary[f'{key}_std'] = float(np.std(values))
        
        # Add totals
        metrics_summary['num_pred_mean'] = float(np.mean([r['num_instances'] for r in all_results]))
        metrics_summary['num_pred_std'] = float(np.std([r['num_instances'] for r in all_results]))
        metrics_summary['num_gt_mean'] = float(np.mean([r.get('gt_instances', 0) for r in all_results]))
        metrics_summary['num_gt_std'] = float(np.std([r.get('gt_instances', 0) for r in all_results]))
        metrics_summary['num_matched_mean'] = float(np.mean([r.get('matched', 0) for r in all_results]))
        metrics_summary['num_matched_std'] = float(np.std([r.get('matched', 0) for r in all_results]))
        
        metrics_summary['total_pred'] = sum(r['num_instances'] for r in all_results)
        metrics_summary['total_gt'] = sum(r.get('gt_instances', 0) for r in all_results)
        metrics_summary['total_matched'] = sum(r.get('matched', 0) for r in all_results)
        metrics_summary['num_images'] = len(all_results)
        
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics_summary, f, indent=2)
    
    print(f"\n✅ Results saved to {output_dir}")
    print("=" * 60 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description='Test Mask R-CNN for Panel Segmentation')
    
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory with images')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights')
    parser.add_argument('--output', type=str, default='./results/maskrcnn',
                        help='Output directory')
    parser.add_argument('--input-type', type=str, default='gray',
                        choices=['gray', '3ch'],
                        help='Input type: gray or 3ch (Gray+LSD+SDF)')
    parser.add_argument('--img-size', type=int, nargs=2, default=[384, 512],
                        help='Image size (H W)')
    parser.add_argument('--score-threshold', type=float, default=0.5,
                        help='Score threshold for detections')
    parser.add_argument('--mask-threshold', type=float, default=0.5,
                        help='Mask threshold for binary masks')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    test(args)
