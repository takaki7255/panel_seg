"""
Test/Evaluation script for UNetGray

Evaluates trained model on test dataset and computes comprehensive metrics:
- Dice Score
- IoU (Intersection over Union)
- Precision, Recall, F1
- Boundary F1 (boundary-specific metric)
- PR-AUC (Precision-Recall Area Under Curve)

Usage:
    # Basic evaluation
    python test_unet_gray.py \
        --model ./panel_models/panel_seg-unetgray-01.pt \
        --root ./panel_dataset \
        --split test

    # Save predictions
    python test_unet_gray.py \
        --model ./panel_models/panel_seg-unetgray-01.pt \
        --root ./panel_dataset \
        --split test \
        --save-preds \
        --output ./results/unetgray

    # Custom threshold
    python test_unet_gray.py \
        --model ./panel_models/panel_seg-unetgray-01.pt \
        --root ./panel_dataset \
        --split test \
        --threshold 0.5
"""

import argparse
import time
from pathlib import Path
import glob

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import cv2

from models.unet_gray import UNetGray


# ============================================================================
# Dataset
# ============================================================================
class PanelDataset(Dataset):
    """Dataset for grayscale manga panel segmentation"""
    def __init__(self, img_dir, mask_dir, img_size):
        self.img_paths = sorted(glob.glob(str(img_dir / "*.jpg"))) + \
                        sorted(glob.glob(str(img_dir / "*.png")))
        if not self.img_paths:
            raise FileNotFoundError(f"No images found in {img_dir}")
        
        self.mask_dir = mask_dir
        
        if isinstance(img_size, tuple):
            resize_size = img_size
        else:
            resize_size = (img_size, img_size)
        
        self.img_tf = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.ToTensor(),
        ])
        
        self.mask_tf = transforms.Compose([
            transforms.Resize(resize_size, interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_p = self.img_paths[idx]
        stem = Path(img_p).stem
        mask_p = self.mask_dir / f"{stem}_mask.png"
        
        img = self.img_tf(Image.open(img_p).convert("L"))
        mask = self.mask_tf(Image.open(mask_p).convert("L"))
        mask = (mask > 0.5).float()
        
        return img, mask, stem


# ============================================================================
# Boundary metrics
# ============================================================================
def extract_boundary(mask, width=3):
    """
    Extract boundary from binary mask
    
    Args:
        mask: (H, W) numpy array [0, 1]
        width: boundary width in pixels
    
    Returns:
        boundary: (H, W) numpy array [0, 1]
    """
    kernel = np.ones((width, width), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    boundary = dilated - eroded
    return boundary.astype(np.float32)


def compute_boundary_f1(pred, gt, width=3):
    """
    Compute F1 score on boundary region
    
    Args:
        pred: (H, W) prediction mask [0, 1]
        gt: (H, W) ground truth mask [0, 1]
        width: boundary width
    
    Returns:
        f1: boundary F1 score
    """
    gt_boundary = extract_boundary(gt, width)
    
    # Only consider boundary region
    pred_boundary = pred * (gt_boundary > 0)
    gt_boundary_mask = (gt_boundary > 0)
    
    if gt_boundary_mask.sum() == 0:
        return 1.0  # No boundary, perfect score
    
    # Compute precision and recall on boundary
    tp = (pred_boundary * gt_boundary_mask).sum()
    fp = (pred_boundary * (1 - gt_boundary_mask)).sum()
    fn = ((1 - pred_boundary) * gt_boundary_mask).sum()
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    
    return f1


# ============================================================================
# Evaluation
# ============================================================================
@torch.no_grad()
def evaluate_model(model, loader, device, threshold=0.5, compute_pr=True):
    """
    Comprehensive evaluation of model
    
    Returns:
        metrics: dict with all computed metrics
        all_probs: list of all predicted probabilities (for PR curve)
        all_gts: list of all ground truths (for PR curve)
    """
    model.eval()
    
    metrics = {
        'dice': [],
        'iou': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'boundary_f1': []
    }
    
    all_probs = []
    all_gts = []
    
    for x, y, stems in tqdm(loader, desc="Evaluating"):
        x = x.to(device)
        
        # Predict
        logits = model(x)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        
        # Move to CPU for metrics
        probs_np = probs.cpu().numpy()
        preds_np = preds.cpu().numpy()
        y_np = y.cpu().numpy()
        
        # Batch metrics
        for i in range(len(x)):
            pred = preds_np[i, 0]
            gt = y_np[i, 0]
            prob = probs_np[i, 0]
            
            # Dice
            inter = (pred * gt).sum()
            union = pred.sum() + gt.sum()
            dice = (2 * inter + 1e-7) / (union + 1e-7)
            metrics['dice'].append(dice)
            
            # IoU
            iou = (inter + 1e-7) / (pred.sum() + gt.sum() - inter + 1e-7)
            metrics['iou'].append(iou)
            
            # Precision, Recall, F1
            tp = (pred * gt).sum()
            fp = (pred * (1 - gt)).sum()
            fn = ((1 - pred) * gt).sum()
            
            precision = (tp + 1e-7) / (tp + fp + 1e-7)
            recall = (tp + 1e-7) / (tp + fn + 1e-7)
            f1 = 2 * precision * recall / (precision + recall + 1e-7)
            
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1'].append(f1)
            
            # Boundary F1
            bdry_f1 = compute_boundary_f1(pred, gt, width=3)
            metrics['boundary_f1'].append(bdry_f1)
            
            # Collect for PR curve
            if compute_pr:
                all_probs.append(prob.flatten())
                all_gts.append(gt.flatten())
    
    # Average metrics
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    std_metrics = {k: np.std(v) for k, v in metrics.items()}
    
    # PR curve
    if compute_pr:
        all_probs = np.concatenate(all_probs)
        all_gts = np.concatenate(all_gts)
        precision_curve, recall_curve, _ = precision_recall_curve(all_gts, all_probs)
        pr_auc = auc(recall_curve, precision_curve)
        avg_metrics['pr_auc'] = pr_auc
    
    return avg_metrics, std_metrics, all_probs, all_gts


def save_predictions(model, loader, device, output_dir, threshold=0.5):
    """Save prediction masks"""
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pred_dir = output_dir / 'predictions'
    vis_dir = output_dir / 'visualizations'
    pred_dir.mkdir(exist_ok=True)
    vis_dir.mkdir(exist_ok=True)
    
    print(f"\nSaving predictions to {output_dir}")
    
    with torch.no_grad():
        for x, y, stems in tqdm(loader, desc="Saving"):
            x = x.to(device)
            
            probs = torch.sigmoid(model(x))
            preds = (probs > threshold).float()
            
            probs_np = probs.cpu().numpy()
            preds_np = preds.cpu().numpy()
            y_np = y.cpu().numpy()
            x_np = x.cpu().numpy()
            
            for i in range(len(x)):
                stem = stems[i]
                
                # Save prediction mask
                pred_mask = (preds_np[i, 0] * 255).astype(np.uint8)
                Image.fromarray(pred_mask).save(pred_dir / f"{stem}_pred.png")
                
                # Save visualization
                orig = (x_np[i, 0] * 255).astype(np.uint8)
                gt_mask = (y_np[i, 0] * 255).astype(np.uint8)
                
                # Create side-by-side visualization
                vis = np.concatenate([
                    np.stack([orig]*3, 2),
                    np.stack([gt_mask]*3, 2),
                    np.stack([pred_mask]*3, 2)
                ], axis=1)
                
                Image.fromarray(vis).save(vis_dir / f"{stem}_vis.png")
    
    print(f"✅ Predictions saved to {pred_dir}")
    print(f"✅ Visualizations saved to {vis_dir}")


def plot_pr_curve(all_probs, all_gts, output_dir):
    """Plot and save PR curve"""
    precision, recall, _ = precision_recall_curve(all_gts, all_probs)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f'PR-AUC = {pr_auc:.4f}')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'pr_curve.png', dpi=150)
    print(f"✅ PR curve saved to {output_dir / 'pr_curve.png'}")


# ============================================================================
# Main
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='UNetGray Evaluation Script')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--root', type=str, required=True,
                        help='Dataset root directory')
    parser.add_argument('--split', type=str, default='',
                        help='Dataset split to evaluate (default: empty for flat structure, or test/train/val)')
    parser.add_argument('--batch', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--img-size', type=int, nargs=2, default=[384, 512],
                        help='Image size (H W) (default: 384 512)')
    parser.add_argument('--base-channels', type=int, default=64,
                        help='Base channels (must match training) (default: 64)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Binarization threshold (default: 0.5)')
    parser.add_argument('--save-preds', action='store_true',
                        help='Save prediction masks')
    parser.add_argument('--output', type=str, default='./results/unetgray',
                        help='Output directory for results (default: ./results/unetgray)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Device selection (CUDA > MPS > CPU)
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print("\n" + "="*60)
    print("UNetGray Evaluation")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.root}")
    print(f"Split: {args.split}")
    print(f"Device: {device}")
    print(f"Threshold: {args.threshold}")
    print("="*60 + "\n")
    
    # Load model
    model = UNetGray(in_channels=1, n_classes=1, base_channels=args.base_channels).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    print(f"✅ Model loaded from {args.model}\n")
    
    # Load dataset
    root = Path(args.root)
    img_size = tuple(args.img_size)
    
    # Handle split directory structure
    if args.split:
        img_dir = root / args.split / 'images'
        mask_dir = root / args.split / 'masks'
    else:
        img_dir = root / 'images'
        mask_dir = root / 'masks'
    
    dataset = PanelDataset(img_dir, mask_dir, img_size)
    
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0)
    print(f"Dataset: {len(dataset)} images\n")
    
    # Evaluate
    start_time = time.time()
    avg_metrics, std_metrics, all_probs, all_gts = evaluate_model(
        model, loader, device, threshold=args.threshold, compute_pr=True
    )
    elapsed = time.time() - start_time
    
    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Dice:          {avg_metrics['dice']:.4f} ± {std_metrics['dice']:.4f}")
    print(f"IoU:           {avg_metrics['iou']:.4f} ± {std_metrics['iou']:.4f}")
    print(f"Precision:     {avg_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}")
    print(f"Recall:        {avg_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}")
    print(f"F1:            {avg_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}")
    print(f"Boundary F1:   {avg_metrics['boundary_f1']:.4f} ± {std_metrics['boundary_f1']:.4f}")
    print(f"PR-AUC:        {avg_metrics['pr_auc']:.4f}")
    print(f"\nTime: {elapsed:.2f}s ({elapsed/len(dataset)*1000:.1f}ms per image)")
    print("="*60 + "\n")
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics to file
    with open(output_dir / 'metrics.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("UNetGray Evaluation Results\n")
        f.write("="*60 + "\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Dataset: {args.root}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Threshold: {args.threshold}\n")
        f.write(f"Num images: {len(dataset)}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Dice:          {avg_metrics['dice']:.4f} ± {std_metrics['dice']:.4f}\n")
        f.write(f"IoU:           {avg_metrics['iou']:.4f} ± {std_metrics['iou']:.4f}\n")
        f.write(f"Precision:     {avg_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}\n")
        f.write(f"Recall:        {avg_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}\n")
        f.write(f"F1:            {avg_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}\n")
        f.write(f"Boundary F1:   {avg_metrics['boundary_f1']:.4f} ± {std_metrics['boundary_f1']:.4f}\n")
        f.write(f"PR-AUC:        {avg_metrics['pr_auc']:.4f}\n")
        f.write(f"\nTime: {elapsed:.2f}s ({elapsed/len(dataset)*1000:.1f}ms per image)\n")
    
    print(f"✅ Metrics saved to {output_dir / 'metrics.txt'}")
    
    # Plot PR curve
    plot_pr_curve(all_probs, all_gts, output_dir)
    
    # Save predictions if requested
    if args.save_preds:
        save_predictions(model, loader, device, output_dir, threshold=args.threshold)
    
    print("\n" + "="*60)
    print("Evaluation completed!")
    print("="*60)


if __name__ == '__main__':
    main()
