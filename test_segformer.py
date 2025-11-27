"""
Test/Evaluation script for SegFormer

Usage:
    python test_segformer.py \
        --model ./panel_models/panel_seg-segformer-b2-01.pt \
        --model-name nvidia/mit-b2 \
        --root ./panel_dataset_processed \
        --split test \
        --save-preds \
        --output ./results/segformer-b2
"""

import argparse, time, glob
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import cv2

from models.segformer import SegFormerPanel

# ============================================================================
# Dataset
# ============================================================================
class PanelDatasetLSDSDF(Dataset):
    def __init__(self, img_dir, mask_dir, lsd_dir, sdf_dir, img_size):
        self.img_paths = sorted(glob.glob(str(img_dir / "*.jpg"))) + sorted(glob.glob(str(img_dir / "*.png")))
        if not self.img_paths:
            raise FileNotFoundError(f"No images in {img_dir}")
        self.mask_dir, self.lsd_dir, self.sdf_dir = mask_dir, lsd_dir, sdf_dir
        resize_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.img_tf = transforms.Compose([transforms.Resize(resize_size), transforms.ToTensor()])
        self.mask_tf = transforms.Compose([transforms.Resize(resize_size, interpolation=Image.NEAREST), transforms.ToTensor()])
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_p = self.img_paths[idx]
        stem = Path(img_p).stem
        img_gray = self.img_tf(Image.open(img_p).convert("L"))
        lsd = self.img_tf(Image.open(self.lsd_dir / f"{stem}_lsd.png").convert("L"))
        sdf = self.img_tf(Image.open(self.sdf_dir / f"{stem}_sdf.png").convert("L"))
        img = torch.cat([img_gray, lsd, sdf], dim=0)
        mask = self.mask_tf(Image.open(self.mask_dir / f"{stem}_mask.png").convert("L"))
        mask = (mask > 0.5).float()
        return img, mask, stem

# ============================================================================
# Boundary metrics
# ============================================================================
def extract_boundary(mask, width=3):
    kernel = np.ones((width, width), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    return (dilated - eroded).astype(np.float32)

def compute_boundary_f1(pred, gt, width=3):
    gt_boundary = extract_boundary(gt, width)
    pred_boundary = pred * (gt_boundary > 0)
    gt_boundary_mask = (gt_boundary > 0)
    if gt_boundary_mask.sum() == 0:
        return 1.0
    tp = (pred_boundary * gt_boundary_mask).sum()
    fp = (pred_boundary * (1 - gt_boundary_mask)).sum()
    fn = ((1 - pred_boundary) * gt_boundary_mask).sum()
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    return 2 * precision * recall / (precision + recall + 1e-7)

# ============================================================================
# Evaluation
# ============================================================================
@torch.no_grad()
def evaluate_model(model, loader, device, threshold=0.5, compute_pr=True):
    model.eval()
    metrics = {'dice': [], 'iou': [], 'precision': [], 'recall': [], 'f1': [], 'boundary_f1': []}
    all_probs, all_gts = [], []
    
    for x, y, stems in tqdm(loader, desc="Evaluating"):
        x = x.to(device)
        probs = torch.sigmoid(model(x))
        preds = (probs > threshold).float()
        probs_np, preds_np, y_np = probs.cpu().numpy(), preds.cpu().numpy(), y.cpu().numpy()
        
        for i in range(len(x)):
            pred, gt, prob = preds_np[i, 0], y_np[i, 0], probs_np[i, 0]
            inter = (pred * gt).sum()
            union = pred.sum() + gt.sum()
            metrics['dice'].append((2 * inter + 1e-7) / (union + 1e-7))
            metrics['iou'].append((inter + 1e-7) / (pred.sum() + gt.sum() - inter + 1e-7))
            
            tp = (pred * gt).sum()
            fp = (pred * (1 - gt)).sum()
            fn = ((1 - pred) * gt).sum()
            precision = (tp + 1e-7) / (tp + fp + 1e-7)
            recall = (tp + 1e-7) / (tp + fn + 1e-7)
            f1 = 2 * precision * recall / (precision + recall + 1e-7)
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1'].append(f1)
            metrics['boundary_f1'].append(compute_boundary_f1(pred, gt, width=3))
            
            if compute_pr:
                all_probs.append(prob.flatten())
                all_gts.append(gt.flatten())
    
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    std_metrics = {k: np.std(v) for k, v in metrics.items()}
    
    if compute_pr:
        all_probs = np.concatenate(all_probs)
        all_gts = np.concatenate(all_gts)
        precision_curve, recall_curve, _ = precision_recall_curve(all_gts, all_probs)
        avg_metrics['pr_auc'] = auc(recall_curve, precision_curve)
    
    return avg_metrics, std_metrics, all_probs, all_gts

def save_predictions(model, loader, device, output_dir, threshold=0.5):
    model.eval()
    output_dir = Path(output_dir)
    img_dir = output_dir / 'images'
    pred_dir = output_dir / 'predictions'
    vis_dir = output_dir / 'visualizations'
    img_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for x, y, stems in tqdm(loader, desc="Saving"):
            x = x.to(device)
            preds = ((torch.sigmoid(model(x)) > threshold).float().cpu().numpy() * 255).astype(np.uint8)
            y_np = (y.cpu().numpy() * 255).astype(np.uint8)
            x_np = (x.cpu().numpy() * 255).astype(np.uint8)
            
            for i in range(len(x)):
                # Save original image
                Image.fromarray(x_np[i, 0]).save(img_dir / f"{stems[i]}.png")
                # Save prediction
                Image.fromarray(preds[i, 0]).save(pred_dir / f"{stems[i]}_pred.png")
                # Save visualization
                vis = np.concatenate([
                    np.stack([x_np[i, 0]]*3, 2),
                    np.stack([y_np[i, 0]]*3, 2),
                    np.stack([preds[i, 0]]*3, 2)
                ], axis=1)
                Image.fromarray(vis).save(vis_dir / f"{stems[i]}_vis.png")
    
    print(f"✅ Original images saved to {img_dir}")
    print(f"✅ Predictions saved to {pred_dir}")
    print(f"✅ Visualizations saved to {vis_dir}")

def plot_pr_curve(all_probs, all_gts, output_dir):
    precision, recall, _ = precision_recall_curve(all_gts, all_probs)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f'PR-AUC = {pr_auc:.4f}')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(output_dir) / 'pr_curve.png', dpi=150)
    print(f"✅ PR curve saved")

# ============================================================================
# Main
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='SegFormer Evaluation')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--model-name', type=str, default='nvidia/mit-b2', 
                        help='SegFormer model name (must match training)')
    parser.add_argument('--root', type=str, required=True, help='Dataset root (preprocessed)')
    parser.add_argument('--split', type=str, default='',
                        help='Dataset split to evaluate (default: empty for flat structure, or test/train/val)')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--img-size', type=int, nargs=2, default=[512, 512])
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--save-preds', action='store_true')
    parser.add_argument('--output', type=str, default='./results/segformer')
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
    print("SegFormer Evaluation (Gray + LSD + SDF)")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Backbone: {args.model_name}")
    print(f"Device: {device}")
    print("="*60 + "\n")
    
    # Load model
    model = SegFormerPanel(
        model_name=args.model_name,
        in_channels=3,
        num_labels=1,
        pretrained=False  # We're loading trained weights
    ).to(device)
    
    model.load_state_dict(torch.load(args.model, map_location=device))
    print(f"✅ Model loaded\n")
    
    # Dataset
    root = Path(args.root)
    img_size = tuple(args.img_size)
    
    # Handle split directory structure
    if args.split:
        img_dir = root / args.split / 'images'
        mask_dir = root / args.split / 'masks'
        lsd_dir = root / args.split / 'lsd'
        sdf_dir = root / args.split / 'sdf'
    else:
        img_dir = root / 'images'
        mask_dir = root / 'masks'
        lsd_dir = root / 'lsd'
        sdf_dir = root / 'sdf'
    
    dataset = PanelDatasetLSDSDF(img_dir, mask_dir, lsd_dir, sdf_dir, img_size)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0)
    print(f"Dataset: {len(dataset)} images\n")
    
    # Evaluate
    start_time = time.time()
    avg_metrics, std_metrics, all_probs, all_gts = evaluate_model(model, loader, device, args.threshold, True)
    elapsed = time.time() - start_time
    
    # Print results
    print("\n" + "="*60)
    print("Results")
    print("="*60)
    print(f"Dice:        {avg_metrics['dice']:.4f} ± {std_metrics['dice']:.4f}")
    print(f"IoU:         {avg_metrics['iou']:.4f} ± {std_metrics['iou']:.4f}")
    print(f"Precision:   {avg_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}")
    print(f"Recall:      {avg_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}")
    print(f"F1:          {avg_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}")
    print(f"Boundary F1: {avg_metrics['boundary_f1']:.4f} ± {std_metrics['boundary_f1']:.4f}")
    print(f"PR-AUC:      {avg_metrics['pr_auc']:.4f}")
    print(f"\nTime: {elapsed:.2f}s ({elapsed/len(dataset)*1000:.1f}ms/image)")
    print("="*60 + "\n")
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'metrics.txt', 'w') as f:
        f.write("SegFormer Evaluation Results\n")
        f.write("="*60 + "\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Backbone: {args.model_name}\n")
        f.write(f"Dataset: {args.root}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Num images: {len(dataset)}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Dice:        {avg_metrics['dice']:.4f} ± {std_metrics['dice']:.4f}\n")
        f.write(f"IoU:         {avg_metrics['iou']:.4f} ± {std_metrics['iou']:.4f}\n")
        f.write(f"Precision:   {avg_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}\n")
        f.write(f"Recall:      {avg_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}\n")
        f.write(f"F1:          {avg_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}\n")
        f.write(f"Boundary F1: {avg_metrics['boundary_f1']:.4f} ± {std_metrics['boundary_f1']:.4f}\n")
        f.write(f"PR-AUC:      {avg_metrics['pr_auc']:.4f}\n")
        f.write(f"\nTime: {elapsed:.2f}s\n")
    
    print(f"✅ Metrics saved to {output_dir / 'metrics.txt'}")
    
    # Plot PR curve
    plot_pr_curve(all_probs, all_gts, output_dir)
    
    # Save predictions if requested
    if args.save_preds:
        save_predictions(model, loader, device, output_dir, args.threshold)
    
    print("\n" + "="*60)
    print("Evaluation completed!")
    print("="*60)

if __name__ == '__main__':
    main()
