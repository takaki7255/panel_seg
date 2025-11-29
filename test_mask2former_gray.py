"""
Test/Evaluation script for Mask2Former Panel Instance Segmentation (Grayscale input)

Usage:
    python test_mask2former_gray.py \
        --model ./panel_models/mask2former_gray/mask2former_gray_best.pt \
        --root ./frame_dataset/test100_instance \
        --save-preds \
        --output ./results/mask2former_gray
"""

import argparse
import time
import glob
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import cv2

from models.mask2former_gray import create_mask2former_gray
from utils.instance_metrics import (
    compute_metrics_from_instance_maps,
    extract_instance_masks,
    MetricsAggregator
)


# ============================================================================
# Dataset
# ============================================================================
class PanelInstanceGrayTestDataset(Dataset):
    """Test dataset for instance segmentation with grayscale input"""
    
    def __init__(self, root_dir, img_size=(384, 512)):
        self.root = Path(root_dir)
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        
        self.img_dir = self.root / 'images'
        self.mask_dir = self.root / 'masks'
        self.instance_dir = self.root / 'instance_masks'
        
        # Get image list
        self.img_paths = sorted(glob.glob(str(self.img_dir / "*.jpg"))) + \
                        sorted(glob.glob(str(self.img_dir / "*.png")))
        
        if not self.img_paths:
            raise FileNotFoundError(f"No images found in {self.img_dir}")
        
        self.has_instance = self.instance_dir.exists()
        
        self.resize = transforms.Resize(self.img_size)
        self.to_tensor = transforms.ToTensor()
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        stem = Path(img_path).stem
        
        # Load grayscale image
        img_gray = Image.open(img_path).convert('L')
        original_size = img_gray.size[::-1]  # (H, W)
        img_gray = self.resize(img_gray)
        
        # Convert to tensor: (1, H, W)
        pixel_values = self.to_tensor(img_gray)
        
        # Load ground truth instance mask if available
        gt_instance = np.zeros((self.img_size[0], self.img_size[1]), dtype=np.int32)
        has_gt = False
        if self.has_instance:
            instance_path = self.instance_dir / f"{stem}_instance.png"
            if instance_path.exists():
                gt_instance = cv2.imread(str(instance_path), cv2.IMREAD_UNCHANGED)
                gt_instance = cv2.resize(
                    gt_instance,
                    (self.img_size[1], self.img_size[0]),
                    interpolation=cv2.INTER_NEAREST
                )
                has_gt = True
        
        return {
            'pixel_values': pixel_values,
            'stem': stem,
            'original_size': original_size,
            'gt_instance': gt_instance,
            'has_gt': has_gt
        }


# ============================================================================
# Visualization
# ============================================================================
def visualize_instances(image, instance_map, num_instances):
    """Create colored visualization of instances"""
    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(num_instances + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]
    
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()
    
    overlay = np.zeros_like(vis)
    
    if isinstance(instance_map, torch.Tensor):
        instance_map = instance_map.cpu().numpy()
    
    unique_ids = np.unique(instance_map)
    unique_ids = unique_ids[unique_ids > 0]
    
    for i, inst_id in enumerate(unique_ids):
        mask = (instance_map == inst_id)
        color_idx = (i % (len(colors) - 1)) + 1
        overlay[mask] = colors[color_idx]
    
    result = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
    
    # Draw boundaries
    for inst_id in unique_ids:
        mask = (instance_map == inst_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    
    return result


def get_bboxes_from_instance_map(instance_map):
    """Extract bounding boxes from instance map"""
    if isinstance(instance_map, torch.Tensor):
        instance_map = instance_map.cpu().numpy()
    
    bboxes = []
    unique_ids = np.unique(instance_map)
    unique_ids = unique_ids[unique_ids > 0]
    
    for inst_id in unique_ids:
        mask = (instance_map == inst_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            area = mask.sum()
            bboxes.append({
                'id': int(inst_id),
                'bbox': [int(x), int(y), int(w), int(h)],
                'area': int(area)
            })
    
    return bboxes


# ============================================================================
# Main
# ============================================================================
@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5, output_dir=None, save_preds=False):
    """Evaluate model on test set"""
    model.eval()
    
    # Use unified metrics aggregator
    metrics_agg = MetricsAggregator()
    results = []
    
    for batch in tqdm(loader, desc="Evaluating"):
        pixel_values = batch['pixel_values'].to(device)
        stems = batch['stem']
        gt_instances = batch['gt_instance']
        has_gts = batch['has_gt']
        
        # Predict
        instance_maps, instance_infos = model.predict_instances(
            pixel_values, threshold=threshold
        )
        
        for i in range(len(pixel_values)):
            stem = stems[i]
            pred_map = instance_maps[i]
            pred_info = instance_infos[i]
            
            # Convert to numpy
            if isinstance(pred_map, torch.Tensor):
                pred_map_np = pred_map.cpu().numpy()
            else:
                pred_map_np = np.array(pred_map)
            
            # Convert to int32 (Mask2Former returns float with -1 for background)
            if pred_map_np.dtype in [np.float32, np.float64]:
                pred_map_np = pred_map_np.astype(np.int32)
            
            # Get scores from instance info, matching instance IDs
            pred_scores = None
            if pred_info:
                # Create a mapping from instance_id to score
                instance_ids = np.unique(pred_map_np)
                instance_ids = instance_ids[instance_ids >= 0]  # Remove background (-1)
                
                # Build score mapping from segments_info
                id_to_score = {}
                for info in pred_info:
                    inst_id = info.get('id', -1)
                    score = info.get('score', 1.0)
                    id_to_score[inst_id] = score
                
                # Get scores in the same order as instance_ids
                pred_scores = np.array([id_to_score.get(int(i), 1.0) for i in instance_ids])
            
            # Get original image for visualization
            orig_img = (pixel_values[i, 0].cpu().numpy() * 255).astype(np.uint8)
            
            # Compute metrics if ground truth available
            has_gt = has_gts[i].item() if isinstance(has_gts[i], torch.Tensor) else has_gts[i]
            
            if has_gt:
                gt_instance = gt_instances[i]
                if isinstance(gt_instance, torch.Tensor):
                    gt_instance = gt_instance.cpu().numpy()
                
                # Use unified metrics
                metrics = compute_metrics_from_instance_maps(
                    pred_map_np, gt_instance, pred_scores, iou_threshold=0.5
                )
                metrics_agg.add(metrics)
                
                num_pred = metrics['num_pred']
                num_gt = metrics['num_gt']
            else:
                pred_ids = np.unique(pred_map_np)
                num_pred = len(pred_ids[pred_ids > 0])
                num_gt = -1
            
            results.append({
                'stem': stem,
                'pred_map': pred_map_np,
                'num_pred_instances': num_pred,
                'num_gt_instances': num_gt,
                'orig_img': orig_img
            })
    
    # Get summary metrics
    summary = metrics_agg.compute_summary()
    
    # Save results
    if save_preds and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pred_dir = output_dir / 'predictions'
        vis_dir = output_dir / 'visualizations'
        bbox_dir = output_dir / 'bboxes'
        
        pred_dir.mkdir(exist_ok=True)
        vis_dir.mkdir(exist_ok=True)
        bbox_dir.mkdir(exist_ok=True)
        
        for result in tqdm(results, desc="Saving"):
            stem = result['stem']
            pred_map = result['pred_map']
            orig_img = result['orig_img']
            num_instances = result['num_pred_instances']
            
            # Save instance mask
            cv2.imwrite(str(pred_dir / f"{stem}_instance.png"), pred_map.astype(np.uint16))
            
            # Save visualization
            vis = visualize_instances(orig_img, pred_map, num_instances)
            cv2.imwrite(str(vis_dir / f"{stem}_vis.png"), vis)
            
            # Save bboxes
            bboxes = get_bboxes_from_instance_map(pred_map)
            with open(bbox_dir / f"{stem}_bboxes.json", 'w') as f:
                json.dump(bboxes, f, indent=2)
        
        # Save metrics
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Results saved to {output_dir}")
    
    return summary, results


def parse_args():
    parser = argparse.ArgumentParser(description='Test Mask2Former Gray for Panel Segmentation')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--model-name', type=str,
                        default='facebook/mask2former-swin-tiny-coco-instance',
                        help='Model architecture name')
    parser.add_argument('--root', type=str, required=True,
                        help='Test dataset root')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--img-size', type=int, nargs=2, default=[384, 512])
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--save-preds', action='store_true')
    parser.add_argument('--output', type=str, default='./results/mask2former_gray')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print("\n" + "=" * 60)
    print("Mask2Former Gray Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Input: Grayscale (1 channel)")
    print(f"Device: {device}")
    print("=" * 60 + "\n")
    
    # Load model
    model = create_mask2former_gray(
        model_name=args.model_name,
        num_labels=1,
        pretrained=False
    )
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)
    print("✅ Model loaded\n")
    
    # Dataset
    img_size = tuple(args.img_size)
    dataset = PanelInstanceGrayTestDataset(args.root, img_size=img_size)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0)
    print(f"Dataset: {len(dataset)} images\n")
    
    # Evaluate
    start_time = time.time()
    
    summary, results = evaluate(
        model, loader, device,
        threshold=args.threshold,
        output_dir=args.output,
        save_preds=args.save_preds
    )
    
    elapsed = time.time() - start_time
    
    # Print results using unified format
    print("\n" + "=" * 60)
    print("Mask2Former Gray Evaluation Results")
    print("=" * 60)
    print(f"Images evaluated: {summary.get('num_images', len(results))}")
    print(f"Total predictions: {summary.get('total_pred', 0)}")
    print(f"Total ground truth: {summary.get('total_gt', 0)}")
    print(f"Total matched: {summary.get('total_matched', 0)}")
    print("-" * 60)
    print(f"Precision:  {summary.get('precision_mean', 0):.4f} ± {summary.get('precision_std', 0):.4f}")
    print(f"Recall:     {summary.get('recall_mean', 0):.4f} ± {summary.get('recall_std', 0):.4f}")
    print(f"F1 Score:   {summary.get('f1_mean', 0):.4f} ± {summary.get('f1_std', 0):.4f}")
    print(f"Mean IoU:   {summary.get('mean_iou_mean', 0):.4f} ± {summary.get('mean_iou_std', 0):.4f}")
    print("-" * 60)
    print(f"AP@50:      {summary.get('AP@50_mean', 0):.4f} ± {summary.get('AP@50_std', 0):.4f}")
    print(f"AP@75:      {summary.get('AP@75_mean', 0):.4f} ± {summary.get('AP@75_std', 0):.4f}")
    print(f"mAP:        {summary.get('mAP_mean', 0):.4f} ± {summary.get('mAP_std', 0):.4f}")
    print("-" * 60)
    print(f"Time: {elapsed:.2f}s ({elapsed/len(dataset)*1000:.1f}ms/image)")
    print("=" * 60)


if __name__ == '__main__':
    main()
