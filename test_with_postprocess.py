"""
Test script with post-processing for panel separation

Applies morphological/watershed post-processing to separate touching panels.

Usage:
    python test_with_postprocess.py \
        --model ./panel_models/panel_seg-unetgray-01.pt \
        --model-type unetgray \
        --root ./frame_dataset/test100_dataset \
        --postprocess watershed \
        --save-preds \
        --output ./results/unetgray-01-watershed
"""

import argparse
import time
import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

# Import models
from models.unet_gray import UNetGray
from models.unet_gray_lsd import UNetGrayLSD
from models.unet_gray_lsd_sdf import UNetGrayLSDSDF
from models.segformer_gray import SegFormerGray
from models.segformer import SegFormerPanel

# Import post-processing
from utils.postprocess import (
    separate_panels_morphological,
    separate_panels_watershed,
    separate_panels_combined,
    visualize_panels,
    get_panel_bboxes
)


# ============================================================================
# Datasets
# ============================================================================
class PanelDataset(Dataset):
    """Dataset for grayscale images only"""
    def __init__(self, img_dir, mask_dir, img_size):
        self.img_paths = sorted(glob.glob(str(img_dir / "*.jpg"))) + \
                        sorted(glob.glob(str(img_dir / "*.png")))
        if not self.img_paths:
            raise FileNotFoundError(f"No images found in {img_dir}")
        
        self.mask_dir = mask_dir
        resize_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        
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


class PanelDatasetLSD(Dataset):
    """Dataset for Gray + LSD"""
    def __init__(self, img_dir, mask_dir, lsd_dir, img_size):
        self.img_paths = sorted(glob.glob(str(img_dir / "*.jpg"))) + \
                        sorted(glob.glob(str(img_dir / "*.png")))
        if not self.img_paths:
            raise FileNotFoundError(f"No images found in {img_dir}")
        
        self.mask_dir = mask_dir
        self.lsd_dir = lsd_dir
        resize_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        
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
        
        img_gray = self.img_tf(Image.open(img_p).convert("L"))
        lsd = self.img_tf(Image.open(self.lsd_dir / f"{stem}_lsd.png").convert("L"))
        img = torch.cat([img_gray, lsd], dim=0)
        
        mask = self.mask_tf(Image.open(self.mask_dir / f"{stem}_mask.png").convert("L"))
        mask = (mask > 0.5).float()
        
        return img, mask, stem


class PanelDatasetLSDSDF(Dataset):
    """Dataset for Gray + LSD + SDF"""
    def __init__(self, img_dir, mask_dir, lsd_dir, sdf_dir, img_size):
        self.img_paths = sorted(glob.glob(str(img_dir / "*.jpg"))) + \
                        sorted(glob.glob(str(img_dir / "*.png")))
        if not self.img_paths:
            raise FileNotFoundError(f"No images found in {img_dir}")
        
        self.mask_dir = mask_dir
        self.lsd_dir = lsd_dir
        self.sdf_dir = sdf_dir
        resize_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        
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
        
        img_gray = self.img_tf(Image.open(img_p).convert("L"))
        lsd = self.img_tf(Image.open(self.lsd_dir / f"{stem}_lsd.png").convert("L"))
        sdf = self.img_tf(Image.open(self.sdf_dir / f"{stem}_sdf.png").convert("L"))
        img = torch.cat([img_gray, lsd, sdf], dim=0)
        
        mask = self.mask_tf(Image.open(self.mask_dir / f"{stem}_mask.png").convert("L"))
        mask = (mask > 0.5).float()
        
        return img, mask, stem


# ============================================================================
# Post-processing wrapper
# ============================================================================
def apply_postprocess(mask, method='watershed', **kwargs):
    """
    Apply post-processing to separate panels
    
    Args:
        mask: Binary mask (H, W) uint8 0-255
        method: 'morphological', 'watershed', 'combined'
        **kwargs: Additional parameters for the method
    
    Returns:
        separated_mask, labeled_mask, num_panels
    """
    if method == 'morphological':
        kernel_size = kwargs.get('kernel_size', 5)
        min_area = kwargs.get('min_area', 500)
        return separate_panels_morphological(mask, kernel_size, min_area)
    
    elif method == 'watershed':
        min_distance = kwargs.get('min_distance', 20)
        min_area = kwargs.get('min_area', 500)
        return separate_panels_watershed(mask, min_distance, min_area)
    
    elif method == 'combined':
        erosion_kernel = kwargs.get('erosion_kernel', 5)
        min_distance = kwargs.get('min_distance', 15)
        min_area = kwargs.get('min_area', 500)
        return separate_panels_combined(mask, erosion_kernel, min_distance, min_area)
    
    else:
        raise ValueError(f"Unknown method: {method}")


# ============================================================================
# Evaluation with post-processing
# ============================================================================
@torch.no_grad()
def evaluate_with_postprocess(model, loader, device, threshold=0.5, 
                               postprocess_method='watershed', **pp_kwargs):
    """Evaluate model and apply post-processing"""
    model.eval()
    
    results = []
    
    for x, y, stems in tqdm(loader, desc="Evaluating"):
        x = x.to(device)
        
        # Model prediction
        logits = model(x)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        
        # Move to numpy
        preds_np = (preds.cpu().numpy() * 255).astype(np.uint8)
        y_np = (y.cpu().numpy() * 255).astype(np.uint8)
        x_np = (x.cpu().numpy() * 255).astype(np.uint8)
        
        for i in range(len(x)):
            stem = stems[i]
            pred_mask = preds_np[i, 0]
            gt_mask = y_np[i, 0]
            
            # Apply post-processing
            separated_mask, labeled_mask, num_panels = apply_postprocess(
                pred_mask, method=postprocess_method, **pp_kwargs
            )
            
            # Get original image (first channel for multi-channel input)
            orig_img = x_np[i, 0]
            
            results.append({
                'stem': stem,
                'original': orig_img,
                'gt_mask': gt_mask,
                'pred_mask': pred_mask,
                'separated_mask': separated_mask,
                'labeled_mask': labeled_mask,
                'num_panels': num_panels
            })
    
    return results


def save_results(results, output_dir, save_visualizations=True):
    """Save all results"""
    output_dir = Path(output_dir)
    
    # Create directories
    pred_dir = output_dir / 'predictions'
    separated_dir = output_dir / 'separated'
    vis_dir = output_dir / 'visualizations'
    bbox_dir = output_dir / 'bboxes'
    
    pred_dir.mkdir(parents=True, exist_ok=True)
    separated_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    bbox_dir.mkdir(parents=True, exist_ok=True)
    
    total_panels = 0
    
    for result in tqdm(results, desc="Saving results"):
        stem = result['stem']
        
        # Save raw prediction
        Image.fromarray(result['pred_mask']).save(pred_dir / f"{stem}_pred.png")
        
        # Save separated mask
        Image.fromarray(result['separated_mask'].astype(np.uint8)).save(
            separated_dir / f"{stem}_separated.png"
        )
        
        # Save visualization with colored panels
        if save_visualizations:
            vis = visualize_panels(
                result['original'], 
                result['labeled_mask'], 
                result['num_panels']
            )
            cv2.imwrite(str(vis_dir / f"{stem}_panels.png"), vis)
            
            # Create comparison visualization
            orig_rgb = cv2.cvtColor(result['original'], cv2.COLOR_GRAY2BGR)
            gt_rgb = cv2.cvtColor(result['gt_mask'], cv2.COLOR_GRAY2BGR)
            pred_rgb = cv2.cvtColor(result['pred_mask'], cv2.COLOR_GRAY2BGR)
            sep_rgb = cv2.cvtColor(result['separated_mask'].astype(np.uint8), cv2.COLOR_GRAY2BGR)
            
            comparison = np.concatenate([orig_rgb, gt_rgb, pred_rgb, sep_rgb, vis], axis=1)
            cv2.imwrite(str(vis_dir / f"{stem}_comparison.png"), comparison)
        
        # Save bounding boxes
        bboxes = get_panel_bboxes(result['labeled_mask'], result['num_panels'])
        with open(bbox_dir / f"{stem}_bboxes.txt", 'w') as f:
            f.write(f"# {result['num_panels']} panels detected\n")
            for i, (x, y, w, h) in enumerate(bboxes):
                f.write(f"{i+1}: x={x}, y={y}, w={w}, h={h}\n")
        
        total_panels += result['num_panels']
    
    # Summary
    avg_panels = total_panels / len(results) if results else 0
    print(f"\n✅ Results saved to {output_dir}")
    print(f"   - Predictions: {pred_dir}")
    print(f"   - Separated masks: {separated_dir}")
    print(f"   - Visualizations: {vis_dir}")
    print(f"   - Bounding boxes: {bbox_dir}")
    print(f"\n📊 Statistics:")
    print(f"   - Total images: {len(results)}")
    print(f"   - Total panels detected: {total_panels}")
    print(f"   - Average panels per image: {avg_panels:.2f}")
    
    # Save summary
    with open(output_dir / 'summary.txt', 'w') as f:
        f.write("Panel Segmentation with Post-processing Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total images: {len(results)}\n")
        f.write(f"Total panels detected: {total_panels}\n")
        f.write(f"Average panels per image: {avg_panels:.2f}\n\n")
        f.write("Per-image breakdown:\n")
        for result in results:
            f.write(f"  {result['stem']}: {result['num_panels']} panels\n")


# ============================================================================
# Model loading
# ============================================================================
def load_model(model_type, model_path, device, **kwargs):
    """Load model based on type"""
    
    if model_type == 'unetgray':
        base_channels = kwargs.get('base_channels', 64)
        model = UNetGray(in_channels=1, n_classes=1, base_channels=base_channels)
    
    elif model_type == 'unetgraylsd':
        base_channels = kwargs.get('base_channels', 64)
        model = UNetGrayLSD(in_channels=2, n_classes=1, base_channels=base_channels)
    
    elif model_type == 'unetgraylsdsdf':
        base_channels = kwargs.get('base_channels', 64)
        model = UNetGrayLSDSDF(in_channels=3, n_classes=1, base_channels=base_channels)
    
    elif model_type == 'segformergray':
        model_name = kwargs.get('model_name', 'nvidia/mit-b2')
        model = SegFormerGray(model_name=model_name, num_labels=1, pretrained=False)
    
    elif model_type == 'segformer':
        model_name = kwargs.get('model_name', 'nvidia/mit-b2')
        model = SegFormerPanel(model_name=model_name, in_channels=3, num_labels=1, pretrained=False)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model


def get_dataset(model_type, root, img_size):
    """Get appropriate dataset based on model type"""
    root = Path(root)
    
    if model_type in ['unetgray', 'segformergray']:
        img_dir = root / 'images'
        mask_dir = root / 'masks'
        return PanelDataset(img_dir, mask_dir, img_size)
    
    elif model_type == 'unetgraylsd':
        img_dir = root / 'images'
        mask_dir = root / 'masks'
        lsd_dir = root / 'lsd'
        return PanelDatasetLSD(img_dir, mask_dir, lsd_dir, img_size)
    
    elif model_type in ['unetgraylsdsdf', 'segformer']:
        img_dir = root / 'images'
        mask_dir = root / 'masks'
        lsd_dir = root / 'lsd'
        sdf_dir = root / 'sdf'
        return PanelDatasetLSDSDF(img_dir, mask_dir, lsd_dir, sdf_dir, img_size)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ============================================================================
# Main
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Panel Segmentation with Post-processing')
    
    # Model
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--model-type', type=str, required=True,
                        choices=['unetgray', 'unetgraylsd', 'unetgraylsdsdf', 
                                'segformergray', 'segformer'],
                        help='Model type')
    parser.add_argument('--model-name', type=str, default='nvidia/mit-b2',
                        help='SegFormer backbone name (for segformer models)')
    parser.add_argument('--base-channels', type=int, default=64,
                        help='Base channels for UNet models')
    
    # Dataset
    parser.add_argument('--root', type=str, required=True, help='Dataset root')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--img-size', type=int, nargs=2, default=[384, 512])
    
    # Inference
    parser.add_argument('--threshold', type=float, default=0.5)
    
    # Post-processing
    parser.add_argument('--postprocess', type=str, default='watershed',
                        choices=['morphological', 'watershed', 'combined'],
                        help='Post-processing method')
    parser.add_argument('--kernel-size', type=int, default=5,
                        help='Kernel size for morphological operations')
    parser.add_argument('--min-area', type=int, default=500,
                        help='Minimum panel area to keep')
    parser.add_argument('--min-distance', type=int, default=20,
                        help='Minimum distance between panel centers (for watershed)')
    
    # Output
    parser.add_argument('--save-preds', action='store_true')
    parser.add_argument('--output', type=str, default='./results/postprocessed')
    
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
    print("Panel Segmentation with Post-processing")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Model type: {args.model_type}")
    print(f"Post-processing: {args.postprocess}")
    print(f"Device: {device}")
    print("=" * 60 + "\n")
    
    # Load model
    model = load_model(
        args.model_type, 
        args.model, 
        device,
        model_name=args.model_name,
        base_channels=args.base_channels
    )
    print("✅ Model loaded\n")
    
    # Load dataset
    img_size = tuple(args.img_size)
    dataset = get_dataset(args.model_type, args.root, img_size)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0)
    print(f"Dataset: {len(dataset)} images\n")
    
    # Evaluate with post-processing
    start_time = time.time()
    
    pp_kwargs = {
        'kernel_size': args.kernel_size,
        'min_area': args.min_area,
        'min_distance': args.min_distance,
        'erosion_kernel': args.kernel_size
    }
    
    results = evaluate_with_postprocess(
        model, loader, device,
        threshold=args.threshold,
        postprocess_method=args.postprocess,
        **pp_kwargs
    )
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Processing time: {elapsed:.2f}s ({elapsed/len(dataset)*1000:.1f}ms/image)")
    
    # Save results
    if args.save_preds:
        save_results(results, args.output, save_visualizations=True)
    
    print("\n" + "=" * 60)
    print("Processing completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
