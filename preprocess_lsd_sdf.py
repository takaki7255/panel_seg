"""
Preprocess script to generate LSD and SDF features from manga images

Line Segment Detector (LSD):
- Detects line segments in the image
- Outputs a line map where detected lines are highlighted
- Supports both OpenCV LSD and pylsd implementations

Signed Distance Field (SDF):
- Computes the distance from each pixel to the nearest line segment
- Normalized to [0, 1] range

Usage:
    python preprocess_lsd_sdf.py --root ./panel_dataset --output ./panel_dataset_processed
    
    # Custom parameters with line length filtering
    python preprocess_lsd_sdf.py \
        --root ./panel_dataset \
        --output ./panel_dataset_processed \
        --lsd-scale 0.8 \
        --min-line-length 10 \
        --max-line-length 500 \
        --sdf-max-dist 50
    
    # Compare OpenCV LSD and pylsd
    python preprocess_lsd_sdf.py \
        --root ./panel_dataset \
        --output ./panel_dataset_processed \
        --lsd-method pylsd \
        --compare-methods

Output structure:
    panel_dataset_processed/
    ├── train/
    │   ├── images/        # Original images (copied)
    │   ├── masks/         # Original masks (copied)
    │   ├── lsd/           # LSD line maps (single channel, 0-255)
    │   └── sdf/           # SDF distance maps (single channel, 0-255, normalized)
    ├── val/
    │   ├── images/
    │   ├── masks/
    │   ├── lsd/
    │   └── sdf/
    └── test/
        ├── images/
        ├── masks/
        ├── lsd/
        └── sdf/
"""

import os
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import distance_transform_edt

# Try to import pylsd (optional)
try:
    from pylsd.lsd import lsd as pylsd_detect
    PYLSD_AVAILABLE = True
except ImportError:
    PYLSD_AVAILABLE = False
    print("⚠️  pylsd not available. Install with: pip install pylsd")


def filter_lines_by_length(lines, min_length=10, max_length=None):
    """
    Filter line segments by length
    
    Args:
        lines: Array of lines [[x1, y1, x2, y2], ...]
        min_length: Minimum line length in pixels
        max_length: Maximum line length in pixels (None = no limit)
    
    Returns:
        filtered_lines: Lines within the length range
    """
    if lines is None or len(lines) == 0:
        return None
    
    filtered = []
    for line in lines:
        if len(line.shape) == 2:
            x1, y1, x2, y2 = line[0]
        else:
            x1, y1, x2, y2 = line[:4]
        
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        if max_length is None:
            if length >= min_length:
                filtered.append(line)
        else:
            if min_length <= length <= max_length:
                filtered.append(line)
    
    return np.array(filtered) if filtered else None


def detect_lines_opencv(image_path, scale=0.8, min_length=10, max_length=None):
    """
    Detect line segments using OpenCV LSD
    
    Args:
        image_path: Path to input image
        scale: LSD scale parameter (default: 0.8)
        min_length: Minimum line length to keep
        max_length: Maximum line length to keep (None = no limit)
    
    Returns:
        line_map: (H, W) numpy array with lines drawn (0-255)
        num_lines: Number of detected lines after filtering
    """
    # Read image as grayscale
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    
    # Create LSD detector
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    
    # Detect lines
    lines, width, prec, nfa = lsd.detect(img)
    
    # Filter by length
    if lines is not None:
        lines = filter_lines_by_length(lines, min_length, max_length)
    
    # Create blank image to draw lines
    line_map = np.zeros_like(img)
    num_lines = 0
    
    if lines is not None:
        num_lines = len(lines)
        # Draw detected lines
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_map, (int(x1), int(y1)), (int(x2), int(y2)), 255, 1)
    
    return line_map, num_lines


def detect_lines_pylsd(image_path, min_length=10, max_length=None):
    """
    Detect line segments using pylsd
    
    Args:
        image_path: Path to input image
        min_length: Minimum line length to keep
        max_length: Maximum line length to keep (None = no limit)
    
    Returns:
        line_map: (H, W) numpy array with lines drawn (0-255)
        num_lines: Number of detected lines after filtering
    """
    if not PYLSD_AVAILABLE:
        raise ImportError("pylsd is not installed. Use OpenCV LSD instead.")
    
    # Read image as grayscale
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    
    # Detect lines using pylsd
    # Returns: [[x1, y1, x2, y2, width], ...]
    lines = pylsd_detect(img)
    
    # Filter by length
    if lines is not None and len(lines) > 0:
        lines = filter_lines_by_length(lines, min_length, max_length)
    
    # Create blank image to draw lines
    line_map = np.zeros_like(img)
    num_lines = 0
    
    if lines is not None and len(lines) > 0:
        num_lines = len(lines)
        # Draw detected lines
        for line in lines:
            x1, y1, x2, y2 = line[:4]
            cv2.line(line_map, (int(x1), int(y1)), (int(x2), int(y2)), 255, 1)
    
    return line_map, num_lines


def detect_lines_lsd(image_path, scale=0.8, method='opencv', min_length=10, max_length=None):
    """
    Detect line segments using LSD (Line Segment Detector)
    
    Args:
        image_path: Path to input image
        scale: LSD scale parameter (default: 0.8) - only for OpenCV
        method: 'opencv' or 'pylsd'
        min_length: Minimum line length to keep
        max_length: Maximum line length to keep (None = no limit)
    
    Returns:
        line_map: (H, W) numpy array with lines drawn (0-255)
        num_lines: Number of detected lines after filtering
    """
    if method == 'opencv':
        return detect_lines_opencv(image_path, scale, min_length, max_length)
    elif method == 'pylsd':
        return detect_lines_pylsd(image_path, min_length, max_length)
    else:
        raise ValueError(f"Unknown LSD method: {method}. Use 'opencv' or 'pylsd'")


def compute_sdf(line_map, max_distance=50):
    """
    Compute Signed Distance Field (SDF) from line map
    
    Distance from each pixel to the nearest line segment, normalized to [0, 1]
    
    Args:
        line_map: (H, W) binary map where lines are white (255)
        max_distance: Maximum distance for normalization (default: 50 pixels)
    
    Returns:
        sdf_map: (H, W) numpy array with normalized distances [0, 255]
    """
    # Binary threshold
    binary_map = (line_map > 127).astype(np.uint8)
    
    # Compute distance transform
    # distance_transform_edt computes distance to nearest zero pixel
    # So we invert: distance to nearest white (line) pixel
    distance = distance_transform_edt(1 - binary_map)
    
    # Normalize to [0, 1]
    distance_normalized = np.clip(distance / max_distance, 0, 1)
    
    # Convert to [0, 255] for saving as image
    sdf_map = (distance_normalized * 255).astype(np.uint8)
    
    return sdf_map


def process_dataset(root_dir, output_dir, lsd_scale=0.8, sdf_max_dist=50, 
                    lsd_method='opencv', min_length=10, max_length=None):
    """
    Process entire dataset: generate LSD and SDF for all splits
    
    Args:
        root_dir: Root directory of original dataset
        output_dir: Output directory for processed dataset
        lsd_scale: LSD scale parameter (OpenCV only)
        sdf_max_dist: Maximum distance for SDF normalization
        lsd_method: 'opencv' or 'pylsd'
        min_length: Minimum line length
        max_length: Maximum line length (None = no limit)
    """
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    
    # Process each split (train, val, test)
    splits = ['train', 'val', 'test']
    
    total_stats = {'opencv': [], 'pylsd': []} if lsd_method == 'both' else {lsd_method: []}
    
    for split in splits:
        split_dir = root_dir / split
        
        if not split_dir.exists():
            print(f"⚠️  Skipping {split}: directory not found")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {split} split (method: {lsd_method})")
        print(f"{'='*60}")
        
        # Create output directories
        out_split = output_dir / split
        out_images = out_split / 'images'
        out_masks = out_split / 'masks'
        out_lsd = out_split / 'lsd'
        out_sdf = out_split / 'sdf'
        
        out_images.mkdir(parents=True, exist_ok=True)
        out_masks.mkdir(parents=True, exist_ok=True)
        out_lsd.mkdir(parents=True, exist_ok=True)
        out_sdf.mkdir(parents=True, exist_ok=True)
        
        # Get image paths
        images_dir = split_dir / 'images'
        masks_dir = split_dir / 'masks'
        
        image_paths = sorted(list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')))
        
        if len(image_paths) == 0:
            print(f"⚠️  No images found in {images_dir}")
            continue
        
        print(f"Found {len(image_paths)} images")
        if max_length is None:
            print(f"Line length filter: length >= {min_length} pixels")
        else:
            print(f"Line length filter: {min_length} <= length <= {max_length} pixels")
        
        # Process each image
        split_stats = []
        for img_path in tqdm(image_paths, desc=f"{split}"):
            stem = img_path.stem
            
            # Find corresponding mask
            mask_path = masks_dir / f"{stem}_mask.png"
            if not mask_path.exists():
                mask_path = masks_dir / f"{stem}.png"
            
            if not mask_path.exists():
                print(f"⚠️  Mask not found for {stem}, skipping")
                continue
            
            try:
                # 1. Copy original image and mask
                shutil.copy(img_path, out_images / img_path.name)
                shutil.copy(mask_path, out_masks / mask_path.name)
                
                # 2. Generate LSD line map
                line_map, num_lines = detect_lines_lsd(
                    img_path, 
                    scale=lsd_scale, 
                    method=lsd_method if lsd_method != 'both' else 'opencv',
                    min_length=min_length,
                    max_length=max_length
                )
                split_stats.append(num_lines)
                
                lsd_output = out_lsd / f"{stem}_lsd.png"
                Image.fromarray(line_map).save(lsd_output)
                
                # 3. Generate SDF from line map
                sdf_map = compute_sdf(line_map, max_distance=sdf_max_dist)
                sdf_output = out_sdf / f"{stem}_sdf.png"
                Image.fromarray(sdf_map).save(sdf_output)
                
            except Exception as e:
                print(f"❌ Error processing {stem}: {e}")
                continue
        
        method_key = lsd_method if lsd_method != 'both' else 'opencv'
        total_stats[method_key].extend(split_stats)
        
        print(f"✅ {split} split completed")
        print(f"   Images: {len(list(out_images.glob('*')))}")
        print(f"   Masks: {len(list(out_masks.glob('*')))}")
        print(f"   LSD: {len(list(out_lsd.glob('*')))}")
        print(f"   SDF: {len(list(out_sdf.glob('*')))}")
        if split_stats:
            print(f"   Lines detected (avg): {np.mean(split_stats):.1f} ± {np.std(split_stats):.1f}")
            print(f"   Lines detected (min/max): {np.min(split_stats)} / {np.max(split_stats)}")
    
    print(f"\n{'='*60}")
    print(f"✅ All processing completed!")
    print(f"Output directory: {output_dir}")
    
    # Print overall statistics
    for method, stats in total_stats.items():
        if stats:
            print(f"\n{method.upper()} Statistics:")
            print(f"  Total images: {len(stats)}")
            print(f"  Average lines per image: {np.mean(stats):.1f} ± {np.std(stats):.1f}")
            print(f"  Min/Max lines: {np.min(stats)} / {np.max(stats)}")
    
    print(f"{'='*60}")
    
    return total_stats


def compare_lsd_methods(image_path, min_length=10, max_length=None):
    """
    Compare OpenCV LSD and pylsd on a single image
    
    Args:
        image_path: Path to test image
        min_length: Minimum line length
        max_length: Maximum line length (None = no limit)
    
    Returns:
        Visualization comparing both methods
    """
    import matplotlib.pyplot as plt
    
    if not PYLSD_AVAILABLE:
        print("❌ pylsd not available. Install with: pip install pylsd")
        return
    
    # Detect with both methods
    opencv_map, opencv_count = detect_lines_opencv(image_path, min_length=min_length, max_length=max_length)
    pylsd_map, pylsd_count = detect_lines_pylsd(image_path, min_length=min_length, max_length=max_length)
    
    # Load original image
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(opencv_map, cmap='hot')
    axes[1].set_title(f'OpenCV LSD\n({opencv_count} lines)')
    axes[1].axis('off')
    
    axes[2].imshow(pylsd_map, cmap='hot')
    axes[2].set_title(f'pylsd\n({pylsd_count} lines)')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save comparison
    output_path = Path(image_path).parent.parent / 'lsd_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n{'='*60}")
    print(f"LSD Method Comparison")
    print(f"{'='*60}")
    print(f"OpenCV LSD: {opencv_count} lines detected")
    print(f"pylsd:      {pylsd_count} lines detected")
    print(f"Difference: {abs(opencv_count - pylsd_count)} lines")
    print(f"Saved: {output_path}")
    print(f"{'='*60}")
    
    plt.show()


def visualize_sample(output_dir, split='train', sample_idx=0, method='opencv'):
    """
    Visualize a sample from processed dataset
    
    Args:
        output_dir: Processed dataset directory
        split: Split to visualize (train/val/test)
        sample_idx: Index of sample to visualize
    """
    import matplotlib.pyplot as plt
    
    output_dir = Path(output_dir)
    split_dir = output_dir / split
    
    # Get file lists
    images = sorted(list((split_dir / 'images').glob('*')))
    masks = sorted(list((split_dir / 'masks').glob('*')))
    lsds = sorted(list((split_dir / 'lsd').glob('*')))
    sdfs = sorted(list((split_dir / 'sdf').glob('*')))
    
    if sample_idx >= len(images):
        print(f"❌ Sample index {sample_idx} out of range (max: {len(images)-1})")
        return
    
    # Load sample
    img = Image.open(images[sample_idx]).convert('L')
    mask = Image.open(masks[sample_idx]).convert('L')
    lsd = Image.open(lsds[sample_idx]).convert('L')
    sdf = Image.open(sdfs[sample_idx]).convert('L')
    
    # Visualize
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original (Gray)')
    axes[0].axis('off')
    
    axes[1].imshow(lsd, cmap='hot')
    axes[1].set_title('LSD Line Map')
    axes[1].axis('off')
    
    axes[2].imshow(sdf, cmap='viridis')
    axes[2].set_title('SDF Distance Map')
    axes[2].axis('off')
    
    axes[3].imshow(mask, cmap='gray')
    axes[3].set_title('GT Mask')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    vis_path = vis_dir / f'{split}_sample_{sample_idx}.png'
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved: {vis_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Preprocess manga images with LSD and SDF')
    
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory of original dataset')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for processed dataset')
    parser.add_argument('--lsd-scale', type=float, default=0.8,
                        help='LSD scale parameter (default: 0.8, OpenCV only)')
    parser.add_argument('--lsd-method', type=str, default='opencv', choices=['opencv', 'pylsd'],
                        help='LSD detection method (default: opencv)')
    parser.add_argument('--min-line-length', type=float, default=10,
                        help='Minimum line length in pixels (default: 10)')
    parser.add_argument('--max-line-length', type=float, default=None,
                        help='Maximum line length in pixels (default: None, no limit)')
    parser.add_argument('--sdf-max-dist', type=int, default=50,
                        help='Maximum distance for SDF normalization (default: 50)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize samples after processing')
    parser.add_argument('--vis-split', type=str, default='train', choices=['train', 'val', 'test'],
                        help='Split to visualize (default: train)')
    parser.add_argument('--vis-idx', type=int, default=0,
                        help='Sample index to visualize (default: 0)')
    parser.add_argument('--compare-methods', action='store_true',
                        help='Compare OpenCV LSD and pylsd on sample images')
    
    args = parser.parse_args()
    
    # Compare methods if requested
    if args.compare_methods:
        print(f"\n{'='*60}")
        print("Comparing LSD methods...")
        print(f"{'='*60}")
        
        # Find a sample image
        root_dir = Path(args.root)
        sample_dir = root_dir / args.vis_split / 'images'
        image_paths = sorted(list(sample_dir.glob('*.jpg')) + list(sample_dir.glob('*.png')))
        
        if len(image_paths) > args.vis_idx:
            compare_lsd_methods(
                image_paths[args.vis_idx],
                min_length=args.min_line_length,
                max_length=args.max_line_length
            )
        else:
            print(f"❌ No image found at index {args.vis_idx}")
        
        return
    
    # Process dataset
    stats = process_dataset(
        root_dir=args.root,
        output_dir=args.output,
        lsd_scale=args.lsd_scale,
        sdf_max_dist=args.sdf_max_dist,
        lsd_method=args.lsd_method,
        min_length=args.min_line_length,
        max_length=args.max_line_length
    )
    
    # Visualize if requested
    if args.visualize:
        print(f"\n{'='*60}")
        print("Generating visualization...")
        print(f"{'='*60}")
        visualize_sample(
            output_dir=args.output,
            split=args.vis_split,
            sample_idx=args.vis_idx,
            method=args.lsd_method
        )


if __name__ == '__main__':
    main()
