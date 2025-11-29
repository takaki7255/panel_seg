"""
Convert binary semantic masks to instance masks

This script converts binary (0/255) masks to instance masks where each
connected component (panel) gets a unique label (1, 2, 3, ...).

For touching panels, morphological erosion is applied before connected
components analysis to separate them.

Usage:
    python convert_to_instance_masks.py \
        --input ./frame_dataset/1000_dataset \
        --output ./frame_dataset/1000_instance \
        --erosion-kernel 3

    # With LSD/SDF preprocessing
    python convert_to_instance_masks.py \
        --input ./frame_dataset/1000_preprocessed \
        --output ./frame_dataset/1000_instance \
        --copy-auxiliary
"""

import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import json


def separate_instances(mask, erosion_kernel=3, min_area=500):
    """
    Convert binary mask to instance mask
    
    Args:
        mask: Binary mask (H, W) with values 0 or 255
        erosion_kernel: Kernel size for erosion to separate touching panels
        min_area: Minimum area threshold for valid panels
    
    Returns:
        instance_mask: (H, W) with unique label per panel (0=background, 1,2,3...=panels)
        num_instances: Number of detected panels
        instance_info: List of dicts with instance information
    """
    # Ensure binary
    binary = (mask > 127).astype(np.uint8)
    
    # Apply erosion to separate touching panels
    if erosion_kernel > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erosion_kernel, erosion_kernel)
        )
        eroded = cv2.erode(binary, kernel, iterations=1)
    else:
        eroded = binary
    
    # Connected components
    num_labels, labels = cv2.connectedComponents(eroded)
    
    # Filter by area and create instance mask
    instance_mask = np.zeros_like(mask, dtype=np.uint16)
    instance_info = []
    valid_label = 0
    
    for label in range(1, num_labels):  # Skip background (0)
        component_mask = (labels == label).astype(np.uint8)
        area = component_mask.sum()
        
        if area >= min_area:
            valid_label += 1
            
            # Dilate back to approximate original size
            if erosion_kernel > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (erosion_kernel, erosion_kernel)
                )
                recovered = cv2.dilate(component_mask, kernel, iterations=1)
                # Constrain to original mask
                recovered = recovered & binary
            else:
                recovered = component_mask
            
            instance_mask[recovered > 0] = valid_label
            
            # Get bounding box
            contours, _ = cv2.findContours(
                recovered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                x, y, w, h = cv2.boundingRect(contours[0])
                instance_info.append({
                    'id': valid_label,
                    'area': int(recovered.sum()),
                    'bbox': [int(x), int(y), int(w), int(h)]
                })
    
    return instance_mask, valid_label, instance_info


def visualize_instances(image, instance_mask, num_instances):
    """Create colored visualization of instances"""
    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(num_instances + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background
    
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()
    
    overlay = np.zeros_like(vis)
    for label in range(1, num_instances + 1):
        mask = (instance_mask == label)
        overlay[mask] = colors[label]
    
    result = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
    
    # Draw boundaries
    for label in range(1, num_instances + 1):
        mask = (instance_mask == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    
    return result


def process_dataset(input_dir, output_dir, split, erosion_kernel=3, min_area=500, 
                   copy_auxiliary=False, visualize=False):
    """
    Process a dataset split (train/val) or flat structure
    
    Args:
        input_dir: Input dataset directory
        output_dir: Output directory for instance masks
        split: 'train', 'val', or '' for flat structure
        erosion_kernel: Kernel size for erosion
        min_area: Minimum panel area
        copy_auxiliary: Whether to copy LSD/SDF files
        visualize: Whether to create visualizations
    """
    # Handle flat structure (empty split)
    if split == '' or split is None:
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        split_name = 'flat'
    else:
        input_path = Path(input_dir) / split
        output_path = Path(output_dir) / split
        split_name = split
        
        # Check if split exists, otherwise try flat structure
        if not input_path.exists():
            input_path = Path(input_dir)
            output_path = Path(output_dir)
            if not (input_path / 'images').exists():
                print(f"⚠️  Skipping {split}: directory not found")
                return
    
    img_dir = input_path / 'images'
    mask_dir = input_path / 'masks'
    
    if not img_dir.exists() or not mask_dir.exists():
        print(f"⚠️  Skipping {split_name}: images or masks directory not found")
        return
    
    # Create output directories
    out_img_dir = output_path / 'images'
    out_mask_dir = output_path / 'masks'
    out_instance_dir = output_path / 'instance_masks'
    
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    out_instance_dir.mkdir(parents=True, exist_ok=True)
    
    if visualize:
        out_vis_dir = output_path / 'visualizations'
        out_vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy auxiliary files if requested
    if copy_auxiliary:
        for aux_name in ['lsd', 'sdf']:
            aux_dir = input_path / aux_name
            if aux_dir.exists():
                out_aux_dir = output_path / aux_name
                out_aux_dir.mkdir(parents=True, exist_ok=True)
                for f in aux_dir.glob('*'):
                    shutil.copy2(f, out_aux_dir / f.name)
                print(f"✅ Copied {aux_name} files")
    
    # Process masks
    mask_files = sorted(mask_dir.glob('*_mask.png'))
    
    all_info = []
    total_instances = 0
    
    for mask_file in tqdm(mask_files, desc=f"Processing {split_name}"):
        stem = mask_file.stem.replace('_mask', '')
        
        # Find corresponding image
        img_file = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = img_dir / f"{stem}{ext}"
            if candidate.exists():
                img_file = candidate
                break
        
        if img_file is None:
            print(f"⚠️  Image not found for {stem}")
            continue
        
        # Load mask and image
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        
        # Convert to instance mask
        instance_mask, num_instances, instance_info = separate_instances(
            mask, erosion_kernel, min_area
        )
        
        total_instances += num_instances
        
        # Save files
        # Copy original image
        shutil.copy2(img_file, out_img_dir / img_file.name)
        
        # Copy original mask
        shutil.copy2(mask_file, out_mask_dir / mask_file.name)
        
        # Save instance mask (16-bit PNG to support many instances)
        cv2.imwrite(str(out_instance_dir / f"{stem}_instance.png"), instance_mask)
        
        # Save visualization
        if visualize:
            vis = visualize_instances(image, instance_mask, num_instances)
            cv2.imwrite(str(out_vis_dir / f"{stem}_vis.png"), vis)
        
        # Record info
        all_info.append({
            'stem': stem,
            'num_instances': num_instances,
            'instances': instance_info
        })
    
    # Save metadata
    metadata = {
        'split': split_name,
        'num_images': len(mask_files),
        'total_instances': total_instances,
        'avg_instances_per_image': total_instances / len(mask_files) if mask_files else 0,
        'erosion_kernel': erosion_kernel,
        'min_area': min_area,
        'images': all_info
    }
    
    with open(output_path / 'instance_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ {split_name}: {len(mask_files)} images, {total_instances} instances "
          f"(avg {total_instances/len(mask_files):.1f}/image)")


def main():
    parser = argparse.ArgumentParser(description='Convert binary masks to instance masks')
    
    parser.add_argument('--input', type=str, required=True,
                        help='Input dataset directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for instance dataset')
    parser.add_argument('--erosion-kernel', type=int, default=3,
                        help='Kernel size for erosion (0 to disable)')
    parser.add_argument('--min-area', type=int, default=500,
                        help='Minimum panel area to keep')
    parser.add_argument('--copy-auxiliary', action='store_true',
                        help='Copy LSD/SDF files')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualizations')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val'],
                        help='Splits to process (use --flat for no splits)')
    parser.add_argument('--flat', action='store_true',
                        help='Use flat directory structure (no train/val splits)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("Converting Binary Masks to Instance Masks")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Erosion kernel: {args.erosion_kernel}")
    print(f"Min area: {args.min_area}")
    print(f"Flat structure: {args.flat}")
    print("=" * 60 + "\n")
    
    if args.flat:
        # Process as flat directory (no splits)
        process_dataset(
            args.input, args.output, '',
            erosion_kernel=args.erosion_kernel,
            min_area=args.min_area,
            copy_auxiliary=args.copy_auxiliary,
            visualize=args.visualize
        )
    else:
        for split in args.splits:
            process_dataset(
                args.input, args.output, split,
                erosion_kernel=args.erosion_kernel,
                min_area=args.min_area,
                copy_auxiliary=args.copy_auxiliary,
                visualize=args.visualize
            )
    
    print("\n" + "=" * 60)
    print("Conversion completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
