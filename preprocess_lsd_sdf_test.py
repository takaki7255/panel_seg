"""
LSD & SDF preprocessing for test datasets (without train/val split)

Test datasets have structure:
    dataset/
    ├── images/
    └── masks/

This script processes them and creates:
    output/
    ├── images/
    ├── masks/
    ├── lsd/
    └── sdf/

Usage:
    python preprocess_lsd_sdf_test.py \
        --root ./frame_dataset/test100_dataset \
        --output ./frame_dataset/test100_preprocessed \
        --min-line-length 10
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil


def detect_lines_lsd(image_gray, min_line_length=30):
    """
    LSD (Line Segment Detector) で線分を検出
    
    Args:
        image_gray: グレースケール画像 (numpy array)
        min_line_length: 最小線分長
    
    Returns:
        lines: 検出された線分 [[x1, y1, x2, y2], ...]
    """
    lsd = cv2.createLineSegmentDetector(0)
    lines, _, _, _ = lsd.detect(image_gray)
    
    if lines is None:
        return np.array([])
    
    lines = lines.reshape(-1, 4)
    
    # 最小線分長でフィルタリング
    lengths = np.sqrt((lines[:, 2] - lines[:, 0])**2 + (lines[:, 3] - lines[:, 1])**2)
    lines = lines[lengths >= min_line_length]
    
    return lines


def create_lsd_map(image_shape, lines, thickness=2):
    """
    LSD線分からバイナリマップを作成
    
    Args:
        image_shape: 画像サイズ (H, W)
        lines: 線分配列 [[x1, y1, x2, y2], ...]
        thickness: 線の太さ
    
    Returns:
        lsd_map: LSD線分マップ (0-255)
    """
    h, w = image_shape
    lsd_map = np.zeros((h, w), dtype=np.uint8)
    
    for line in lines:
        x1, y1, x2, y2 = line.astype(int)
        cv2.line(lsd_map, (x1, y1), (x2, y2), 255, thickness)
    
    return lsd_map


def compute_sdf_from_mask(mask):
    """
    マスクから符号付き距離場 (SDF) を計算
    
    Args:
        mask: バイナリマスク (0 or 255)
    
    Returns:
        sdf_normalized: 正規化されたSDF (0-255)
    """
    mask_binary = (mask > 127).astype(np.uint8)
    
    # 内側の距離変換（正の距離）
    dist_inside = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
    
    # 外側の距離変換（負の距離）
    dist_outside = cv2.distanceTransform(1 - mask_binary, cv2.DIST_L2, 5)
    
    # 符号付き距離場: 内側が正、外側が負
    sdf = dist_inside - dist_outside
    
    # 正規化 (0-255)
    sdf_min, sdf_max = sdf.min(), sdf.max()
    if sdf_max > sdf_min:
        sdf_normalized = ((sdf - sdf_min) / (sdf_max - sdf_min) * 255).astype(np.uint8)
    else:
        sdf_normalized = np.zeros_like(sdf, dtype=np.uint8)
    
    return sdf_normalized


def preprocess_test_dataset(root_dir, output_dir, min_line_length=30, lsd_thickness=2):
    """
    テストデータセットを前処理
    
    Args:
        root_dir: 入力データセットのルート (images/, masks/ を含む)
        output_dir: 出力ディレクトリ
        min_line_length: LSD最小線分長
        lsd_thickness: LSD線の太さ
    """
    root_path = Path(root_dir)
    output_path = Path(output_dir)
    
    # ディレクトリ構造確認
    images_dir = root_path / "images"
    masks_dir = root_path / "masks"
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")
    
    # 出力ディレクトリ作成
    output_images_dir = output_path / "images"
    output_masks_dir = output_path / "masks"
    output_lsd_dir = output_path / "lsd"
    output_sdf_dir = output_path / "sdf"
    
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_masks_dir.mkdir(parents=True, exist_ok=True)
    output_lsd_dir.mkdir(parents=True, exist_ok=True)
    output_sdf_dir.mkdir(parents=True, exist_ok=True)
    
    # 画像ファイル一覧取得
    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    
    if len(image_files) == 0:
        raise FileNotFoundError(f"No images found in {images_dir}")
    
    print(f"\n{'='*60}")
    print(f"LSD & SDF Preprocessing (Test Dataset)")
    print(f"{'='*60}")
    print(f"Input: {root_dir}")
    print(f"Output: {output_dir}")
    print(f"Images found: {len(image_files)}")
    print(f"Min line length: {min_line_length}")
    print(f"LSD thickness: {lsd_thickness}")
    print(f"{'='*60}\n")
    
    success_count = 0
    failed_files = []
    
    for img_path in tqdm(image_files, desc="Processing"):
        try:
            stem = img_path.stem
            mask_path = masks_dir / f"{stem}_mask.png"
            
            if not mask_path.exists():
                print(f"\n  Warning: Mask not found for {img_path.name}, skipping...")
                failed_files.append((img_path.name, "Mask not found"))
                continue
            
            # 画像読み込み
            img = cv2.imread(str(img_path))
            if img is None:
                failed_files.append((img_path.name, "Failed to load image"))
                continue
            
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # マスク読み込み
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                failed_files.append((img_path.name, "Failed to load mask"))
                continue
            
            # LSD検出
            lines = detect_lines_lsd(img_gray, min_line_length=min_line_length)
            lsd_map = create_lsd_map(img_gray.shape, lines, thickness=lsd_thickness)
            
            # SDF計算
            sdf_map = compute_sdf_from_mask(mask)
            
            # 保存
            shutil.copy(img_path, output_images_dir / img_path.name)
            shutil.copy(mask_path, output_masks_dir / mask_path.name)
            cv2.imwrite(str(output_lsd_dir / f"{stem}_lsd.png"), lsd_map)
            cv2.imwrite(str(output_sdf_dir / f"{stem}_sdf.png"), sdf_map)
            
            success_count += 1
            
        except Exception as e:
            failed_files.append((img_path.name, str(e)))
            print(f"\n  Error processing {img_path.name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Preprocessing Complete")
    print(f"{'='*60}")
    print(f"Success: {success_count}/{len(image_files)}")
    
    if failed_files:
        print(f"\nFailed files ({len(failed_files)}):")
        for filename, reason in failed_files:
            print(f"  - {filename}: {reason}")
    
    print(f"\nOutput saved to: {output_dir}")
    print(f"  - images/: {len(list(output_images_dir.glob('*')))}")
    print(f"  - masks/: {len(list(output_masks_dir.glob('*')))}")
    print(f"  - lsd/: {len(list(output_lsd_dir.glob('*')))}")
    print(f"  - sdf/: {len(list(output_sdf_dir.glob('*')))}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess test dataset with LSD and SDF features"
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory of test dataset (contains images/ and masks/)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for preprocessed data"
    )
    parser.add_argument(
        "--min-line-length",
        type=int,
        default=30,
        help="Minimum line length for LSD detection (default: 30)"
    )
    parser.add_argument(
        "--lsd-thickness",
        type=int,
        default=2,
        help="Thickness of LSD lines (default: 2)"
    )
    
    args = parser.parse_args()
    
    preprocess_test_dataset(
        root_dir=args.root,
        output_dir=args.output,
        min_line_length=args.min_line_length,
        lsd_thickness=args.lsd_thickness
    )


if __name__ == "__main__":
    main()
