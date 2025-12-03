"""
Manga109 COCO形式アノテーションからインスタンスセグメンテーション用データセットを作成
RLEマスクを直接使用し、train/val/test分割を行う
LSD（線分検出）とSDF（距離場）も生成

Usage:
    python create_instance_dataset.py --name my_dataset --total 1000
    python create_instance_dataset.py --name full_dataset  # 全画像使用
    python create_instance_dataset.py --name my_dataset --total 1000 --no-lsd-sdf  # LSD/SDFなし
"""

import os
import json
import random
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
import cv2
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

# ===== 設定 =====
ANNOTATION_ROOT = "../manga109/manga_seg_jsons"
IMAGE_ROOT = "../manga109/images/"
OUTPUT_ROOT = "./instance_dataset"

# カテゴリフィルタ（Noneの場合は全カテゴリ、リストで指定可能）
TARGET_CATEGORIES = ["frame"]  # frame（コマ）のみ抽出

# 分割比率（train/valのみ、testは別途作成）
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2
TEST_RATIO = 0.0  # テストは別途作成

# ランダムシード
RANDOM_SEED = 42

# LSD/SDF設定
LSD_MIN_LENGTH = 10  # 最小線分長
LSD_MAX_LENGTH = None  # 最大線分長 (Noneで無制限)
SDF_MAX_DIST = 50  # SDF正規化の最大距離


def parse_args():
    parser = argparse.ArgumentParser(
        description="Manga109からインスタンスセグメンテーション用データセットを作成"
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        required=True,
        help="データセット名（出力フォルダ名）"
    )
    parser.add_argument(
        "--total", "-t",
        type=int,
        default=None,
        help="使用する総画像数（省略時は全画像を使用）"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=OUTPUT_ROOT,
        help=f"出力先ディレクトリ（デフォルト: {OUTPUT_ROOT}）"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=RANDOM_SEED,
        help=f"ランダムシード（デフォルト: {RANDOM_SEED}）"
    )
    parser.add_argument(
        "--no-lsd-sdf",
        action="store_true",
        help="LSD/SDFの生成をスキップ"
    )
    parser.add_argument(
        "--min-line-length",
        type=float,
        default=LSD_MIN_LENGTH,
        help=f"LSD最小線分長（デフォルト: {LSD_MIN_LENGTH}）"
    )
    parser.add_argument(
        "--sdf-max-dist",
        type=int,
        default=SDF_MAX_DIST,
        help=f"SDF正規化の最大距離（デフォルト: {SDF_MAX_DIST}）"
    )
    parser.add_argument(
        "--test-name",
        type=str,
        default=None,
        help="テストデータセット名（指定時は別途作成、未指定時はスキップ）"
    )
    parser.add_argument(
        "--test-total",
        type=int,
        default=100,
        help="テストデータセットの画像数（デフォルト: 100）"
    )
    return parser.parse_args()


# ============================================================================
# LSD / SDF Functions
# ============================================================================
def detect_lines_opencv(image_path, min_length=10, max_length=None):
    """
    OpenCV LSDを使用して線分を検出
    
    Args:
        image_path: 入力画像のパス
        min_length: 最小線分長
        max_length: 最大線分長 (Noneで無制限)
    
    Returns:
        line_map: (H, W) 線分が描画されたnumpy配列 (0-255)
        num_lines: 検出された線分数
    """
    # グレースケールで読み込み
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    
    # LSD検出器を作成
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    
    # 線分検出
    lines, width, prec, nfa = lsd.detect(img)
    
    # 長さでフィルタリング
    if lines is not None:
        filtered = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            if max_length is None:
                if length >= min_length:
                    filtered.append(line)
            else:
                if min_length <= length <= max_length:
                    filtered.append(line)
        
        lines = np.array(filtered) if filtered else None
    
    # 線分を描画
    line_map = np.zeros_like(img)
    num_lines = 0
    
    if lines is not None:
        num_lines = len(lines)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_map, (int(x1), int(y1)), (int(x2), int(y2)), 255, 1)
    
    return line_map, num_lines


def compute_sdf(line_map, max_distance=50):
    """
    線分マップからSDF（Signed Distance Field）を計算
    
    Args:
        line_map: (H, W) 線分が白(255)のバイナリマップ
        max_distance: 正規化の最大距離
    
    Returns:
        sdf_map: (H, W) 正規化された距離マップ [0, 255]
                 線に近いほど白(255)、遠いほど黒(0)
    """
    # バイナリ化
    binary_map = (line_map > 127).astype(np.uint8)
    
    # 距離変換（最も近い線分への距離を計算）
    distance = distance_transform_edt(1 - binary_map)
    
    # 正規化
    distance_normalized = np.clip(distance / max_distance, 0, 1)
    
    # [0, 255]に変換し、反転（線に近いほど白、遠いほど黒）
    sdf_map = (255 - distance_normalized * 255).astype(np.uint8)
    
    return sdf_map


def load_all_annotations(annotation_root):
    """
    全てのJSONファイルからアノテーションを読み込む
    
    Returns:
        dict: 統合されたCOCO形式データ
        dict: 作品名からimage_idへのマッピング
    """
    all_images = []
    all_annotations = []
    categories = None
    
    image_id_counter = 0
    annotation_id_counter = 0
    
    # 作品名 -> image_id マッピング
    title_to_images = defaultdict(list)
    
    json_files = sorted(Path(annotation_root).glob("*.json"))
    print(f"Found {len(json_files)} annotation files")
    
    for json_path in json_files:
        title = json_path.stem  # ファイル名（拡張子なし）= 作品名
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # カテゴリ情報を初回のみ取得
        if categories is None:
            categories = data.get('categories', [])
        
        # 元のimage_id -> 新しいimage_idのマッピング
        old_to_new_image_id = {}
        
        # 画像情報を追加
        for img in data.get('images', []):
            old_id = img['id']
            new_id = image_id_counter
            image_id_counter += 1
            
            old_to_new_image_id[old_id] = new_id
            
            # 画像パスを更新
            new_img = img.copy()
            new_img['id'] = new_id
            new_img['title'] = title  # 作品名を追加
            # file_nameは既に "タイトル/ファイル名" の形式になっているのでそのまま使用
            
            all_images.append(new_img)
            title_to_images[title].append(new_id)
        
        # アノテーション情報を追加
        for ann in data.get('annotations', []):
            new_ann = ann.copy()
            new_ann['id'] = annotation_id_counter
            annotation_id_counter += 1
            new_ann['image_id'] = old_to_new_image_id[ann['image_id']]
            
            all_annotations.append(new_ann)
    
    combined_data = {
        'images': all_images,
        'annotations': all_annotations,
        'categories': categories
    }
    
    return combined_data, dict(title_to_images)


def filter_by_category(data, target_categories):
    """
    指定カテゴリのアノテーションのみをフィルタリング
    """
    if target_categories is None:
        return data
    
    # カテゴリ名からIDへのマッピング
    cat_name_to_id = {cat['name']: cat['id'] for cat in data['categories']}
    target_ids = [cat_name_to_id[name] for name in target_categories if name in cat_name_to_id]
    
    if not target_ids:
        print(f"Warning: No matching categories found for {target_categories}")
        return data
    
    print(f"Filtering annotations for categories: {target_categories} (IDs: {target_ids})")
    
    # アノテーションをフィルタリング
    filtered_annotations = [
        ann for ann in data['annotations'] 
        if ann['category_id'] in target_ids
    ]
    
    # 対応する画像IDを取得
    image_ids_with_annotations = set(ann['image_id'] for ann in filtered_annotations)
    
    # 画像をフィルタリング
    filtered_images = [
        img for img in data['images']
        if img['id'] in image_ids_with_annotations
    ]
    
    # カテゴリをフィルタリング
    filtered_categories = [
        cat for cat in data['categories']
        if cat['id'] in target_ids
    ]
    
    return {
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': filtered_categories
    }


def split_random(all_image_ids, train_ratio, val_ratio, seed=42, total_images=None):
    """
    全画像からランダムにtrain/valに分割
    
    Args:
        all_image_ids: 全画像IDのリスト
        train_ratio: 訓練データの比率
        val_ratio: 検証データの比率
        seed: ランダムシード
        total_images: 使用する総画像数（Noneの場合は全画像を使用）
    
    Returns:
        dict: 各splitに含まれるimage_idのリスト
        list: 使用したimage_idのリスト
    """
    random.seed(seed)
    
    # 画像IDをシャッフル
    image_ids = list(all_image_ids)
    random.shuffle(image_ids)
    
    # total_imagesが指定されている場合、画像数を制限
    if total_images is not None:
        image_ids = image_ids[:total_images]
        print(f"\nLimited to {len(image_ids)} images")
    
    n_images = len(image_ids)
    n_train = int(n_images * train_ratio)
    
    splits = {
        'train': image_ids[:n_train],
        'val': image_ids[n_train:]
    }
    
    print(f"\nRandom split:")
    print(f"  Train: {len(splits['train'])} images")
    print(f"  Val: {len(splits['val'])} images")
    print(f"  Total: {len(splits['train']) + len(splits['val'])} images")
    
    return splits, image_ids


def create_split_dataset(data, image_ids, output_dir, image_root, split_name, 
                         generate_lsd_sdf=True, min_line_length=10, sdf_max_dist=50,
                         test_only=False):
    """
    指定されたimage_idsのデータでCOCO形式データセットを作成
    PNG形式のinstance_masksも同時に生成
    
    Args:
        test_only: Trueの場合、split_nameを無視してルート直下に作成
    """
    image_id_set = set(image_ids)
    
    # 画像をフィルタリング
    split_images = [img for img in data['images'] if img['id'] in image_id_set]
    
    # アノテーションをフィルタリング
    split_annotations = [ann for ann in data['annotations'] if ann['image_id'] in image_id_set]
    
    # 新しいIDを振り直す
    old_to_new_image_id = {}
    new_images = []
    
    for i, img in enumerate(split_images):
        old_to_new_image_id[img['id']] = i
        new_img = img.copy()
        new_img['id'] = i
        new_images.append(new_img)
    
    new_annotations = []
    for i, ann in enumerate(split_annotations):
        new_ann = ann.copy()
        new_ann['id'] = i
        new_ann['image_id'] = old_to_new_image_id[ann['image_id']]
        new_annotations.append(new_ann)
    
    split_data = {
        'images': new_images,
        'annotations': new_annotations,
        'categories': data['categories']
    }
    
    # 出力ディレクトリ作成（test_onlyの場合はルート直下）
    if test_only:
        split_output = Path(output_dir)
    else:
        split_output = Path(output_dir) / split_name
    images_dir = split_output / "images"
    instance_masks_dir = split_output / "instance_masks"  # PNG形式マスク用
    images_dir.mkdir(parents=True, exist_ok=True)
    instance_masks_dir.mkdir(parents=True, exist_ok=True)
    
    # LSD/SDFディレクトリ作成
    if generate_lsd_sdf:
        lsd_dir = split_output / "lsd"
        sdf_dir = split_output / "sdf"
        lsd_dir.mkdir(parents=True, exist_ok=True)
        sdf_dir.mkdir(parents=True, exist_ok=True)
    
    # image_id -> annotations マッピングを作成
    img_to_anns = defaultdict(list)
    for ann in new_annotations:
        img_to_anns[ann['image_id']].append(ann)
    
    # 画像をコピーし、LSD/SDF、instance_masksを生成
    copied_count = 0
    lsd_stats = []
    
    desc_label = split_name if split_name else "test"
    for img in tqdm(new_images, desc=f"  {desc_label}", unit="img"):
        src_path = Path(image_root) / img['file_name']
        # ファイル名を連番に変更
        dst_filename = f"{img['id']:05d}.jpg"
        dst_path = images_dir / dst_filename
        
        # アノテーション内のファイル名も更新
        img['file_name'] = dst_filename
        
        if src_path.exists():
            shutil.copy(src_path, dst_path)
            copied_count += 1
            
            # 画像サイズを取得（instance_mask生成用）
            src_img = cv2.imread(str(src_path))
            if src_img is not None:
                img_h, img_w = src_img.shape[:2]
                
                # ===== instance_masks生成（PNG形式） =====
                # 各インスタンスに異なるIDを割り当てたマスクを作成
                instance_mask = np.zeros((img_h, img_w), dtype=np.uint16)
                anns = img_to_anns[img['id']]
                
                for inst_idx, ann in enumerate(anns, start=1):
                    if isinstance(ann['segmentation'], dict):
                        # RLEをデコード
                        rle = ann['segmentation']
                        mask = mask_utils.decode(rle)
                        # インスタンスIDを割り当て
                        instance_mask[mask > 0] = inst_idx
                
                # PNG形式で保存（16bit対応）
                instance_mask_filename = f"{img['id']:05d}_instance.png"
                cv2.imwrite(str(instance_masks_dir / instance_mask_filename), instance_mask)
            
            # LSD/SDF生成
            if generate_lsd_sdf:
                try:
                    # LSD検出
                    line_map, num_lines = detect_lines_opencv(
                        dst_path, 
                        min_length=min_line_length
                    )
                    lsd_stats.append(num_lines)
                    
                    # LSD保存
                    lsd_filename = f"{img['id']:05d}_lsd.png"
                    cv2.imwrite(str(lsd_dir / lsd_filename), line_map)
                    
                    # SDF計算・保存
                    sdf_map = compute_sdf(line_map, max_distance=sdf_max_dist)
                    sdf_filename = f"{img['id']:05d}_sdf.png"
                    cv2.imwrite(str(sdf_dir / sdf_filename), sdf_map)
                    
                except Exception as e:
                    print(f"Warning: LSD/SDF generation failed for {dst_filename}: {e}")
        else:
            print(f"Warning: Image not found: {src_path}")
    
    # アノテーションファイルを保存
    annotation_file = split_output / "annotations.json"
    with open(annotation_file, 'w') as f:
        json.dump(split_data, f)
    
    print(f"  {split_name}: {len(new_images)} images, {len(new_annotations)} annotations, {copied_count} images copied")
    print(f"    instance_masks: {copied_count} PNG files generated")
    
    if generate_lsd_sdf and lsd_stats:
        print(f"    LSD: avg {np.mean(lsd_stats):.1f} ± {np.std(lsd_stats):.1f} lines/image")
    
    return split_data


def create_visualization_samples(split_data, output_dir, split_name, num_samples=5):
    """
    可視化用サンプルを作成
    """
    try:
        import cv2
        from PIL import Image
    except ImportError:
        print("  Skipping visualization (cv2 or PIL not available)")
        return
    
    split_output = Path(output_dir) / split_name
    vis_dir = split_output / "visualization"
    vis_dir.mkdir(exist_ok=True)
    
    # ランダムに画像を選択
    images = split_data['images']
    if len(images) < num_samples:
        sample_images = images
    else:
        sample_images = random.sample(images, num_samples)
    
    # image_id -> annotations マッピング
    img_to_anns = defaultdict(list)
    for ann in split_data['annotations']:
        img_to_anns[ann['image_id']].append(ann)
    
    for img_info in sample_images:
        img_path = split_output / "images" / img_info['file_name']
        if not img_path.exists():
            continue
        
        # 画像読み込み
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # 各インスタンスに色を付けてマスクを描画
        overlay = img.copy()
        
        anns = img_to_anns[img_info['id']]
        for i, ann in enumerate(anns):
            # RLEマスクをデコード
            if isinstance(ann['segmentation'], dict):
                rle = ann['segmentation']
                mask = mask_utils.decode(rle)
                
                # ランダムな色を生成
                color = [int(c) for c in np.random.randint(0, 255, 3)]
                
                # マスクを重ねる
                overlay[mask > 0] = color
        
        # 半透明で合成
        result = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
        
        # 保存
        vis_path = vis_dir / f"vis_{img_info['id']:05d}.jpg"
        cv2.imwrite(str(vis_path), result)
    
    print(f"  Visualization samples saved to {vis_dir}")


def print_statistics(data):
    """
    データセットの統計情報を表示
    """
    print("\n=== Dataset Statistics ===")
    print(f"Total images: {len(data['images'])}")
    print(f"Total annotations: {len(data['annotations'])}")
    print(f"Categories: {[cat['name'] for cat in data['categories']]}")
    
    # カテゴリ別アノテーション数
    cat_counts = defaultdict(int)
    cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    
    for ann in data['annotations']:
        cat_name = cat_id_to_name.get(ann['category_id'], 'unknown')
        cat_counts[cat_name] += 1
    
    print("\nAnnotations per category:")
    for cat_name, count in sorted(cat_counts.items()):
        print(f"  {cat_name}: {count}")
    
    # 画像あたりのアノテーション数の統計
    anns_per_image = defaultdict(int)
    for ann in data['annotations']:
        anns_per_image[ann['image_id']] += 1
    
    if anns_per_image:
        counts = list(anns_per_image.values())
        print(f"\nAnnotations per image:")
        print(f"  Mean: {np.mean(counts):.2f}")
        print(f"  Min: {min(counts)}")
        print(f"  Max: {max(counts)}")


def main():
    # コマンドライン引数をパース
    args = parse_args()
    
    print("=" * 60)
    print("Instance Segmentation Dataset Creation")
    print("=" * 60)
    print(f"\nDataset name: {args.name}")
    print(f"Total images: {args.total if args.total else 'All'}")
    print(f"Output directory: {args.output}/{args.name}")
    print(f"Random seed: {args.seed}")
    
    # 全アノテーションを読み込み
    print("\n[1] Loading all annotations...")
    data, title_to_images = load_all_annotations(ANNOTATION_ROOT)
    print(f"Loaded {len(data['images'])} images, {len(data['annotations'])} annotations")
    print(f"From {len(title_to_images)} titles")
    
    # カテゴリでフィルタリング
    if TARGET_CATEGORIES:
        print(f"\n[2] Filtering by categories: {TARGET_CATEGORIES}")
        data = filter_by_category(data, TARGET_CATEGORIES)
        
        # title_to_imagesも更新
        valid_image_ids = set(img['id'] for img in data['images'])
        title_to_images = {
            title: [img_id for img_id in img_ids if img_id in valid_image_ids]
            for title, img_ids in title_to_images.items()
        }
        title_to_images = {k: v for k, v in title_to_images.items() if v}
    
    print_statistics(data)
    
    # 全画像からランダムに分割（train/valのみ）
    print(f"\n[3] Splitting dataset randomly...")
    all_image_ids = [img['id'] for img in data['images']]
    splits, used_image_ids = split_random(
        all_image_ids, 
        TRAIN_RATIO, 
        VAL_RATIO, 
        args.seed,
        total_images=args.total
    )
    
    # 出力ディレクトリ作成（output_root/dataset_name）
    output_dir = Path(args.output) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 各splitのデータセットを作成
    print(f"\n[4] Creating train/val datasets...")
    generate_lsd_sdf = not args.no_lsd_sdf
    if generate_lsd_sdf:
        print(f"    LSD/SDF generation: ENABLED")
        print(f"    Min line length: {args.min_line_length}")
        print(f"    SDF max distance: {args.sdf_max_dist}")
    else:
        print(f"    LSD/SDF generation: DISABLED")
    
    for split_name, image_ids in tqdm(splits.items(), desc="Creating splits", unit="split"):
        if not image_ids:
            print(f"  {split_name}: No images (skipped)")
            continue
        
        print(f"\n  Processing {split_name} ({len(image_ids)} images)...")
        split_data = create_split_dataset(
            data, image_ids, output_dir, IMAGE_ROOT, split_name,
            generate_lsd_sdf=generate_lsd_sdf,
            min_line_length=args.min_line_length,
            sdf_max_dist=args.sdf_max_dist
        )
        
        # 可視化サンプル作成
        create_visualization_samples(split_data, output_dir, split_name)
    
    # テストデータセットを別途作成（--test-name指定時）
    test_output_dir = None
    if args.test_name:
        print(f"\n[5] Creating separate test dataset: {args.test_name}...")
        
        # 使用済みでない画像IDから選択
        used_set = set(used_image_ids)
        available_for_test = [img_id for img_id in all_image_ids if img_id not in used_set]
        
        if len(available_for_test) < args.test_total:
            print(f"  Warning: Available images ({len(available_for_test)}) < requested ({args.test_total})")
            test_image_ids = available_for_test
        else:
            random.seed(args.seed + 1)  # 異なるシード
            test_image_ids = random.sample(available_for_test, args.test_total)
        
        print(f"  Test images: {len(test_image_ids)} (from unused pool of {len(available_for_test)})")
        
        # テストデータセット出力先
        test_output_dir = Path(args.output) / args.test_name
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # テストデータセット作成（test_onlyフラグ付き）
        test_data = create_split_dataset(
            data, test_image_ids, test_output_dir, IMAGE_ROOT, "",  # split_name空でルート直下
            generate_lsd_sdf=generate_lsd_sdf,
            min_line_length=args.min_line_length,
            sdf_max_dist=args.sdf_max_dist,
            test_only=True
        )
        
        # テストデータセットの分割情報を保存
        test_split_info = {
            'dataset_name': args.test_name,
            'is_test_only': True,
            'test_images': len(test_image_ids),
            'target_categories': TARGET_CATEGORIES,
            'source_dataset': args.name,
            'random_seed': args.seed + 1,
            'lsd_sdf_enabled': not args.no_lsd_sdf,
            'min_line_length': args.min_line_length,
            'sdf_max_dist': args.sdf_max_dist
        }
        
        with open(test_output_dir / 'split_info.json', 'w') as f:
            json.dump(test_split_info, f, indent=2, ensure_ascii=False)
        
        # テストデータセットの可視化
        create_visualization_samples(test_data, test_output_dir, "")
    
    # 分割情報を保存
    split_info = {
        'dataset_name': args.name,
        'train_images': len(splits['train']),
        'val_images': len(splits['val']),
        'target_categories': TARGET_CATEGORIES,
        'total_images': args.total,
        'ratios': {
            'train': TRAIN_RATIO,
            'val': VAL_RATIO
        },
        'random_seed': args.seed,
        'lsd_sdf_enabled': not args.no_lsd_sdf,
        'min_line_length': args.min_line_length,
        'sdf_max_dist': args.sdf_max_dist,
        'test_dataset': args.test_name if args.test_name else None
    }
    
    with open(output_dir / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n[6] Dataset creation completed!")
    print(f"\n=== Train/Val Dataset ===")
    print(f"Output directory: {output_dir}")
    print(f"\nDirectory structure:")
    print(f"  {output_dir}/")
    print(f"  ├── train/")
    print(f"  │   ├── images/")
    print(f"  │   ├── instance_masks/")
    if not args.no_lsd_sdf:
        print(f"  │   ├── lsd/")
        print(f"  │   ├── sdf/")
    print(f"  │   └── visualization/")
    print(f"  ├── val/")
    print(f"  │   ├── images/")
    print(f"  │   ├── instance_masks/")
    if not args.no_lsd_sdf:
        print(f"  │   ├── lsd/")
        print(f"  │   ├── sdf/")
    print(f"  │   └── visualization/")
    print(f"  └── split_info.json")
    
    if test_output_dir:
        print(f"\n=== Test Dataset (separate) ===")
        print(f"Output directory: {test_output_dir}")
        print(f"\nDirectory structure:")
        print(f"  {test_output_dir}/")
        print(f"  ├── images/")
        print(f"  ├── instance_masks/")
        if not args.no_lsd_sdf:
            print(f"  ├── lsd/")
            print(f"  ├── sdf/")
        print(f"  ├── visualization/")
        print(f"  └── split_info.json")


if __name__ == "__main__":
    main()
