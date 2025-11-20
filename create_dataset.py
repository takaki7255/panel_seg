"""
Manga109_released_2023_12_07 から balloon カテゴリのマスクペアを抽出して
real_dataset（200枚）と test_dataset（100枚）を作成するスクリプト
"""

import os
import random
import shutil
from glob import glob
from pathlib import Path

# ===== 設定 =====
mask_root = "./../Manga109_released_2023_12_07/masks/"         # マスクのルート（masks/作品名/カテゴリ名/*.png）
image_root = "./../Manga109_released_2023_12_07/images"       # 画像のルート
target_category = "frame"                                   # 処理対象カテゴリ
target_num = 1000                                       # 処理対象総数
target_num_test = 0                                   # test_dataset用枚数'

# データセット設定（target_categoryベースで出力先を決定）
datasets = {
    "real_dataset": {
        "output_root": f"./{target_category}_dataset/{target_num}_dataset",
        "num_samples": target_num,
        "structure": "train_val"  # train/val 構造
    },
    "test_dataset": {
        "output_root": f"./{target_category}_dataset/test{target_num_test}_dataset", 
        "num_samples": target_num_test,
        "structure": "test_only"  # test のみ
    }
}

def create_dataset(config, available_masks, used_indices=None):
    """
    データセットを作成する
    
    Args:
        config: データセット設定
        available_masks: 利用可能なマスクパスのリスト
        used_indices: 既に使用済みのインデックスセット
    
    Returns:
        使用したインデックスのセット
    """
    output_root = config["output_root"]
    num_samples = config["num_samples"]
    structure = config["structure"]
    
    print(f"\n=== {output_root} 作成開始 ===")
    
    # 出力フォルダ作成
    if structure == "train_val":
        # train/val 構造で直接作成
        os.makedirs(os.path.join(output_root, "train", "images"), exist_ok=True)
        os.makedirs(os.path.join(output_root, "train", "masks"), exist_ok=True)
        os.makedirs(os.path.join(output_root, "val", "images"), exist_ok=True)
        os.makedirs(os.path.join(output_root, "val", "masks"), exist_ok=True)
    else:  # test_only
        os.makedirs(os.path.join(output_root, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_root, "masks"), exist_ok=True)
    
    # 使用可能なインデックスを計算
    if used_indices is None:
        used_indices = set()
    
    available_indices = [i for i in range(len(available_masks)) if i not in used_indices]
    
    if len(available_indices) < num_samples:
        print(f"警告: 利用可能なマスク数（{len(available_indices)}）が必要数（{num_samples}）より少ないです")
        num_samples = len(available_indices)
    
    # ランダム選択
    selected_indices = random.sample(available_indices, num_samples)
    
    # train/val 分割（8:2）
    if structure == "train_val":
        train_count = int(num_samples * 0.8)
        train_indices = selected_indices[:train_count]
        val_indices = selected_indices[train_count:]
        
        splits = {"train": train_indices, "val": val_indices}
    else:
        splits = {"": selected_indices}  # test_only の場合
    
    success_count = 0
    
    for split_name, indices in splits.items():
        for local_idx, mask_idx in enumerate(indices):
            mask_path = available_masks[mask_idx]
            
            # パスを安全に分解
            class_dir = os.path.dirname(mask_path)                  # e.g., masks/Title/balloon
            title = os.path.basename(os.path.dirname(class_dir))    # "Title"
            mask_filename = os.path.basename(mask_path)             # "000_mask.png"

            # 対応する画像の推定
            img_filename = mask_filename.replace("_mask.png", ".jpg")
            image_path = os.path.join(image_root, title, img_filename)

            # 出力先ファイル名
            if structure == "train_val":
                out_img_path = os.path.join(output_root, split_name, "images", f"{local_idx:03}.jpg")
                out_mask_path = os.path.join(output_root, split_name, "masks", f"{local_idx:03}_mask.png")
            else:  # test_only
                out_img_path = os.path.join(output_root, "images", f"{local_idx:03}.jpg")
                out_mask_path = os.path.join(output_root, "masks", f"{local_idx:03}_mask.png")

            # エラーチェック
            if not os.path.isfile(image_path):
                print(f"[スキップ] 元画像が存在しません: {image_path}")
                continue

            try:
                shutil.copy(image_path, out_img_path)
                shutil.copy(mask_path, out_mask_path)
                if structure == "train_val":
                    print(f"[{split_name}:{local_idx+1:03}] コピー完了: {title}/{img_filename}")
                else:
                    print(f"[{success_count+1:03}] コピー完了: {title}/{img_filename}")
                success_count += 1
            except Exception as e:
                print(f"[エラー] コピー失敗 {image_path}: {e}")

    print(f"✅ {output_root} 作成完了: {success_count}枚")
    
    # 使用したインデックスを返す
    return used_indices.union(set(selected_indices))

def main():
    print(f"Manga109 {target_category} カテゴリからデータセット作成開始")
    
    # 指定カテゴリのマスクを収集
    print(f"{target_category} カテゴリのマスクを検索中...")
    mask_paths = glob(os.path.join(mask_root, "*", target_category, "*_mask.png"))
    
    if not mask_paths:
        print(f"エラー: {target_category} カテゴリのマスクが見つかりません")
        print(f"検索パス: {os.path.join(mask_root, '*', target_category, '*_mask.png')}")
        return
    
    print(f"見つかったマスク数: {len(mask_paths)}")
    print(f"出力先ディレクトリ: ./{target_category}_dataset/")
    
    # 必要な総数をチェック
    total_needed = sum(config["num_samples"] for config in datasets.values())
    if len(mask_paths) < total_needed:
        print(f"警告: 利用可能なマスク数（{len(mask_paths)}）が必要総数（{total_needed}）より少ないです")
    
    # ランダムシード設定
    random.seed(42)
    
    # データセットを順番に作成（重複しないように）
    used_indices = set()
    
    # 1. real_dataset 作成
    used_indices = create_dataset(datasets["real_dataset"], mask_paths, used_indices)
    
    # 2. test_dataset 作成  
    used_indices = create_dataset(datasets["test_dataset"], mask_paths, used_indices)
    
    print(f"\n=== 全体統計 ===")
    print(f"総マスク数: {len(mask_paths)}")
    print(f"使用済み: {len(used_indices)}")
    print(f"残り: {len(mask_paths) - len(used_indices)}")
    
    print(f"\n=== 作成されたデータセット ===")
    for name, config in datasets.items():
        output_root = config["output_root"]
        if config["structure"] == "train_val":
            train_img_count = len(list(Path(output_root).glob("train/images/*.jpg")))
            train_mask_count = len(list(Path(output_root).glob("train/masks/*_mask.png")))
            val_img_count = len(list(Path(output_root).glob("val/images/*.jpg")))
            val_mask_count = len(list(Path(output_root).glob("val/masks/*_mask.png")))
            print(f"{name}:")
            print(f"  train: {train_img_count} 画像, {train_mask_count} マスク")
            print(f"  val:   {val_img_count} 画像, {val_mask_count} マスク")
        else:  # test_only
            img_count = len(list(Path(output_root).glob("images/*.jpg")))
            mask_count = len(list(Path(output_root).glob("masks/*_mask.png")))
            print(f"{name}: {img_count} 画像, {mask_count} マスク")
    
    print("\n✅ 全データセット作成完了")

if __name__ == "__main__":
    main()
