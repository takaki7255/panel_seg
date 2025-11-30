"""
COCO形式データセットを使用したインスタンスセグメンテーション学習・評価スクリプト

Mask R-CNN と Mask2Former に対応

Usage:
    # Mask R-CNN 学習
    python train_instance_seg.py --model maskrcnn --data ./instance_dataset/1000_instance --epochs 50

    # Mask2Former 学習 (100エポック推奨、OneCycleLR使用)
    python train_instance_seg.py --model mask2former --data ./instance_dataset/1000_instance \\
        --epochs 100 --scheduler onecycle --dropout 0.1

    # テストのみ
    python train_instance_seg.py --model maskrcnn --data ./instance_dataset/1000_instance \\
        --test-only --weights ./instance_outputs/maskrcnn/best.pt

Options:
    Mask R-CNN:
        --trainable-layers: バックボーンの学習可能レイヤー数 (0-5, default: 3)
    
    Mask2Former:
        --mask2former-model: 使用するモデル (default: facebook/mask2former-swin-tiny-coco-instance)
        --dropout: ドロップアウト率 (default: 0.0, 推奨: 0.1)
        --scheduler onecycle: OneCycleLRでwarmup付き学習
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, OneCycleLR
from PIL import Image
from tqdm import tqdm
import cv2

# メトリクス計算用
from utils.instance_metrics import (
    compute_instance_metrics,
    MetricsAggregator,
    extract_instance_masks
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ============================================================================
# Dataset
# ============================================================================
class COCOInstanceDataset(torch.utils.data.Dataset):
    """
    インスタンスセグメンテーションデータセット
    PNG形式のinstance_masksから読み込む（高速）
    
    input_type:
        'gray': グレースケール1チャンネル
        '3ch': Gray + LSD + SDF の3チャンネル
    """
    
    def __init__(self, root_dir, split='train', img_size=(512, 512), augment=True, input_type='3ch'):
        self.root = Path(root_dir) / split
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.augment = augment and (split == 'train')
        self.input_type = input_type
        
        self.img_dir = self.root / 'images'
        self.instance_masks_dir = self.root / 'instance_masks'
        
        # PNG instance_masks が存在するか確認
        if not self.instance_masks_dir.exists():
            raise FileNotFoundError(
                f"instance_masks directory not found: {self.instance_masks_dir}\n"
                "Please regenerate dataset with updated create_instance_dataset.py"
            )
        
        # 画像リストを取得（instance_masksから逆引き）
        self.samples = []
        for mask_path in sorted(self.instance_masks_dir.glob('*_instance.png')):
            # 00001_instance.png -> 00001
            base_name = mask_path.stem.replace('_instance', '')
            img_path = self.img_dir / f"{base_name}.jpg"
            if img_path.exists():
                self.samples.append({
                    'id': int(base_name),
                    'img_path': img_path,
                    'mask_path': mask_path,
                    'base_name': base_name
                })
        
        # LSD/SDFディレクトリ (3chの場合)
        self.lsd_dir = self.root / 'lsd'
        self.sdf_dir = self.root / 'sdf'
        
        # 3chモードでLSD/SDFが存在するか確認
        if self.input_type == '3ch':
            if not self.lsd_dir.exists() or not self.sdf_dir.exists():
                print(f"  Warning: LSD/SDF directories not found, falling back to gray mode")
                self.input_type = 'gray'
        
        # カテゴリIDは1のみ（frame）
        self.cat_ids = [1]
        
        print(f"  {split}: {len(self.samples)} images (PNG instance_masks), input={self.input_type}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_id = sample['id']
        base_name = sample['base_name']
        
        # 画像読み込み
        img = Image.open(sample['img_path']).convert('L')  # グレースケールで読み込み
        orig_w, orig_h = img.size
        
        # リサイズ
        img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
        gray_np = np.array(img, dtype=np.float32) / 255.0
        
        if self.input_type == '3ch':
            # LSD読み込み
            lsd_path = self.lsd_dir / f"{base_name}_lsd.png"
            if lsd_path.exists():
                lsd = Image.open(lsd_path).convert('L')
                lsd = lsd.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
                lsd_np = np.array(lsd, dtype=np.float32) / 255.0
            else:
                lsd_np = np.zeros_like(gray_np)
            
            # SDF読み込み
            sdf_path = self.sdf_dir / f"{base_name}_sdf.png"
            if sdf_path.exists():
                sdf = Image.open(sdf_path).convert('L')
                sdf = sdf.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
                sdf_np = np.array(sdf, dtype=np.float32) / 255.0
            else:
                sdf_np = np.zeros_like(gray_np)
            
            # 3チャンネルに結合 (Gray, LSD, SDF)
            img_np = np.stack([gray_np, lsd_np, sdf_np], axis=0)  # (3, H, W)
        else:
            # グレースケールを3チャンネルに複製（事前学習済みモデル用）
            img_np = np.stack([gray_np, gray_np, gray_np], axis=0)  # (3, H, W)
        
        # スケール計算
        scale_x = self.img_size[1] / orig_w
        scale_y = self.img_size[0] / orig_h
        
        # ===== PNG instance_masks から読み込み（高速） =====
        instance_mask = cv2.imread(str(sample['mask_path']), cv2.IMREAD_UNCHANGED)
        
        # 元サイズでインスタンスIDを取得
        instance_ids = np.unique(instance_mask)
        instance_ids = instance_ids[instance_ids > 0]  # 0（背景）を除外
        
        boxes = []
        masks = []
        labels = []
        areas = []
        
        for inst_id in instance_ids:
            # 各インスタンスのマスクを抽出
            mask = (instance_mask == inst_id).astype(np.uint8)
            
            # マスクをリサイズ
            mask_resized = cv2.resize(
                mask, 
                (self.img_size[1], self.img_size[0]),
                interpolation=cv2.INTER_NEAREST
            )
            
            # バウンディングボックス計算
            pos = np.where(mask_resized > 0)
            if len(pos[0]) == 0:
                continue
            
            ymin, ymax = pos[0].min(), pos[0].max()
            xmin, xmax = pos[1].min(), pos[1].max()
            
            if xmax <= xmin or ymax <= ymin:
                continue
            
            boxes.append([xmin, ymin, xmax, ymax])
            masks.append(mask_resized)
            labels.append(1)  # 全てframeカテゴリ (ID=1)
            areas.append((xmax - xmin) * (ymax - ymin))
        
        # データ拡張
        if self.augment and len(boxes) > 0:
            if np.random.random() > 0.5:
                # 水平反転
                img_np = np.flip(img_np, axis=2).copy()
                masks = [np.fliplr(m).copy() for m in masks]
                boxes = [[self.img_size[1] - b[2], b[1], self.img_size[1] - b[0], b[3]] for b in boxes]
        
        # テンソル変換
        image = torch.from_numpy(img_np).float()
        
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            masks = torch.zeros((0, self.img_size[0], self.img_size[1]), dtype=torch.uint8)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([img_id]),
            'area': areas,
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64),
            'orig_size': torch.tensor([orig_h, orig_w]),
        }
        
        return image, target


def collate_fn(batch):
    """Mask R-CNN用のcollate関数"""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


# ============================================================================
# Model Creation
# ============================================================================
def create_maskrcnn(num_classes, pretrained=True, trainable_backbone_layers=3):
    """Mask R-CNNモデルを作成"""
    import torchvision
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    
    # 事前学習済みモデル
    if pretrained:
        model = maskrcnn_resnet50_fpn_v2(
            weights='DEFAULT',
            trainable_backbone_layers=trainable_backbone_layers
        )
    else:
        model = maskrcnn_resnet50_fpn_v2(
            weights=None,
            trainable_backbone_layers=trainable_backbone_layers
        )
    
    # 分類器を置き換え
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # マスク予測器を置き換え
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    
    return model


def create_mask2former(num_classes, pretrained=True, dropout=0.0, model_name='facebook/mask2former-swin-tiny-coco-instance'):
    """Mask2Formerモデルを作成（Transformers使用）"""
    try:
        from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig
        
        if pretrained:
            model = Mask2FormerForUniversalSegmentation.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            config = Mask2FormerConfig(num_labels=num_classes)
            model = Mask2FormerForUniversalSegmentation(config)
        
        # Dropout設定
        # 注意: Mask2Formerではconfig.dropoutを変更すると
        # attention maskサイズの不整合エラーが発生する可能性があります
        # 代わりにweight_decayを使用することを推奨
        if dropout > 0:
            print(f"Warning: Mask2Former dropout設定は互換性の問題があります。weight_decayの使用を推奨。")
            # model.config.dropout = dropout  # 無効化
        
        return model, model_name
    except ImportError:
        raise ImportError("transformersライブラリが必要です: pip install transformers")


# ============================================================================
# Training
# ============================================================================
def train_one_epoch_maskrcnn(model, loader, optimizer, device, epoch):
    """Mask R-CNNの1エポック学習"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += losses.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{losses.item():.4f}',
            'cls': f'{loss_dict.get("loss_classifier", torch.tensor(0)).item():.3f}',
            'mask': f'{loss_dict.get("loss_mask", torch.tensor(0)).item():.3f}'
        })
    
    return total_loss / max(num_batches, 1)


def train_one_epoch_mask2former(model, loader, optimizer, device, epoch, scheduler=None, model_name='facebook/mask2former-swin-tiny-coco-instance'):
    """Mask2Formerの1エポック学習"""
    from transformers import Mask2FormerImageProcessor
    
    model.train()
    total_loss = 0
    num_batches = 0
    
    processor = Mask2FormerImageProcessor.from_pretrained(model_name)
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    
    for images, targets in pbar:
        # 画像をリストからバッチに変換
        batch_images = torch.stack(images).to(device)
        
        # マスクとラベルを準備
        batch_masks = []
        batch_labels = []
        
        for target in targets:
            masks = target['masks'].numpy()
            labels = target['labels'].numpy()
            batch_masks.append(masks)
            batch_labels.append(labels)
        
        # Mask2Former用の入力を準備
        inputs = processor(
            images=[img.permute(1, 2, 0).numpy() * 255 for img in images],
            segmentation_maps=batch_masks,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate_maskrcnn(model, loader, device):
    """Mask R-CNNの検証"""
    model.train()  # 損失計算のためtrainモード
    total_loss = 0
    num_batches = 0
    
    for images, targets in tqdm(loader, desc="Validating"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        total_loss += losses.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate_mask2former(model, loader, device, model_name='facebook/mask2former-swin-tiny-coco-instance'):
    """Mask2Formerの検証"""
    from transformers import Mask2FormerImageProcessor
    
    model.eval()
    total_loss = 0
    num_batches = 0
    
    processor = Mask2FormerImageProcessor.from_pretrained(model_name)
    
    for images, targets in tqdm(loader, desc="Validating"):
        # マスクとラベルを準備
        batch_masks = []
        batch_labels = []
        
        for target in targets:
            masks = target['masks'].numpy()
            labels = target['labels'].numpy()
            batch_masks.append(masks)
            batch_labels.append(labels)
        
        # Mask2Former用の入力を準備
        inputs = processor(
            images=[img.permute(1, 2, 0).numpy() * 255 for img in images],
            segmentation_maps=batch_masks,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = model(**inputs)
        loss = outputs.loss
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)


# ============================================================================
# Evaluation (Instance Metrics)
# ============================================================================
@torch.no_grad()
def evaluate_instance_metrics(model, loader, device, model_type='maskrcnn', model_name='facebook/mask2former-swin-tiny-coco-instance'):
    """
    インスタンスセグメンテーションの評価（既存形式と同じメトリクス出力）
    
    Returns:
        dict: precision, recall, f1, IoU, AP等のメトリクス（mean/std付き）
    """
    model.eval()
    aggregator = MetricsAggregator()
    
    print("\nRunning evaluation...")
    for images, targets in tqdm(loader, desc="Evaluating"):
        images_device = [img.to(device) for img in images]
        
        if model_type == 'maskrcnn':
            outputs = model(images_device)
            
            for output, target in zip(outputs, targets):
                # 予測マスクを抽出
                pred_masks = []
                pred_scores = []
                
                masks = output['masks'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                
                for i in range(len(masks)):
                    if scores[i] < 0.5:  # 閾値
                        continue
                    mask = (masks[i, 0] > 0.5).astype(np.uint8)
                    pred_masks.append(mask)
                    pred_scores.append(scores[i])
                
                pred_scores = np.array(pred_scores) if pred_scores else np.array([])
                
                # GTマスクを抽出
                gt_masks = []
                gt_masks_tensor = target['masks'].cpu().numpy()
                for i in range(len(gt_masks_tensor)):
                    gt_masks.append(gt_masks_tensor[i].astype(np.uint8))
                
                # メトリクス計算
                metrics = compute_instance_metrics(
                    pred_masks, gt_masks, pred_scores, iou_threshold=0.5
                )
                aggregator.add(metrics)
        
        elif model_type == 'mask2former':
            # Mask2Formerの出力処理
            from transformers import Mask2FormerImageProcessor
            processor = Mask2FormerImageProcessor.from_pretrained(model_name)
            
            # バッチ処理
            batch_images = torch.stack(images).to(device)
            outputs = model(pixel_values=batch_images)
            
            for idx, target in enumerate(targets):
                # 後処理
                result = processor.post_process_instance_segmentation(
                    outputs, 
                    target_sizes=[(images[idx].shape[1], images[idx].shape[2])]
                )[0]
                
                pred_masks = []
                pred_scores = []
                
                if 'segments_info' in result:
                    segmentation = result['segmentation'].cpu().numpy()
                    for seg_info in result['segments_info']:
                        mask = (segmentation == seg_info['id']).astype(np.uint8)
                        pred_masks.append(mask)
                        pred_scores.append(seg_info.get('score', 1.0))
                
                pred_scores = np.array(pred_scores) if pred_scores else np.array([])
                
                # GTマスク
                gt_masks = []
                gt_masks_tensor = target['masks'].cpu().numpy()
                for i in range(len(gt_masks_tensor)):
                    gt_masks.append(gt_masks_tensor[i].astype(np.uint8))
                
                metrics = compute_instance_metrics(
                    pred_masks, gt_masks, pred_scores, iou_threshold=0.5
                )
                aggregator.add(metrics)
    
    # サマリー計算
    summary = aggregator.compute_summary()
    aggregator.print_summary()
    
    return summary


def save_visualization(model, loader, device, output_dir, model_type='maskrcnn', model_name=None, num_samples=10):
    """予測結果の可視化を保存"""
    model.eval()
    vis_dir = Path(output_dir) / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    # Mask2Former用プロセッサ
    processor = None
    if model_type == 'mask2former':
        from transformers import Mask2FormerImageProcessor
        processor = Mask2FormerImageProcessor.from_pretrained(
            model_name or 'facebook/mask2former-swin-tiny-coco-instance'
        )
    
    count = 0
    with torch.no_grad():
        for images, targets in loader:
            if count >= num_samples:
                break
            
            images_device = [img.to(device) for img in images]
            
            if model_type == 'maskrcnn':
                outputs = model(images_device)
                
                for img, output, target in zip(images, outputs, targets):
                    if count >= num_samples:
                        break
                    
                    # 画像をnumpyに変換
                    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    overlay = img_np.copy()
                    
                    # 予測マスクを描画
                    masks = output['masks'].cpu().numpy()
                    scores = output['scores'].cpu().numpy()
                    
                    for i in range(len(masks)):
                        if scores[i] < 0.5:
                            continue
                        
                        mask = masks[i, 0] > 0.5
                        color = np.random.randint(0, 255, 3).tolist()
                        overlay[mask] = color
                        
                        # 境界線を描画
                        contours, _ = cv2.findContours(
                            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
                    
                    result = cv2.addWeighted(img_np, 0.4, overlay, 0.6, 0)
                    
                    # 保存
                    img_id = target['image_id'].item()
                    cv2.imwrite(str(vis_dir / f'pred_{img_id:05d}.jpg'), result)
                    count += 1
                    
            elif model_type == 'mask2former':
                # Mask2Formerの出力処理
                batch_images = torch.stack(images).to(device)
                outputs = model(pixel_values=batch_images)
                
                for idx, (img, target) in enumerate(zip(images, targets)):
                    if count >= num_samples:
                        break
                    
                    # 後処理
                    result = processor.post_process_instance_segmentation(
                        outputs, 
                        target_sizes=[(img.shape[1], img.shape[2])]
                    )[0]
                    
                    # 画像をnumpyに変換
                    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    overlay = img_np.copy()
                    
                    if 'segments_info' in result:
                        segmentation = result['segmentation'].cpu().numpy()
                        for seg_info in result['segments_info']:
                            mask = (segmentation == seg_info['id']).astype(np.uint8)
                            color = np.random.randint(0, 255, 3).tolist()
                            overlay[mask > 0] = color
                            
                            # 境界線を描画
                            contours, _ = cv2.findContours(
                                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                            )
                            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
                    
                    vis_result = cv2.addWeighted(img_np, 0.4, overlay, 0.6, 0)
                    
                    # 保存
                    img_id = target['image_id'].item()
                    cv2.imwrite(str(vis_dir / f'pred_{img_id:05d}.jpg'), vis_result)
                    count += 1
    
    print(f"Saved {count} visualizations to {vis_dir}")


# ============================================================================
# Main Training Loop
# ============================================================================
def train(args):
    """メイン学習関数"""
    
    # デバイス設定
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print("\n" + "=" * 60)
    print(f"Instance Segmentation Training - {args.model.upper()}")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Dataset: {args.data}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print("=" * 60 + "\n")
    
    # データセット
    img_size = tuple(args.img_size)
    input_type = args.input_type
    
    train_dataset = COCOInstanceDataset(args.data, 'train', img_size, augment=True, input_type=input_type)
    val_dataset = COCOInstanceDataset(args.data, 'val', img_size, augment=False, input_type=input_type)
    
    # testデータセットの存在確認
    test_dir = Path(args.data) / 'test'
    test_masks_dir = test_dir / 'instance_masks'
    if test_dir.exists() and test_masks_dir.exists():
        test_dataset = COCOInstanceDataset(args.data, 'test', img_size, augment=False, input_type=input_type)
    else:
        print(f"Warning: test dataset not found at {test_dir}, using val dataset for testing")
        test_dataset = val_dataset
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=True
    )
    
    # モデル作成
    num_classes = len(train_dataset.cat_ids) + 1  # +1 for background
    model_name = None  # Mask2Former用
    
    if args.model == 'maskrcnn':
        model = create_maskrcnn(
            num_classes, 
            pretrained=args.pretrained,
            trainable_backbone_layers=args.trainable_layers
        )
        train_fn = train_one_epoch_maskrcnn
        validate_fn = validate_maskrcnn
    elif args.model == 'mask2former':
        model, model_name = create_mask2former(
            num_classes, 
            pretrained=args.pretrained,
            dropout=args.dropout,
            model_name=args.mask2former_model
        )
        train_fn = None  # 後で設定
        validate_fn = None  # 後で設定
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    model = model.to(device)
    print(f"✅ Model created: {args.model} (num_classes={num_classes})")
    if args.model == 'mask2former':
        print(f"   Model: {model_name}")
        print(f"   Dropout: {args.dropout}")
    
    # テストのみモード
    if args.test_only:
        if args.weights:
            state_dict = torch.load(args.weights, map_location=device)
            model.load_state_dict(state_dict)
            print(f"✅ Loaded weights from {args.weights}")
        
        print("\n" + "=" * 60)
        print("Test Evaluation")
        print("=" * 60)
        
        # インスタンスメトリクス評価
        model_name_for_eval = args.mask2former_model if args.model == 'mask2former' else None
        metrics = evaluate_instance_metrics(model, test_loader, device, args.model, model_name_for_eval)
        
        # 結果保存
        metrics_output = {
            'precision_mean': metrics.get('precision_mean', 0.0),
            'precision_std': metrics.get('precision_std', 0.0),
            'recall_mean': metrics.get('recall_mean', 0.0),
            'recall_std': metrics.get('recall_std', 0.0),
            'f1_mean': metrics.get('f1_mean', 0.0),
            'f1_std': metrics.get('f1_std', 0.0),
            'mean_iou_mean': metrics.get('mean_iou_mean', 0.0),
            'mean_iou_std': metrics.get('mean_iou_std', 0.0),
            'AP@50_mean': metrics.get('AP@50_mean', 0.0),
            'AP@50_std': metrics.get('AP@50_std', 0.0),
            'AP@75_mean': metrics.get('AP@75_mean', 0.0),
            'AP@75_std': metrics.get('AP@75_std', 0.0),
            'mAP_mean': metrics.get('mAP_mean', 0.0),
            'mAP_std': metrics.get('mAP_std', 0.0),
            'total_matched': metrics.get('total_matched', 0),
            'total_predicted': metrics.get('total_pred', 0),
            'total_gt': metrics.get('total_gt', 0),
            'num_images': metrics.get('num_images', 0),
        }
        
        # 出力ディレクトリ（モデル名_入力タイプ）
        output_dir = Path(args.output) / f"{args.model}_{args.input_type}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'test_metrics.json', 'w') as f:
            json.dump(metrics_output, f, indent=2)
        
        print(f"\n\u2705 Metrics saved to {output_dir / 'test_metrics.json'}")
        
        # 可視化
        model_name_for_vis = args.mask2former_model if args.model == 'mask2former' else None
        save_visualization(model, test_loader, device, output_dir, args.model, model_name_for_vis)
        
        return metrics_output
    
    # オプティマイザ（モデル別のweight_decay）
    params = [p for p in model.parameters() if p.requires_grad]
    weight_decay = args.weight_decay
    if args.model == 'mask2former' and args.weight_decay == 0.0001:
        weight_decay = 0.01  # Mask2Formerのデフォルト
    
    if args.optimizer == 'sgd':
        optimizer = SGD(params, lr=args.lr, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = AdamW(params, lr=args.lr, weight_decay=weight_decay)
    
    # スケジューラ
    total_steps = len(train_loader) * args.epochs
    step_scheduler = False
    
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    elif args.scheduler == 'onecycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=total_steps,
            pct_start=0.1  # 10% warmup
        )
        step_scheduler = True
    else:
        scheduler = None
    
    # 出力ディレクトリ（モデル名_入力タイプ）
    output_dir = Path(args.output) / f"{args.model}_{args.input_type}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # WandB
    if args.wandb and WANDB_AVAILABLE:
        wandb.init(
            project="manga-instance-seg",
            name=f"{args.model}_{args.input_type}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(args)
        )
    
    # 学習ループ
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print('='*60)
        
        # 学習
        if args.model == 'mask2former':
            train_loss = train_one_epoch_mask2former(
                model, train_loader, optimizer, device, epoch,
                scheduler=scheduler if step_scheduler else None,
                model_name=model_name
            )
            val_loss = validate_mask2former(model, val_loader, device, model_name)
        else:
            train_loss = train_fn(model, train_loader, optimizer, device, epoch)
            val_loss = validate_fn(model, val_loader, device)
        
        # スケジューラ（エポックベース）
        if scheduler is not None and not step_scheduler:
            scheduler.step()
        
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # WandBログ
        if args.wandb and WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # ベストモデル保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / 'best.pt')
            print(f"✅ Best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"⏳ No improvement for {patience_counter}/{args.patience} epochs")
        
        # Early Stopping
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\n🛑 Early stopping triggered after {epoch} epochs (no improvement for {args.patience} epochs)")
            break
        
        # チェックポイント
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, output_dir / f'checkpoint_epoch{epoch}.pt')
    
    # 最終モデル保存
    torch.save(model.state_dict(), output_dir / 'final.pt')
    
    # テスト評価
    print("\n" + "=" * 60)
    print("Final Test Evaluation")
    print("=" * 60)
    
    # ベストモデルをロード
    model.load_state_dict(torch.load(output_dir / 'best.pt', map_location=device))
    
    # インスタンスメトリクス評価
    test_metrics = evaluate_instance_metrics(model, test_loader, device, args.model, model_name)
    
    print("\n📊 Test Results:")
    print(f"  Precision: {test_metrics['precision_mean']:.4f} ± {test_metrics['precision_std']:.4f}")
    print(f"  Recall:    {test_metrics['recall_mean']:.4f} ± {test_metrics['recall_std']:.4f}")
    print(f"  F1:        {test_metrics['f1_mean']:.4f} ± {test_metrics['f1_std']:.4f}")
    print(f"  Mean IoU:  {test_metrics['mean_iou_mean']:.4f} ± {test_metrics['mean_iou_std']:.4f}")
    
    # 可視化
    save_visualization(model, test_loader, device, output_dir, args.model, model_name)
    
    # メトリクス出力形式（ユーザー指定形式）
    metrics_output = {
        'precision_mean': test_metrics.get('precision_mean', 0.0),
        'precision_std': test_metrics.get('precision_std', 0.0),
        'recall_mean': test_metrics.get('recall_mean', 0.0),
        'recall_std': test_metrics.get('recall_std', 0.0),
        'f1_mean': test_metrics.get('f1_mean', 0.0),
        'f1_std': test_metrics.get('f1_std', 0.0),
        'mean_iou_mean': test_metrics.get('mean_iou_mean', 0.0),
        'mean_iou_std': test_metrics.get('mean_iou_std', 0.0),
        'AP@50_mean': test_metrics.get('AP@50_mean', 0.0),
        'AP@50_std': test_metrics.get('AP@50_std', 0.0),
        'AP@75_mean': test_metrics.get('AP@75_mean', 0.0),
        'AP@75_std': test_metrics.get('AP@75_std', 0.0),
        'mAP_mean': test_metrics.get('mAP_mean', 0.0),
        'mAP_std': test_metrics.get('mAP_std', 0.0),
        'total_matched': test_metrics.get('total_matched', 0),
        'total_predicted': test_metrics.get('total_pred', 0),
        'total_gt': test_metrics.get('total_gt', 0),
        'num_images': test_metrics.get('num_images', 0),
    }
    
    # 結果を保存
    results = {
        'model': args.model,
        'best_val_loss': best_val_loss,
        'test_metrics': metrics_output,
        'history': history,
        'config': vars(args)
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # メトリクスのみのファイルも保存
    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(metrics_output, f, indent=2)
    
    if args.wandb and WANDB_AVAILABLE:
        wandb.log({
            'test_precision': metrics_output['precision_mean'],
            'test_recall': metrics_output['recall_mean'],
            'test_f1': metrics_output['f1_mean'],
            'test_mean_iou': metrics_output['mean_iou_mean'],
        })
        wandb.finish()
    
    print(f"\n✅ Training completed!")
    print(f"📁 Results saved to {output_dir}")
    print(f"📁 Metrics saved to {output_dir / 'test_metrics.json'}")
    
    return metrics_output


def parse_args():
    parser = argparse.ArgumentParser(description='Instance Segmentation Training')
    
    # モデル
    parser.add_argument('--model', type=str, default='maskrcnn',
                        choices=['maskrcnn', 'mask2former'],
                        help='Model type')
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--no-pretrained', action='store_false', dest='pretrained')
    
    # Mask R-CNN固有
    parser.add_argument('--trainable-layers', type=int, default=3,
                        help='Number of trainable backbone layers for Mask R-CNN (0-5)')
    
    # Mask2Former固有
    parser.add_argument('--mask2former-model', type=str,
                        default='facebook/mask2former-swin-tiny-coco-instance',
                        help='Pretrained Mask2Former model name')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate for Mask2Former (0.0-0.5 recommended)')
    
    # データ
    parser.add_argument('--data', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--input-type', type=str, default='3ch',
                        choices=['gray', '3ch'],
                        help='Input type: gray (grayscale only) or 3ch (Gray+LSD+SDF)')
    parser.add_argument('--img-size', type=int, nargs=2, default=[384, 512],
                        help='Image size (H W)')
    
    # 学習
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs (default: 50 for Mask R-CNN, use 100 for Mask2Former)')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                        help='Weight decay (default: 0.0001 for Mask R-CNN, 0.01 for Mask2Former)')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'sgd'])
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'onecycle', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (0 to disable, default: 15)')
    
    # 出力
    parser.add_argument('--output', type=str, default='./instance_outputs')
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--wandb', action='store_true')
    
    # テストのみ
    parser.add_argument('--test-only', action='store_true',
                        help='Run evaluation only')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to model weights for test-only mode')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
