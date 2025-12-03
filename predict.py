"""
任意の画像に対してインスタンスセグメンテーションを実行し、オーバーレイ画像を出力

Usage:
    # Mask R-CNN
    python predict.py --model maskrcnn --weights ./outputs/maskrcnn_3ch/best.pt --input image.jpg
    
    # Mask2Former
    python predict.py --model mask2former --weights ./outputs/mask2former_3ch/best.pt --input image.jpg
    
    # 複数画像（フォルダ指定）
    python predict.py --model maskrcnn --weights ./outputs/maskrcnn_3ch/best.pt --input ./test_images/
    
    # 出力先指定
    python predict.py --model maskrcnn --weights ./outputs/maskrcnn_3ch/best.pt --input image.jpg --output ./results/
    
    # グレースケール入力
    python predict.py --model maskrcnn --weights ./outputs/maskrcnn_gray/best.pt --input image.jpg --input-type gray
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import cv2


def create_maskrcnn(num_classes=2, trainable_backbone_layers=3):
    """Mask R-CNNモデルを作成"""
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    
    model = maskrcnn_resnet50_fpn_v2(
        weights=None,
        trainable_backbone_layers=trainable_backbone_layers
    )
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    
    return model


def create_mask2former(num_classes=2, model_name='facebook/mask2former-swin-tiny-coco-instance'):
    """Mask2Formerモデルを作成"""
    from transformers import Mask2FormerForUniversalSegmentation
    
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    
    return model, model_name


def load_image(image_path, img_size=(384, 512), input_type='3ch', lsd_dir=None, sdf_dir=None):
    """
    画像を読み込み、前処理を行う
    
    Args:
        image_path: 入力画像パス
        img_size: (H, W)
        input_type: 'gray' or '3ch'
        lsd_dir: LSD画像のディレクトリ（3chの場合、Noneなら自動生成）
        sdf_dir: SDF画像のディレクトリ（3chの場合、Noneなら自動生成）
    
    Returns:
        image_tensor: (3, H, W) テンソル
        orig_image: 元画像（numpy, BGR）
        orig_size: (orig_h, orig_w)
    """
    # 元画像を読み込み（表示用）
    orig_image = cv2.imread(str(image_path))
    if orig_image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    orig_h, orig_w = orig_image.shape[:2]
    
    # グレースケールで読み込み
    img = Image.open(image_path).convert('L')
    img = img.resize((img_size[1], img_size[0]), Image.BILINEAR)
    gray_np = np.array(img, dtype=np.float32) / 255.0
    
    if input_type == '3ch':
        # LSD/SDF生成
        lsd_np, sdf_np = generate_lsd_sdf(image_path, img_size)
        img_np = np.stack([gray_np, lsd_np, sdf_np], axis=0)
    else:
        # グレースケールを3チャンネルに複製
        img_np = np.stack([gray_np, gray_np, gray_np], axis=0)
    
    image_tensor = torch.from_numpy(img_np).float()
    
    return image_tensor, orig_image, (orig_h, orig_w)


def generate_lsd_sdf(image_path, img_size, min_length=10, max_dist=50):
    """
    LSD（線分検出）とSDF（距離場）を生成
    """
    from scipy.ndimage import distance_transform_edt
    
    # グレースケールで読み込み
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((img_size[0], img_size[1]), dtype=np.float32), \
               np.zeros((img_size[0], img_size[1]), dtype=np.float32)
    
    # LSD検出
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines, _, _, _ = lsd.detect(img)
    
    # 線分マップ作成
    line_map = np.zeros_like(img)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length >= min_length:
                cv2.line(line_map, (int(x1), int(y1)), (int(x2), int(y2)), 255, 1)
    
    # リサイズ
    line_map_resized = cv2.resize(line_map, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)
    lsd_np = line_map_resized.astype(np.float32) / 255.0
    
    # SDF計算
    binary_map = (line_map_resized > 127).astype(np.uint8)
    distance = distance_transform_edt(1 - binary_map)
    distance_normalized = np.clip(distance / max_dist, 0, 1)
    sdf_np = (1.0 - distance_normalized).astype(np.float32)
    
    return lsd_np, sdf_np


def predict_maskrcnn(model, image_tensor, device, score_threshold=0.5):
    """
    Mask R-CNNで予測
    
    Returns:
        masks: List of binary masks (H, W)
        scores: List of confidence scores
        boxes: List of bounding boxes [x1, y1, x2, y2]
    """
    model.eval()
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model([image_tensor])[0]
    
    masks = []
    scores = []
    boxes = []
    
    pred_masks = outputs['masks'].cpu().numpy()
    pred_scores = outputs['scores'].cpu().numpy()
    pred_boxes = outputs['boxes'].cpu().numpy()
    
    for i in range(len(pred_scores)):
        if pred_scores[i] >= score_threshold:
            mask = (pred_masks[i, 0] > 0.5).astype(np.uint8)
            masks.append(mask)
            scores.append(pred_scores[i])
            boxes.append(pred_boxes[i])
    
    return masks, scores, boxes


def predict_mask2former(model, image_tensor, device, model_name):
    """
    Mask2Formerで予測
    
    Returns:
        masks: List of binary masks (H, W)
        scores: List of confidence scores
        boxes: List of bounding boxes [x1, y1, x2, y2]
    """
    from transformers import Mask2FormerImageProcessor
    
    model.eval()
    processor = Mask2FormerImageProcessor.from_pretrained(model_name)
    
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        outputs = model(pixel_values=image_tensor)
        
        # 後処理
        result = processor.post_process_instance_segmentation(
            outputs,
            target_sizes=[(image_tensor.shape[2], image_tensor.shape[3])]
        )[0]
    
    masks = []
    scores = []
    boxes = []
    
    if 'segments_info' in result:
        segmentation = result['segmentation'].cpu().numpy()
        
        for seg_info in result['segments_info']:
            mask = (segmentation == seg_info['id']).astype(np.uint8)
            masks.append(mask)
            scores.append(seg_info.get('score', 1.0))
            
            # マスクからバウンディングボックスを計算
            pos = np.where(mask > 0)
            if len(pos[0]) > 0:
                ymin, ymax = pos[0].min(), pos[0].max()
                xmin, xmax = pos[1].min(), pos[1].max()
                boxes.append([xmin, ymin, xmax, ymax])
            else:
                boxes.append([0, 0, 0, 0])
    
    return masks, scores, boxes


def create_overlay(orig_image, masks, scores, boxes, img_size, alpha=0.5, show_boxes=True, show_scores=True):
    """
    オーバーレイ画像を作成
    
    Args:
        orig_image: 元画像（BGR, numpy）
        masks: List of binary masks (model output size)
        scores: List of confidence scores
        boxes: List of bounding boxes
        img_size: モデルの入力サイズ (H, W)
        alpha: オーバーレイの透明度
        show_boxes: バウンディングボックスを表示するか
        show_scores: スコアを表示するか
    
    Returns:
        overlay_image: オーバーレイ画像（BGR, numpy）
    """
    orig_h, orig_w = orig_image.shape[:2]
    overlay = orig_image.copy()
    
    # カラーパレット（インスタンスごとに異なる色）
    np.random.seed(42)
    colors = np.random.randint(0, 255, (100, 3)).tolist()
    
    for i, (mask, score, box) in enumerate(zip(masks, scores, boxes)):
        color = colors[i % len(colors)]
        
        # マスクを元画像サイズにリサイズ
        mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        # マスク領域に色を適用
        overlay[mask_resized > 0] = color
        
        # 輪郭を描画
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        
        if show_boxes:
            # バウンディングボックスを元画像サイズにスケール
            scale_x = orig_w / img_size[1]
            scale_y = orig_h / img_size[0]
            
            x1 = int(box[0] * scale_x)
            y1 = int(box[1] * scale_y)
            x2 = int(box[2] * scale_x)
            y2 = int(box[3] * scale_y)
            
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            if show_scores:
                label = f"{score:.2f}"
                cv2.putText(overlay, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # 元画像とオーバーレイを合成
    result = cv2.addWeighted(orig_image, 1 - alpha, overlay, alpha, 0)
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Instance Segmentation Prediction')
    
    # モデル設定
    parser.add_argument('--model', type=str, required=True,
                        choices=['maskrcnn', 'mask2former'],
                        help='Model type')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights')
    parser.add_argument('--mask2former-model', type=str,
                        default='facebook/mask2former-swin-tiny-coco-instance',
                        help='Pretrained Mask2Former model name')
    
    # 入力設定
    parser.add_argument('--input', type=str, required=True,
                        help='Input image path or directory')
    parser.add_argument('--input-type', type=str, default='3ch',
                        choices=['gray', '3ch'],
                        help='Input type: gray or 3ch (Gray+LSD+SDF)')
    parser.add_argument('--img-size', type=int, nargs=2, default=[384, 512],
                        help='Model input size (H W)')
    
    # 出力設定
    parser.add_argument('--output', type=str, default='./predictions',
                        help='Output directory')
    parser.add_argument('--score-threshold', type=float, default=0.5,
                        help='Score threshold for predictions')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Overlay transparency (0-1)')
    parser.add_argument('--no-boxes', action='store_true',
                        help='Hide bounding boxes')
    parser.add_argument('--no-scores', action='store_true',
                        help='Hide scores')
    
    args = parser.parse_args()
    
    # デバイス設定
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Device: {device}")
    
    # モデル読み込み
    print(f"Loading {args.model} model...")
    model_name = None
    
    if args.model == 'maskrcnn':
        model = create_maskrcnn(num_classes=2)
    else:
        model, model_name = create_mask2former(num_classes=2, model_name=args.mask2former_model)
    
    # 重み読み込み
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"Loaded weights from {args.weights}")
    
    # 入力画像のリストを取得
    input_path = Path(args.input)
    if input_path.is_dir():
        image_paths = list(input_path.glob('*.jpg')) + \
                      list(input_path.glob('*.jpeg')) + \
                      list(input_path.glob('*.png'))
    else:
        image_paths = [input_path]
    
    if not image_paths:
        print(f"No images found in {args.input}")
        return
    
    print(f"Found {len(image_paths)} image(s)")
    
    # 出力ディレクトリ作成
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    img_size = tuple(args.img_size)
    
    # 各画像を処理
    for image_path in image_paths:
        print(f"\nProcessing: {image_path.name}")
        
        try:
            # 画像読み込み
            image_tensor, orig_image, orig_size = load_image(
                image_path, img_size, args.input_type
            )
            
            # 予測
            if args.model == 'maskrcnn':
                masks, scores, boxes = predict_maskrcnn(
                    model, image_tensor, device, args.score_threshold
                )
            else:
                masks, scores, boxes = predict_mask2former(
                    model, image_tensor, device, model_name
                )
            
            print(f"  Detected {len(masks)} instance(s)")
            
            # オーバーレイ画像作成
            overlay = create_overlay(
                orig_image, masks, scores, boxes, img_size,
                alpha=args.alpha,
                show_boxes=not args.no_boxes,
                show_scores=not args.no_scores
            )
            
            # 保存
            output_path = output_dir / f"pred_{image_path.stem}.jpg"
            cv2.imwrite(str(output_path), overlay)
            print(f"  Saved: {output_path}")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    print(f"\nDone! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
