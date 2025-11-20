# Manga Panel Segmentation

漫画コマのセグメンテーションを行うプロジェクト。複数のモデルアーキテクチャを実装・比較します。

## 📁 プロジェクト構造

```
panel_seg/
├── models/
│   ├── __init__.py
│   ├── losses.py                  # 損失関数 (BCE + Dice + Boundary)
│   ├── unet.py                    # ResNet-UNet (既存)
│   ├── unet_gray.py              # U-Net (Grayスケール入力)
│   ├── unet_gray_lsd.py          # U-Net (Gray + LSD入力)
│   ├── unet_gray_lsd_sdf.py      # U-Net (Gray + LSD + SDF入力)
│   ├── segformer.py              # SegFormer (MiT-B2/B3) - 実装予定
│   └── mask2former.py            # Mask2Former + Swin-T - 実装予定
├── preprocess_lsd_sdf.py         # LSD/SDF特徴量生成
├── train_*.py                     # 各モデルの学習スクリプト
├── test_*.py                      # 各モデルの評価スクリプト
└── README.md
```

## 🚀 実装済みモデル

### モデル一覧と使い方

| モデル | 入力 | 学習スクリプト | 評価スクリプト | 前処理 |
|--------|------|--------------|--------------|--------|
| UNetGray | 1ch (Gray) | `train_unet_gray.py` | `test_unet_gray.py` | 不要 |
| UNetGrayLSD | 2ch (Gray+LSD) | `train_unet_gray_lsd.py` | `test_unet_gray_lsd.py` | 必要 |
| UNetGrayLSDSDF | 3ch (Gray+LSD+SDF) | `train_unet_gray_lsd_sdf.py` | `test_unet_gray_lsd_sdf.py` | 必要 |
| SegFormer | 3ch (Gray+LSD+SDF) | `train_segformer.py` | `test_segformer.py` | 必要 |

### 1. U-Net系モデル

#### a) **UNetGray** (Grayスケール入力)
- **入力**: 1ch (Grayscale)
- **前処理**: 不要
- **学習**:
  ```bash
  python train_unet_gray.py \
      --root ./panel_dataset \
      --dataset panel_seg \
      --batch 8 \
      --lr 1e-4 \
      --epochs 200
  ```
- **評価**:
  ```bash
  python test_unet_gray.py \
      --model ./panel_models/panel_seg-unet_gray-01.pt \
      --root ./panel_dataset \
      --split test \
      --save-preds \
      --output ./results/unet_gray
  ```

#### b) **UNetGrayLSD** (Gray + LSD入力)
- **入力**: 2ch (Grayscale + LSD線分マップ)
- **前処理**: 必要（LSD）
- **学習**:
  ```bash
  python train_unet_gray_lsd.py \
      --root ./panel_dataset_processed \
      --dataset panel_seg \
      --batch 8 \
      --lr 1e-4
  ```
- **評価**:
  ```bash
  python test_unet_gray_lsd.py \
      --model ./panel_models/panel_seg-unet_gray_lsd-01.pt \
      --root ./panel_dataset_processed \
      --split test
  ```

#### c) **UNetGrayLSDSDF** (Gray + LSD + SDF入力) ⭐推奨
- **入力**: 3ch (Grayscale + LSD + SDF距離マップ)
- **前処理**: 必要（LSD + SDF）
- **学習**:
  ```bash
  python train_unet_gray_lsd_sdf.py \
      --root ./panel_dataset_processed \
      --dataset panel_seg \
      --batch 8 \
      --lr 1e-4 \
      --boundary-lambda 0.3
  ```
- **評価**:
  ```bash
  python test_unet_gray_lsd_sdf.py \
      --model ./panel_models/panel_seg-unet_gray_lsd_sdf-01.pt \
      --root ./panel_dataset_processed \
      --split test \
      --save-preds \
      --output ./results/unet_gray_lsd_sdf
  ```

### 2. Transformer系モデル

#### d) **SegFormer** (MiT-B2/B3)
- **入力**: 3ch (Grayscale + LSD + SDF)
- **前処理**: 必要（LSD + SDF）
- **特徴**: 広域文脈を活用、密集したページに強い
- **モデル選択**:
  - `nvidia/mit-b2`: 24.7M params (推奨)
  - `nvidia/mit-b3`: 44.6M params (より高精度)
- **学習**: 
  ```bash
  # MiT-B2 (推奨)
  python train_segformer.py \
      --root ./panel_dataset_processed \
      --dataset panel_seg \
      --model-name nvidia/mit-b2 \
      --batch 4 \
      --lr 5e-5 \
      --freeze-encoder \
      --freeze-epochs 5 \
      --boundary-lambda 0.3
  
  # MiT-B3 (より高精度、要GPU)
  python train_segformer.py \
      --root ./panel_dataset_processed \
      --dataset panel_seg \
      --model-name nvidia/mit-b3 \
      --batch 2 \
      --lr 5e-5
  ```
- **評価**:
  ```bash
  python test_segformer.py \
      --model ./panel_models/panel_seg-segformer-b2-01.pt \
      --model-name nvidia/mit-b2 \
      --root ./panel_dataset_processed \
      --split test \
      --save-preds \
      --output ./results/segformer-b2
  ```

## 📊 データセット準備

### 前処理が必要なモデル

| モデル | 前処理 | 使用する特徴量 |
|--------|--------|--------------|
| UNetGray | ❌ 不要 | Gray のみ |
| UNetGrayLSD | ✅ 必要 (LSD) | Gray + LSD |
| UNetGrayLSDSDF | ✅ 必要 (LSD+SDF) | Gray + LSD + SDF |
| SegFormer | ✅ 必要 (LSD+SDF) | Gray + LSD + SDF |

### LSD/SDF前処理の実行

#### 基本的な使い方
```bash
# LSD + SDF を生成（UNetGrayLSDSDF, SegFormer用）
python preprocess_lsd_sdf.py \
    --root ./panel_dataset \
    --output ./panel_dataset_processed \
    --lsd-scale 0.8 \
    --sdf-max-dist 50
```

#### 線分長さフィルタリング
```bash
# 10〜200pxの線分のみを使用（ノイズと長すぎる線を除外）
python preprocess_lsd_sdf.py \
    --root ./panel_dataset \
    --output ./panel_dataset_processed \
    --min-line-length 10 \
    --max-line-length 200 \
    --visualize
```

#### OpenCV LSD と pylsd の比較
```bash
# まずpylsdをインストール
pip install pylsd

# 両方の手法を比較
python preprocess_lsd_sdf.py \
    --root ./panel_dataset \
    --output ./panel_dataset_processed \
    --compare-methods \
    --min-line-length 10

# pylsdを使って前処理
python preprocess_lsd_sdf.py \
    --root ./panel_dataset \
    --output ./panel_dataset_processed \
    --lsd-method pylsd \
    --min-line-length 15
```

#### 前処理オプション一覧
- `--lsd-scale`: LSD検出パラメータ（デフォルト: 0.8、OpenCVのみ）
- `--lsd-method`: 検出手法 `opencv` or `pylsd`（デフォルト: opencv）
- `--min-line-length`: 最小線分長さ（デフォルト: 10px）
- `--max-line-length`: 最大線分長さ（デフォルト: None = 無制限）
- `--sdf-max-dist`: SDF正規化の最大距離（デフォルト: 50px）
- `--visualize`: 可視化を生成
- `--compare-methods`: OpenCV LSDとpylsdを比較

## 🎯 損失関数

### CombinedLoss (BCE + Dice + Boundary)

```python
Loss = α * BCE + β * Dice + λ * Boundary
```

- **BCE (Binary Cross Entropy)**: ピクセル単位の分類損失
- **Dice Loss**: 領域全体の重なりを評価
- **Boundary Loss**: 境界領域(2-3px)に特化した損失
- デフォルト設定:
  - α (BCE_WEIGHT) = 0.5
  - β (DICE_WEIGHT) = 0.5
  - λ (BOUNDARY_LAMBDA) = 0.3
  - 境界幅 (boundary_width) = 3px
  - 境界重み (boundary_weight) = 3.0

### 損失関数のカスタマイズ

学習時に損失関数の重みを調整可能:
```bash
python train_unet_gray_lsd_sdf.py \
    --root ./panel_dataset_processed \
    --dataset panel_seg \
    --bce-weight 0.5 \
    --dice-weight 0.5 \
    --boundary-lambda 0.3
```

## 🔬 評価指標

すべての評価スクリプトで以下の指標を計算:

- **Dice Score**: セグメンテーション精度の総合指標
- **IoU (Intersection over Union)**: 予測と正解の重なり
- **Precision**: 予測した領域のうち正解の割合
- **Recall**: 正解領域のうち予測できた割合
- **F1 Score**: PrecisionとRecallの調和平均
- **Boundary F1**: 境界領域(3px)での精度 - 細い線の検出精度
- **PR-AUC**: Precision-Recall曲線の面積 - 閾値に依存しない総合評価

## 🔧 学習パラメータ

### U-Net系モデルの推奨設定

```bash
python train_unet_gray_lsd_sdf.py \
    --root ./panel_dataset_processed \
    --dataset panel_seg \
    --batch 8 \              # バッチサイズ（GPUメモリに応じて調整）
    --lr 1e-4 \              # 学習率
    --epochs 200 \           # 最大エポック数
    --patience 15 \          # Early stopping
    --bce-weight 0.5 \
    --dice-weight 0.5 \
    --boundary-lambda 0.3
```

### SegFormerの推奨設定

```bash
python train_segformer.py \
    --root ./panel_dataset_processed \
    --dataset panel_seg \
    --model-name nvidia/mit-b2 \
    --batch 4 \              # U-Netより小さめ（モデルが大きいため）
    --lr 5e-5 \              # U-Netより低め（転移学習）
    --epochs 100 \
    --patience 20 \
    --freeze-encoder \       # 最初は encoderを凍結
    --freeze-epochs 5 \      # 5エポック後に解凍
    --boundary-lambda 0.3
```

### パラメータ調整のヒント

- **batch**: GPUメモリ不足の場合は小さくする（4, 2など）
- **lr**: 学習が不安定なら下げる、収束が遅いなら上げる
- **boundary-lambda**: 境界が重要なら0.3〜0.5、そうでなければ0.1〜0.2
- **min-line-length / max-line-length**: データに応じて調整
  - ノイズが多い: min を大きく（15〜20px）
  - 細かい線も使いたい: min を小さく（5〜10px）
  - 長すぎる線を除外: max を設定（200〜300px）

## 💡 推奨ワークフロー

### 1. 前処理（最初に1回だけ）
```bash
# LSD/SDF特徴量を生成
python preprocess_lsd_sdf.py \
    --root ./panel_dataset \
    --output ./panel_dataset_processed \
    --min-line-length 10 \
    --visualize
```

### 2. ベースライン実験
```bash
# Grayのみで学習（前処理不要）
python train_unet_gray.py --root ./panel_dataset --dataset panel_seg
python test_unet_gray.py --model ./panel_models/xxx.pt --root ./panel_dataset --split test
```

### 3. LSD追加の効果確認
```bash
python train_unet_gray_lsd.py --root ./panel_dataset_processed --dataset panel_seg
python test_unet_gray_lsd.py --model ./panel_models/xxx.pt --root ./panel_dataset_processed --split test
```

### 4. LSD+SDF追加（推奨設定）
```bash
python train_unet_gray_lsd_sdf.py --root ./panel_dataset_processed --dataset panel_seg
python test_unet_gray_lsd_sdf.py --model ./panel_models/xxx.pt --root ./panel_dataset_processed --split test
```

### 5. SegFormerで広域文脈を活用
```bash
python train_segformer.py \
    --root ./panel_dataset_processed \
    --dataset panel_seg \
    --model-name nvidia/mit-b2 \
    --freeze-encoder \
    --freeze-epochs 5

python test_segformer.py \
    --model ./panel_models/xxx.pt \
    --model-name nvidia/mit-b2 \
    --root ./panel_dataset_processed \
    --split test \
    --save-preds
```

### 6. 結果を比較
各モデルの評価結果（`./results/*/metrics.txt`）を比較して最適なモデルを選択

## 🔜 次のステップ

- **Mask2Former**: インスタンス分割(将来実装)

## 📦 必要なパッケージ

```bash
pip install torch torchvision transformers opencv-python scipy pillow scikit-learn matplotlib tqdm wandb
```