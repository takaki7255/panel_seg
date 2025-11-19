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

### 1. U-Net系モデル

#### a) **UNetGray** (Grayスケール入力)
- **入力**: 1ch (Grayscale)
- **学習**: `python train_unet_gray.py --root ./panel_dataset --dataset panel_seg`
- **評価**: `python test_unet_gray.py --model ./panel_models/xxx.pt --root ./panel_dataset --split test`

#### b) **UNetGrayLSD** (Gray + LSD入力)
- **入力**: 2ch (Grayscale + LSD線分マップ)
- **学習**: `python train_unet_gray_lsd.py --root ./panel_dataset_processed --dataset panel_seg`

#### c) **UNetGrayLSDSDF** (Gray + LSD + SDF入力) ⭐推奨
- **入力**: 3ch (Grayscale + LSD + SDF距離マップ)
- **学習**: `python train_unet_gray_lsd_sdf.py --root ./panel_dataset_processed --dataset panel_seg`

## 📊 データセット準備

### LSD/SDF前処理 (必須: Gray+LSD, Gray+LSD+SDFモデル用)

```bash
python preprocess_lsd_sdf.py \
    --root ./panel_dataset \
    --output ./panel_dataset_processed \
    --lsd-scale 0.8 \
    --sdf-max-dist 50
```

## 🎯 損失関数

### CombinedLoss (BCE + Dice + Boundary)

```python
Loss = α * BCE + β * Dice + λ * Boundary
```

- **Boundary**: 境界領域(2-3px)に特化した損失
- λ = 0.2-0.4 (デフォルト: 0.3)
- 細い線が消えにくい

## 🔬 評価指標

- Dice Score, IoU, Precision, Recall, F1
- **Boundary F1**: 境界領域での精度
- **PR-AUC**: Precision-Recall曲線

## 🔜 次のステップ

- **SegFormer**: 広域文脈を活用
- **Mask2Former**: インスタンス分割