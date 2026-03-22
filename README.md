# 漫画パネル分割システム (Manga Panel Segmentation)

複数の深層学習モデルを使用した漫画ページのコマ（パネル）自動分割プロジェクト。セマンティックセグメンテーションとインスタンスセグメンテーションの両方に対応しています。

**主な特徴:**
- 🎯 セマンティック・インスタンスセグメンテーション両対応
- 🚀 複数のモデルアーキテクチャ（U-Net, SegFormer, Mask R-CNN, Mask2Former）
- 🔍 LSD/SDF特徴量を活用した高精度境界検出
- ⚡ メモリ監視付きの安定した学習
- 📊 自動評価・バッチ処理対応

---

## 📋 目次

1. [概要](#概要)
2. [セットアップ](#セットアップ)
3. [プロジェクト構造](#プロジェクト構造)
4. [モデル説明](#モデル説明)
5. [使用方法](#使用方法)
6. [評価・結果確認](#評価結果確認)

---

## 概要

### プロジェクト目標

Manga109などのマンガ画像データセットから、個々のコマ（パネル）を自動的に検出・分割するシステムを構築・評価することが目標です。以下の2つのタスクを実装：

1. **セマンティックセグメンテーション**: パネル領域 vs 背景の二値分割
2. **インスタンスセグメンテーション**: 個々のパネルを別々のインスタンスとして分割

### 主な技術

| 要素 | 説明 |
|------|------|
| **LSD処理** | Line Segment Detection により、パネル境界線を検出 |
| **SDF計算** | 符号付き距離場を生成し、パネル境界までの距離情報を提供 |
| **カスタム損失関数** | BCE + Dice + 境界損失（パネル端を重視） |
| **マルチモーダル入力** | グレースケール + LSD + SDF の多層的な特徴情報 |

---

## セットアップ

### 必要なライブラリ

```bash
# 基本パッケージ
pip install torch torchvision
pip install opencv-python pillow numpy scipy
pip install scikit-image scikit-learn

# モデル関連
pip install segmentation-models-pytorch transformers huggingface-hub

# ユーティリティ
pip install pycocotools wandb tqdm

# 前処理（LSD/SDF）
pip install lsd # または OpenCV LSD使用
```

### データセット準備

```bash
# 1. データセット作成（Manga109から）
python create_dataset.py --root <manga109_path> --output ./panel_dataset

# 2. LSD/SDF特徴量生成（必要に応じて）
python preprocess_lsd_sdf.py --root ./panel_dataset --output ./panel_dataset_processed

# 3. インスタンスセグメンテーション用データセット作成
python create_instance_dataset.py --root ./panel_dataset --output ./instance_dataset
```

---

## プロジェクト構造

```
panel_seg/
├── models/                              # モデルアーキテクチャ
│   ├── __init__.py
│   ├── losses.py                        # カスタム損失関数
│   ├── unet.py                          # U-Net (3ch入力)
│   ├── unet_gray.py                     # U-Net (1ch グレースケール)
│   ├── unet_gray_lsd.py                 # U-Net (2ch Gray+LSD)
│   ├── unet_gray_lsd_sdf.py             # U-Net (3ch Gray+LSD+SDF) ⭐
│   ├── segformer.py                     # SegFormer (MiT-B2/B3)
│   ├── segformer_gray.py                # SegFormer グレースケール版
│   ├── maskrcnn.py                      # Mask R-CNN (3ch)
│   ├── maskrcnn_gray.py                 # Mask R-CNN (1ch)
│   ├── mask2former.py                   # Mask2Former (3ch) ⭐最新
│   └── mask2former_gray.py              # Mask2Former (1ch)
│
├── utils/                               # ユーティリティ
│   ├── metrics.py                       # 評価指標（Dice, IoU, F1等）
│   ├── postprocess.py                   # 後処理（ワーターシェッド等）
│   └── data.py                          # データローディング・前処理
│
├── 📌 データ準備スクリプト
│   ├── create_dataset.py                # Manga109からデータセット作成
│   ├── create_instance_dataset.py       # インスタンスマスク生成
│   ├── preprocess_lsd_sdf.py            # LSD/SDF特徴量生成（train/val/test）
│   └── preprocess_lsd_sdf_test.py       # LSD/SDF特徴量生成（新規データ用）
│
├── 📌 学習スクリプト（セマンティックセグメンテーション）
│   ├── train_unet_gray.py               # UNetGray学習
│   ├── train_unet_gray_lsd.py           # UNetGrayLSD学習
│   ├── train_unet_gray_lsd_sdf.py       # UNetGrayLSDSDF学習
│   ├── train_segformer_gray.py          # SegFormer(Gray)学習
│   ├── train_segformer.py               # SegFormer(Gray+LSD+SDF)学習
│   └── train_resnet_unet.py             # ResNet-UNet学習
│
├── 📌 学習スクリプト（インスタンスセグメンテーション）
│   ├── train_maskrcnn.py                # Mask R-CNN学習
│   ├── train_maskrcnn_gray.py           # Mask R-CNN(Gray)学習
│   ├── train_mask2former.py             # Mask2Former学習
│   └── train_mask2former_gray.py        # Mask2Former(Gray)学習
│
├── 📌 評価スクリプト（セマンティック用）
│   ├── test_unet_gray.py
│   ├── test_unet_gray_lsd.py
│   ├── test_unet_gray_lsd_sdf.py
│   ├── test_segformer.py
│   ├── test_segformer_gray.py
│   └── test_with_postprocess.py         # 後処理付き評価
│
├── 📌 評価スクリプト（インスタンス用）
│   ├── test_maskrcnn.py
│   ├── test_mask2former.py              # Mask2Former評価
│   └── test_all_instance_models.ps1     # Windows用バッチ実行
│
├── 📌 推論・バッチ処理
│   ├── predict.py                        # 新規画像への推論
│   ├── convert_to_instance_masks.py     # マスク形式変換
│   ├── evaluate_all_models.bat          # Windows：全モデル自動評価
│   ├── evaluate_all_models.ps1          # PowerShell版自動評価
│   ├── train_all_instance_models.ps1    # 全インスタンスセグ学習
│   └── train_instance_seg_all.ps1       # 別版全学習スクリプト
│
├── 📁 データディレクトリ
│   ├── panel_dataset/                   # セマンティックセグ用データセット
│   ├── instance_dataset/                # インスタンスセグ用データ
│   ├── frame_dataset/                   # フレーム検出用データ（複数サイズ）
│   │   ├── 1000_instance/
│   │   ├── 200_instance/
│   │   ├── 5000_instance/
│   │   └── test100_instance/
│   │
│   ├── 📁 panel_models/                 # 学習済みセマンティックセグモデル
│   │   ├── mask2former/
│   │   ├── mask2former_3ch_*/
│   │   ├── mask2former_gray_*/
│   │   ├── maskrcnn_*/
│   │   └── ...
│   │
│   ├── 📁 instance_models/              # 学習済みインスタンスセグモデル
│   │   ├── mask2former_*/
│   │   ├── maskrcnn_*/
│   │   └── ...
│   │
│   └── 📁 results/                      # 評価・推論結果
│       ├── evaluation_results/
│       ├── segformer*/, unetgray*/, ...
│       └── evaluation_results/20251127_*/
│
├── 📄 EVALUATION_GUIDE.md               # 評価ツール詳細ガイド
└── 📄 README.md (このファイル)

```

### ディレクトリの役割

| パス | 用途 |
|------|------|
| `models/` | 全モデルの実装 |
| `utils/` | 共通ユーティリティ |
| `panel_dataset/` | セマンティックセグメンテーション用データ |
| `instance_dataset/` | インスタンスセグメンテーション用データ |
| `frame_dataset/` | オブジェクト検出（フレーム検出）用データ |
| `*_models/` | 学習済みモデルの保存先 |
| `results/` | 評価・推論結果の出力先 |

---

## モデル説明

### セマンティックセグメンテーション

パネル領域と背景を2値分類するモデル群。各モデルは，入力のチャネル数とアーキテクチャが異なります。

#### 1. U-Net系モデル

**特徴**: シンプルで高速、少ないGPUメモリで実行可能

| モデル | 入力形式 | 特徴 | 速度 | メモリ効率 |
|--------|---------|------|------|----------|
| **UNetGray** | 1ch (グレースケール) | 単純，最小限 | ⚡ 最速 | 💚 最小 |
| **UNetGrayLSD** | 2ch (Gray + LSD) | LSD線分情報を活用 | ⚡ 高速 | 💚 良好 |
| **UNetGrayLSDSDF** ⭐ | 3ch (Gray + LSD + SDF) | 複合特徴, 高精度 | ⚡ 高速 | 💚 良好 |

**学習例（UNetGrayLSDSDF）:**
```bash
python train_unet_gray_lsd_sdf.py \
    --root ./panel_dataset_processed \
    --dataset panel_seg \
    --batch 8 \
    --lr 1e-4 \
    --epochs 200 \
    --save-interval 10 \
    --device cuda
```

#### 2. SegFormer（Transformer ベース）

**特徴**: 広い受容野，最新アーキテクチャ，より高精度

| モデル | バックボーン | 入力 | 精度 | 計算コスト |
|--------|-------------|------|------|----------|
| **SegFormer** | MiT-B2 | 3ch (Gray+LSD+SDF) | 高 | 中 |
| **SegFormer** | MiT-B3 | 3ch | 最高 | 高 |
| **SegFormerGray** | MiT-B2 | 1ch | 中 | 低 |

**学習例:**
```bash
python train_segformer.py \
    --root ./panel_dataset_processed \
    --backbone mit_b2 \
    --batch 4 \
    --lr 5e-5 \
    --epochs 150
```

### インスタンスセグメンテーション

個々のパネルを別々のインスタンスとして識別するモデル。

#### 1. Mask R-CNN

**特徴**: 確立された方法，安定した学習，RPN(Region Proposal Network)で候補領域を検出

| モデル | 入力 | バックボーン | 速度 | 精度 |
|--------|------|------------|------|------|
| **MaskRCNN** | 3ch | ResNet-50 FPN | 中速 | 高 |
| **MaskRCNN(Gray)** | 1ch | ResNet-50 FPN | 速 | 中 |

**学習:**
```bash
python train_maskrcnn.py \
    --root ./instance_dataset \
    --batch 4 \
    --lr 1e-4 \
    --epochs 100
```

#### 2. Mask2Former ⭐ **推奨**

**特徴**: 最新フレームワーク，セマンティック・インスタンス統一処理，最高精度

| モデル | 入力 | バックボーン | 速度 | 精度 |
|--------|------|------------|------|------|
| **Mask2Former** | 3ch | Swin-T | 速 | 最高 |
| **Mask2Former(Gray)** | 1ch | Swin-T | 超速 | 高 |

**学習:**
```bash
python train_mask2former.py \
    --root ./instance_dataset_processed \
    --batch 4 \
    --lr 1e-4 \
    --epochs 120 \
    --use-panoptic-seg
```

---

## 使用方法

### 🚀 クイックスタート（学習から評価まで）

#### ケース 1: U-Net灰色入力で素早く試す

```bash
# 1. データセット準備（グレースケール入力なので前処理不要）
python create_dataset.py --root /path/to/manga109 --output ./panel_dataset

# 2. 学習
python train_unet_gray.py --root ./panel_dataset --batch 8 --epochs 100

# 3. 評価
python test_unet_gray.py --model ./panel_models/panel_seg-unet_gray-*.pt \
    --root ./panel_dataset --output ./results/unet_gray --save-preds
```

#### ケース 2: 高精度を目指す（U-Net + LSD/SDF）

```bash
# 1. データセット準備
python create_dataset.py --root /path/to/manga109 --output ./panel_dataset

# 2. LSD/SDF特徴量生成
python preprocess_lsd_sdf.py --root ./panel_dataset --output ./panel_dataset_processed

# 3. 学習
python train_unet_gray_lsd_sdf.py --root ./panel_dataset_processed \
    --batch 8 --lr 1e-4 --epochs 200

# 4. 評価（後処理付き）
python test_with_postprocess.py --model ./panel_models/unet_*.pt \
    --root ./panel_dataset_processed --postprocess watershed
```

#### ケース 3: インスタンスセグメンテーション（Mask2Former）

```bash
# 1. インスタンスデータセット作成
python create_instance_dataset.py --root ./panel_dataset --output ./instance_dataset

# 2. LSD/SDF生成
python preprocess_lsd_sdf.py --root ./instance_dataset \
    --output ./instance_dataset_processed

# 3. 学習
python train_mask2former.py --root ./instance_dataset_processed \
    --batch 4 --epochs 120

# 4. 評価
python test_mask2former.py --model ./instance_models/mask2former_*.pt \
    --root ./instance_dataset_processed --coco-eval
```

### 📊 新規画像への推論

```bash
python predict.py \
    --model ./panel_models/unet_gray_lsd_sdf.pt \
    --image /path/to/manga_page.png \
    --output ./predictions/ \
    --visualize
```

---

## 評価・結果確認

### 自動評価（全モデル一括）

すべての学習済みモデルを一括で評価し、結果をCSVで出力：

**Windows:**
```cmd
evaluate_all_models.bat
```

**Linux/Mac:**
```bash
# 手動で各モデルを評価（スクリプトはWindows PowerShell用）
python test_unet_gray.py --model ./panel_models/unet_gray.pt --output ./results/
python test_segformer.py --model ./panel_models/segformer.pt --output ./results/
python test_maskrcnn.py --model ./instance_models/maskrcnn.pt --output ./results/
python test_mask2former.py --model ./instance_models/mask2former.pt --output ./results/
```

### 評価メトリクス

| タスク | メトリクス | 説明 |
|--------|----------|------|
| **セマンティックセグ** | Dice Score | 領域の重複度（0-1，高=良） |
| | IoU (Jaccard) | 共有領域 / 合計領域（0-1）|
| | Precision/Recall | 精度・再現率 |
| | Boundary F1 | パネル境界の精度（重要） |
| | F1 Score | Precision と Recall の調和平均 |
| **インスタンスセグ** | AP@50, AP@75 | COCO形式の平均精度 |
| | mAP | 複数IoUしきい値での平均 |
| | Per-Instance IoU | 個別インスタンスの精度 |

### 結果の確認

評価結果は以下に保存されます：

```
results/
├── unet_gray/
│   ├── metrics.json        # 集計結果
│   ├── predictions/        # 予測マスク画像
│   └── pr_curve.png        # Precision-Recall曲線
├── evaluation_results/20251127_114712/
│   ├── evaluation_summary.csv  # 全モデル比較表
│   └── detailed_results.json   # 詳細メトリクス
└── ...
```

詳細な評価ツール・解釈については、[EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) を参照してください。

---

## 損失関数・学習設定

### カスタム損失関数

```
CombinedLoss = 0.5 × BCELoss + 0.5 × DiceLoss + 0.3 × BoundaryLoss
```

**各要素の役割:**

| 成分 | 重み | 役割 |
|-----|------|------|
| **BCE Loss** | 0.5 | ピクセルレベルの二値分類 |
| **Dice Loss** | 0.5 | 領域全体のオーバーラップ重視，クラス不均衡に強い |
| **Boundary Loss** | 0.3 | パネル境界（2-3px付近）を3倍で重みづけ，エッジ精度向上 |

### 最適化設定

| パラメータ | 推奨値 | 備考 |
|----------|--------|------|
| Optimizer | AdamW | L2正則化付き |
| Learning Rate (U-Net) | 1e-4 | CosineAnnealing で段階的に削減 |
| Learning Rate (SegFormer) | 5e-5 | より繊細な調整が必要 |
| Learning Rate (Mask R-CNN) | 1e-4 | OneCycle スケジューラ推奨 |
| Batch Size | 4-8 | GPU メモリに応じて調整 |
| Epochs | 100-200 | Early stopping で早期終了の可能性 |
| Warmup Epochs | 5-10 | 最初は小さい学習率から開始 |

---

## 高度な機能

### 1. メモリ監視と自動チェックポイント保存

学習中に GPU/CPU メモリを監視し，メモリ不足時に自動保存：

```python
# train_*.py内で自動実行
# → メモリ警告時は最新チェックポイント保存 + 終了
```

### 2. 学習の再開

中断した学習を再開：

```bash
python train_unet_gray_lsd_sdf.py \
    --root ./panel_dataset_processed \
    --resume ./checkpoints/latest.pt \
    --epochs 200
```

### 3. 転移学習・エンコーダ固定

SegFormer や Mask2Former で HuggingFace の事前学習済みモデルを活用:

```bash
python train_segformer.py \
    --root ./panel_dataset_processed \
    --pretrained-backbone  # HuggingFace から自動ダウンロード
    --freeze-bn             # Batch Norm を固定
```

### 4. 後処理：パネルの分離

学習済みモデルの出力に対して，触れているパネルを分離：

```bash
python test_with_postprocess.py \
    --model ./panel_models/unet_gray_lsd_sdf.pt \
    --root ./panel_dataset_processed \
    --postprocess watershed  # または "morphological"
```

**利用可能な後処理:**
- `morphological`: モルフォロジー演算 + 連結成分検出（高速）
- `watershed`: ワーターシェッド変換（より正確, 遅い）

---

## トラブルシューティング

### メモリ不足エラー

**症状:** `RuntimeError: CUDA out of memory`

**対処:**
```bash
# バッチサイズを削減
python train_unet_gray_lsd_sdf.py --batch 4  # 通常は 8

# またはモデルを軽量版にカスタマイズ
python train_segformer.py --backbone mit_b2  # B3 から B2 へ
```

### 学習が進まない・精度が低い

**原因 & 対策:**
- LSD/SDF 特徴量が未生成 → `preprocess_lsd_sdf.py` を実行
- 学習率が大きすぎる → `--lr` を 5e-5 に削減
- エポック数不足 → `--epochs 200` に増加

### GPU / CUDA エラー

```bash
# CUDA デバイス確認
python -c "import torch; print(torch.cuda.is_available())"

# CPU モードで実行（低速）
python train_unet_gray.py --device cpu
```

---

## 引用・参考文献

このプロジェクトで使用されているアーキテクチャ：

- **U-Net**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
- **SegFormer**: Xie et al., "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers" (2021)
- **Mask R-CNN**: He et al., "Mask R-CNN" (2018)
- **Mask2Former**: Cheng et al., "Masked-attention Transformer Unified Transformers" (2023)
- **LSD**: von Gioi et al., "LSD: A Fast Line Segment Detector with False Detection Control" (2012)

---

## ライセンス・著作権

このプロジェクト内で使用するデータセット（Manga109）こそ適切に帰属して使用してください。

---

## お問い合わせ

問題が発生した場合は，以下の情報とともに連絡してください：

- エラーメッセージ全文
- 実行したコマンド
- Python バージョン
- GPU / CUDA バージョン（該当する場合）
- 使用モデルと入力データ情報

---

**Last Updated**: 2026年3月22日
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

#### 学習用データセット（train/val分割あり）
```bash
# LSD + SDF を生成（UNetGrayLSDSDF, SegFormer用）
python preprocess_lsd_sdf.py \
    --root ./frame_dataset/1000_dataset \
    --output ./frame_dataset/1000_preprocessed \
    --lsd-scale 0.8 \
    --sdf-max-dist 50
```

#### テストデータセット（train/val分割なし）
```bash
# テスト用データセットの前処理
python preprocess_lsd_sdf_test.py \
    --root ./frame_dataset/test100_dataset \
    --output ./frame_dataset/test100_preprocessed \
    --min-line-length 10
```

**重要**: 
- **学習用データセット**（`train/`と`val/`フォルダあり）: `preprocess_lsd_sdf.py`を使用
- **テストデータセット**（`images/`と`masks/`が直接配置）: `preprocess_lsd_sdf_test.py`を使用

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