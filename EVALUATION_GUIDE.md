# モデル自動評価ツール

## 概要
`panel_models/`ディレクトリ内の全ての学習済みモデル（.pt）を自動的に評価するスクリプトです。

## 機能
- 複数モデルの一括評価
- モデル名から適切なテストスクリプトを自動判定
- 評価結果を整理して保存
- CSV形式でのサマリー出力
- 各モデルの詳細ログ保存

## 対応モデル

| モデルタイプ | テストスクリプト | 必要なデータ |
|------------|----------------|-------------|
| unetgray | test_unet_gray.py | 元画像のみ |
| unetgraylsd | test_unet_gray_lsd.py | 元画像 + LSD |
| unetgraylsdsdf | test_unet_gray_lsd_sdf.py | 元画像 + LSD + SDF |
| segformer | test_segformer.py | 元画像 + LSD + SDF |

## 使用方法

### 方法1: バッチファイルで実行（推奨）
```cmd
evaluate_all_models.bat
```
ダブルクリックでも実行可能です。

### 方法2: PowerShellで直接実行
```powershell
powershell -ExecutionPolicy Bypass -File evaluate_all_models.ps1
```

### 方法3: PowerShell内で実行
```powershell
.\evaluate_all_models.ps1
```

## 前提条件

### 1. 評価対象モデル
`panel_models/`ディレクトリに評価したい`.pt`ファイルを配置してください。

### 2. テストデータセット
以下のデータセットが必要です：
- `frame_dataset/test100_dataset/` - 元画像用
- `frame_dataset/1000_preprocessed/` - LSD/SDF処理済みデータ用

### 3. 前処理データの作成
LSD/SDFを使用するモデルを評価する場合、事前に前処理が必要です：

```cmd
# テストデータセット用の前処理（train/val分割なし）
python preprocess_lsd_sdf_test.py --root ./frame_dataset/test100_dataset --output ./frame_dataset/test100_preprocessed --min-line-length 10

# または通常の前処理（train/val分割あり）
python preprocess_lsd_sdf.py --root ./frame_dataset/1000_dataset --output ./frame_dataset/1000_preprocessed --min-line-length 10
```

**重要**: テストデータセット（`test100_dataset`など）は`images/`と`masks/`が直接配置されているため、
`preprocess_lsd_sdf_test.py`を使用してください。

評価スクリプトは自動的に以下の順で前処理済みデータを探します：
1. `frame_dataset/test100_preprocessed/` （優先）
2. `frame_dataset/1000_preprocessed/` （フォールバック）

## 出力

評価結果は `evaluation_results/YYYYMMDD_HHMMSS/` ディレクトリに保存されます。

### ディレクトリ構造
```
evaluation_results/
└── 20251127_143000/              # タイムスタンプ付きディレクトリ
    ├── evaluation_summary.txt    # サマリー
    ├── evaluation_results.csv    # CSV形式の結果
    ├── model1_evaluation.log     # モデル1の詳細ログ
    ├── model2_evaluation.log     # モデル2の詳細ログ
    └── ...
```

### CSV出力形式
```csv
Model,TestScript,Status,ExecutionTime,OutputFile
panel_seg-200-unetgray-01,test_unet_gray.py,Success,45.3,evaluation_results/.../panel_seg-200-unetgray-01_evaluation.log
```

## カスタマイズ

### テストデータセットの変更
スクリプト内の以下の行を編集：
```powershell
$testDataset = "frame_dataset/test100_dataset"  # ここを変更
```

### 前処理済みデータの場所を変更
各モデルタイプの判定部分で`$datasetRoot`を変更：
```powershell
if ($modelName -match "unetgraylsdsdf") {
    $datasetRoot = "frame_dataset/test100_preprocessed"  # ここを変更
}
```

## トラブルシューティング

### エラー: スクリプトの実行がシステムで無効
PowerShellの実行ポリシーが制限されています。以下のいずれかを実行：

1. バッチファイルを使用（推奨）
2. 管理者権限でPowerShellを開き、以下を実行：
```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### エラー: Dataset Not Found
前処理済みデータが作成されていません。上記の「前処理データの作成」を参照。

### エラー: Script Not Found
テストスクリプト（test_*.py）が見つかりません。
スクリプトは`panel_seg`のルートディレクトリで実行してください。

## 例

### 全モデルを評価
```cmd
evaluate_all_models.bat
```

### 結果の確認
```cmd
# CSVファイルをExcelで開く
start evaluation_results/20251127_143000/evaluation_results.csv

# 特定のモデルのログを確認
type evaluation_results/20251127_143000/panel_seg-200-unetgray-01_evaluation.log
```

## 注意事項

1. **GPU使用**: 評価にはGPUを使用します。CUDA対応環境が推奨されます。
2. **実行時間**: モデル数とテストデータ量に応じて時間がかかります。
3. **メモリ**: 複数モデルの評価では十分なメモリが必要です。
4. **データセット整合性**: モデルのトレーニングデータと評価データが異なることを確認してください。

## 結果の解釈

各ログファイルには以下の指標が含まれます：
- **Dice Score**: セグメンテーションの重なり度（0-1、高いほど良い）
- **IoU**: Intersection over Union（0-1、高いほど良い）
- **Precision**: 正解率（0-1、高いほど良い）
- **Recall**: 再現率（0-1、高いほど良い）
- **F1 Score**: PrecisionとRecallの調和平均
- **Boundary F1**: 境界線の精度
- **PR-AUC**: Precision-Recallカーブの面積（0-1、高いほど良い）
