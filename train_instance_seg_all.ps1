# ============================================================================
# Train All Instance Segmentation Models (Unified Script)
# ============================================================================
# 
# 使用方法:
#   .\train_instance_seg_all.ps1
#
# 前提条件:
#   - instance_dataset/5000_instance が存在すること
#   - 存在しない場合は以下を実行:
#     python create_instance_dataset.py --name 5000_instance --total 5000
#
# 使用スクリプト: train_instance_seg.py (統合版)
#
# 学習モデル:
#   1. Mask R-CNN Gray     (グレースケール入力)
#   2. Mask R-CNN 3ch      (Gray + LSD + SDF)
#   3. Mask2Former Gray    (グレースケール入力)
#   4. Mask2Former 3ch     (Gray + LSD + SDF)
#
# ============================================================================

$ErrorActionPreference = "Stop"

# ============================================================================
# Configuration
# ============================================================================
$DATASET = "./instance_dataset/5000_instance"
$EPOCHS = 50
$BATCH_SIZE = 8
$LR = "1e-4"
$WEIGHT_DECAY = 0.01
$OUTPUT_BASE = "./instance_outputs"
$USE_WANDB = $true

# Scheduler settings
# Mask R-CNN: cosine (default)
# Mask2Former: onecycle (with ~10% warmup built-in)
$SCHEDULER_MASKRCNN = "cosine"
$SCHEDULER_MASK2FORMER = "onecycle"

# Timestamp for this training run
$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Instance Segmentation Training (4 Models)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Dataset: $DATASET"
Write-Host "Epochs: $EPOCHS"
Write-Host "Batch Size: $BATCH_SIZE"
Write-Host "Learning Rate: $LR"
Write-Host "Weight Decay: $WEIGHT_DECAY"
Write-Host "Scheduler (Mask R-CNN): $SCHEDULER_MASKRCNN"
Write-Host "Scheduler (Mask2Former): $SCHEDULER_MASK2FORMER (includes 10% warmup)"
Write-Host "Timestamp: $TIMESTAMP"
Write-Host ""
Write-Host "Models to train:" -ForegroundColor Yellow
Write-Host "  1. Mask R-CNN Gray   (grayscale only)"
Write-Host "  2. Mask R-CNN 3ch    (Gray + LSD + SDF)"
Write-Host "  3. Mask2Former Gray  (grayscale only)"
Write-Host "  4. Mask2Former 3ch   (Gray + LSD + SDF)"
Write-Host "============================================================" -ForegroundColor Cyan

# Check if dataset exists
if (-not (Test-Path $DATASET)) {
    Write-Host "ERROR: Dataset not found at $DATASET" -ForegroundColor Red
    Write-Host "Please run first:" -ForegroundColor Yellow
    Write-Host "  python create_instance_dataset.py --name 5000_instance --total 5000" -ForegroundColor Yellow
    exit 1
}

# Check required files
$trainDir = Join-Path $DATASET "train"
$valDir = Join-Path $DATASET "val"

if (-not (Test-Path (Join-Path $trainDir "annotations.json"))) {
    Write-Host "ERROR: train/annotations.json not found" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path (Join-Path $valDir "annotations.json"))) {
    Write-Host "ERROR: val/annotations.json not found" -ForegroundColor Red
    exit 1
}

Write-Host "Dataset structure verified!" -ForegroundColor Green

# WandB flag
$WANDB_FLAG = if ($USE_WANDB) { "--wandb" } else { "" }

# Track training results
$results = @{}

# ============================================================================
# 1. Mask R-CNN Gray
# ============================================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "[1/4] Training Mask R-CNN Gray (Grayscale only)" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green

$startTime = Get-Date

python train_instance_seg.py `
    --model maskrcnn `
    --input-type gray `
    --data $DATASET `
    --epochs $EPOCHS `
    --batch $BATCH_SIZE `
    --lr $LR `
    --weight-decay $WEIGHT_DECAY `
    --scheduler $SCHEDULER_MASKRCNN `
    --trainable-layers 3 `
    --output $OUTPUT_BASE `
    --save-every 20 `
    $WANDB_FLAG

$endTime = Get-Date
$duration = $endTime - $startTime

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Mask R-CNN Gray training failed" -ForegroundColor Red
    $results["maskrcnn_gray"] = "FAILED"
} else {
    Write-Host "Mask R-CNN Gray training completed in $($duration.TotalMinutes.ToString('F1')) minutes" -ForegroundColor Green
    $results["maskrcnn_gray"] = "SUCCESS ($($duration.TotalMinutes.ToString('F1')) min)"
}

# ============================================================================
# 2. Mask R-CNN 3ch
# ============================================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "[2/4] Training Mask R-CNN 3ch (Gray + LSD + SDF)" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green

$startTime = Get-Date

python train_instance_seg.py `
    --model maskrcnn `
    --input-type 3ch `
    --data $DATASET `
    --epochs $EPOCHS `
    --batch $BATCH_SIZE `
    --lr $LR `
    --weight-decay $WEIGHT_DECAY `
    --scheduler $SCHEDULER_MASKRCNN `
    --trainable-layers 3 `
    --output $OUTPUT_BASE `
    --save-every 20 `
    $WANDB_FLAG

$endTime = Get-Date
$duration = $endTime - $startTime

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Mask R-CNN 3ch training failed" -ForegroundColor Red
    $results["maskrcnn_3ch"] = "FAILED"
} else {
    Write-Host "Mask R-CNN 3ch training completed in $($duration.TotalMinutes.ToString('F1')) minutes" -ForegroundColor Green
    $results["maskrcnn_3ch"] = "SUCCESS ($($duration.TotalMinutes.ToString('F1')) min)"
}

# ============================================================================
# 3. Mask2Former Gray
# ============================================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "[3/4] Training Mask2Former Gray (Grayscale only)" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green

$startTime = Get-Date

python train_instance_seg.py `
    --model mask2former `
    --input-type gray `
    --data $DATASET `
    --epochs $EPOCHS `
    --batch $BATCH_SIZE `
    --lr $LR `
    --weight-decay $WEIGHT_DECAY `
    --scheduler $SCHEDULER_MASK2FORMER `
    --output $OUTPUT_BASE `
    --save-every 20 `
    $WANDB_FLAG

$endTime = Get-Date
$duration = $endTime - $startTime

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Mask2Former Gray training failed" -ForegroundColor Red
    $results["mask2former_gray"] = "FAILED"
} else {
    Write-Host "Mask2Former Gray training completed in $($duration.TotalMinutes.ToString('F1')) minutes" -ForegroundColor Green
    $results["mask2former_gray"] = "SUCCESS ($($duration.TotalMinutes.ToString('F1')) min)"
}

# ============================================================================
# 4. Mask2Former 3ch
# ============================================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "[4/4] Training Mask2Former 3ch (Gray + LSD + SDF)" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green

$startTime = Get-Date

python train_instance_seg.py `
    --model mask2former `
    --input-type 3ch `
    --data $DATASET `
    --epochs $EPOCHS `
    --batch $BATCH_SIZE `
    --lr $LR `
    --weight-decay $WEIGHT_DECAY `
    --scheduler $SCHEDULER_MASK2FORMER `
    --output $OUTPUT_BASE `
    --save-every 20 `
    $WANDB_FLAG

$endTime = Get-Date
$duration = $endTime - $startTime

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Mask2Former 3ch training failed" -ForegroundColor Red
    $results["mask2former_3ch"] = "FAILED"
} else {
    Write-Host "Mask2Former 3ch training completed in $($duration.TotalMinutes.ToString('F1')) minutes" -ForegroundColor Green
    $results["mask2former_3ch"] = "SUCCESS ($($duration.TotalMinutes.ToString('F1')) min)"
}

# ============================================================================
# Summary
# ============================================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Training Complete!" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Results:" -ForegroundColor Yellow
Write-Host "  1. Mask R-CNN Gray:   $($results['maskrcnn_gray'])"
Write-Host "  2. Mask R-CNN 3ch:    $($results['maskrcnn_3ch'])"
Write-Host "  3. Mask2Former Gray:  $($results['mask2former_gray'])"
Write-Host "  4. Mask2Former 3ch:   $($results['mask2former_3ch'])"
Write-Host ""
Write-Host "Models saved to:" -ForegroundColor Yellow
Write-Host "  - $OUTPUT_BASE/maskrcnn_gray/"
Write-Host "  - $OUTPUT_BASE/maskrcnn_3ch/"
Write-Host "  - $OUTPUT_BASE/mask2former_gray/"
Write-Host "  - $OUTPUT_BASE/mask2former_3ch/"
Write-Host ""
Write-Host "Each model directory contains:" -ForegroundColor Yellow
Write-Host "  - best.pt           (best validation loss)"
Write-Host "  - final.pt          (final epoch)"
Write-Host "  - results.json      (training history & metrics)"
Write-Host "  - test_metrics.json (evaluation results)"
Write-Host "  - visualizations/   (prediction samples)"
Write-Host ""
Write-Host "To test a specific model:" -ForegroundColor Yellow
Write-Host "  python train_instance_seg.py --model maskrcnn --input-type gray --data $DATASET --test-only --weights $OUTPUT_BASE/maskrcnn_gray/best.pt"
Write-Host ""
