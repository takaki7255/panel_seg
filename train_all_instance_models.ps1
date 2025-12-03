# ============================================================================
# Train All Instance Segmentation Models
# ============================================================================
# 
# 使用方法:
#   .\train_all_instance_models.ps1
#
# 前提条件:
#   - instance_dataset/5000-instance が存在すること
#   - 存在しない場合は以下を実行:
#     python create_instance_dataset.py --name 5000-instance --total 5000
#
# ============================================================================

$ErrorActionPreference = "Stop"

# Configuration
$DATASET = "./instance_dataset/5000-instance"
$EPOCHS = 100
$BATCH_SIZE = 8
$LR = "1e-4"
$WEIGHT_DECAY = 0.01
$OUTPUT_BASE = "./instance_models"
$USE_WANDB = $true

# Mask2Former specific settings
$WARMUP_EPOCHS = 10   # 10% of epochs for Transformer warmup
# Note: Dropout disabled for Mask2Former due to attention mask compatibility issues

# Timestamp for this training run
$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Instance Segmentation Models Training" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Dataset: $DATASET"
Write-Host "Epochs: $EPOCHS"
Write-Host "Batch Size: $BATCH_SIZE"
Write-Host "Learning Rate: $LR"
Write-Host "Warmup Epochs: $WARMUP_EPOCHS (Mask2Former only)"
Write-Host "Timestamp: $TIMESTAMP"
Write-Host "============================================================" -ForegroundColor Cyan

# Check if dataset exists
if (-not (Test-Path $DATASET)) {
    Write-Host "ERROR: Dataset not found at $DATASET" -ForegroundColor Red
    Write-Host "Please run first:" -ForegroundColor Yellow
    Write-Host "  python create_instance_dataset.py --name 5000-instance --total 5000" -ForegroundColor Yellow
    exit 1
}

# WandB flag
$WANDB_FLAG = if ($USE_WANDB) { "--wandb" } else { "" }

# ============================================================================
# 1. Mask2Former Gray (1ch input)
# ============================================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "[1/4] Training Mask2Former Gray (Grayscale only)" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green

$OUTPUT_M2F_GRAY = "$OUTPUT_BASE/mask2former_gray_$TIMESTAMP"
Write-Host "Output: $OUTPUT_M2F_GRAY"

python train_mask2former_gray.py `
    --root $DATASET `
    --epochs $EPOCHS `
    --batch $BATCH_SIZE `
    --lr $LR `
    --weight-decay $WEIGHT_DECAY `
    --warmup-epochs $WARMUP_EPOCHS `
    --output $OUTPUT_M2F_GRAY `
    $WANDB_FLAG

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Mask2Former Gray training failed" -ForegroundColor Red
} else {
    Write-Host "Mask2Former Gray training completed!" -ForegroundColor Green
}

# ============================================================================
# 2. Mask2Former 3ch (Gray + LSD + SDF)
# ============================================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "[2/4] Training Mask2Former 3ch (Gray + LSD + SDF)" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green

$OUTPUT_M2F_3CH = "$OUTPUT_BASE/mask2former_3ch_$TIMESTAMP"
Write-Host "Output: $OUTPUT_M2F_3CH"

python train_mask2former.py `
    --root $DATASET `
    --epochs $EPOCHS `
    --batch $BATCH_SIZE `
    --lr $LR `
    --weight-decay $WEIGHT_DECAY `
    --warmup-epochs $WARMUP_EPOCHS `
    --output $OUTPUT_M2F_3CH `
    $WANDB_FLAG

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Mask2Former 3ch training failed" -ForegroundColor Red
} else {
    Write-Host "Mask2Former 3ch training completed!" -ForegroundColor Green
}

# ============================================================================
# 3. Mask R-CNN Gray (1ch input)
# ============================================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "[3/4] Training Mask R-CNN Gray (Grayscale only)" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green

$OUTPUT_MRCNN_GRAY = "$OUTPUT_BASE/maskrcnn_gray_$TIMESTAMP"
Write-Host "Output: $OUTPUT_MRCNN_GRAY"

python train_maskrcnn.py `
    --root $DATASET `
    --input-type gray `
    --epochs $EPOCHS `
    --batch $BATCH_SIZE `
    --lr $LR `
    --weight-decay $WEIGHT_DECAY `
    --output $OUTPUT_MRCNN_GRAY `
    $WANDB_FLAG

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Mask R-CNN Gray training failed" -ForegroundColor Red
} else {
    Write-Host "Mask R-CNN Gray training completed!" -ForegroundColor Green
}

# ============================================================================
# 4. Mask R-CNN 3ch (Gray + LSD + SDF)
# ============================================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "[4/4] Training Mask R-CNN 3ch (Gray + LSD + SDF)" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green

$OUTPUT_MRCNN_3CH = "$OUTPUT_BASE/maskrcnn_3ch_$TIMESTAMP"
Write-Host "Output: $OUTPUT_MRCNN_3CH"

python train_maskrcnn.py `
    --root $DATASET `
    --input-type 3ch `
    --epochs $EPOCHS `
    --batch $BATCH_SIZE `
    --lr $LR `
    --weight-decay $WEIGHT_DECAY `
    --output $OUTPUT_MRCNN_3CH `
    $WANDB_FLAG

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Mask R-CNN 3ch training failed" -ForegroundColor Red
} else {
    Write-Host "Mask R-CNN 3ch training completed!" -ForegroundColor Green
}

# ============================================================================
# Summary
# ============================================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Training Complete!" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Models saved to:" -ForegroundColor Yellow
Write-Host "  1. Mask2Former Gray: $OUTPUT_M2F_GRAY"
Write-Host "  2. Mask2Former 3ch:  $OUTPUT_M2F_3CH"
Write-Host "  3. Mask R-CNN Gray:  $OUTPUT_MRCNN_GRAY"
Write-Host "  4. Mask R-CNN 3ch:   $OUTPUT_MRCNN_3CH"
Write-Host ""
Write-Host "To test models, run:" -ForegroundColor Yellow
Write-Host "  .\test_all_instance_models.ps1"
Write-Host ""
