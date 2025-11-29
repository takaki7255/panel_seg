# ============================================================================
# Test All Instance Segmentation Models
# ============================================================================
#
# 使用方法:
#   .\test_all_instance_models.ps1 -Timestamp "20251128_123456"
#
# または最新のモデルを自動検出:
#   .\test_all_instance_models.ps1
#
# ============================================================================

param(
    [string]$Timestamp = ""
)

$ErrorActionPreference = "Stop"

# Configuration
$TEST_DATASET_GRAY = "./frame_dataset/test100_instance"
$TEST_DATASET_3CH = "./frame_dataset/test100_instance"
$MODEL_BASE = "./panel_models"
$OUTPUT_BASE = "./instance_results"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Instance Segmentation Models Testing" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Find latest models if timestamp not provided
if ($Timestamp -eq "") {
    # Find the latest mask2former_gray folder
    $latestFolder = Get-ChildItem -Path $MODEL_BASE -Directory -Filter "mask2former_gray_*" | 
                    Sort-Object Name -Descending | 
                    Select-Object -First 1
    
    if ($latestFolder) {
        $Timestamp = $latestFolder.Name -replace "mask2former_gray_", ""
        Write-Host "Auto-detected timestamp: $Timestamp" -ForegroundColor Yellow
    } else {
        Write-Host "ERROR: No trained models found. Run train_all_instance_models.ps1 first." -ForegroundColor Red
        exit 1
    }
}

# Model paths
$M2F_GRAY_WEIGHTS = "$MODEL_BASE/mask2former_gray_$Timestamp/mask2former_gray_best.pt"
$M2F_3CH_WEIGHTS = "$MODEL_BASE/mask2former/mask2former_best.pt"
$MRCNN_GRAY_WEIGHTS = "$MODEL_BASE/maskrcnn_gray_$Timestamp/maskrcnn_gray_best.pt"
$MRCNN_3CH_WEIGHTS = "$MODEL_BASE/maskrcnn_3ch_$Timestamp/maskrcnn_3ch_best.pt"

Write-Host "Timestamp: $Timestamp"
Write-Host "============================================================" -ForegroundColor Cyan

# ============================================================================
# 1. Test Mask2Former Gray
# ============================================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "[1/4] Testing Mask2Former Gray" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green

if (Test-Path $M2F_GRAY_WEIGHTS) {
    $OUTPUT_M2F_GRAY = "$OUTPUT_BASE/mask2former_gray_$Timestamp"
    
    python test_mask2former_gray.py `
        --model $M2F_GRAY_WEIGHTS `
        --root $TEST_DATASET_GRAY `
        --save-preds `
        --output $OUTPUT_M2F_GRAY
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Mask2Former Gray testing completed!" -ForegroundColor Green
    }
} else {
    Write-Host "SKIP: Weights not found at $M2F_GRAY_WEIGHTS" -ForegroundColor Yellow
}

# ============================================================================
# 2. Test Mask2Former 3ch
# ============================================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "[2/4] Testing Mask2Former 3ch" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green

if (Test-Path $M2F_3CH_WEIGHTS) {
    $OUTPUT_M2F_3CH = "$OUTPUT_BASE/mask2former_3ch_$Timestamp"
    
    python test_mask2former.py `
        --model $M2F_3CH_WEIGHTS `
        --root $TEST_DATASET_3CH `
        --save-preds `
        --output $OUTPUT_M2F_3CH
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Mask2Former 3ch testing completed!" -ForegroundColor Green
    }
} else {
    Write-Host "SKIP: Weights not found at $M2F_3CH_WEIGHTS" -ForegroundColor Yellow
}

# ============================================================================
# 3. Test Mask R-CNN Gray
# ============================================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "[3/4] Testing Mask R-CNN Gray" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green

if (Test-Path $MRCNN_GRAY_WEIGHTS) {
    $OUTPUT_MRCNN_GRAY = "$OUTPUT_BASE/maskrcnn_gray_$Timestamp"
    
    python test_maskrcnn.py `
        --input $TEST_DATASET_GRAY `
        --weights $MRCNN_GRAY_WEIGHTS `
        --input-type gray `
        --output $OUTPUT_MRCNN_GRAY
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Mask R-CNN Gray testing completed!" -ForegroundColor Green
    }
} else {
    Write-Host "SKIP: Weights not found at $MRCNN_GRAY_WEIGHTS" -ForegroundColor Yellow
}

# ============================================================================
# 4. Test Mask R-CNN 3ch
# ============================================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "[4/4] Testing Mask R-CNN 3ch" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green

if (Test-Path $MRCNN_3CH_WEIGHTS) {
    $OUTPUT_MRCNN_3CH = "$OUTPUT_BASE/maskrcnn_3ch_$Timestamp"
    
    python test_maskrcnn.py `
        --input $TEST_DATASET_3CH `
        --weights $MRCNN_3CH_WEIGHTS `
        --input-type 3ch `
        --output $OUTPUT_MRCNN_3CH
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Mask R-CNN 3ch testing completed!" -ForegroundColor Green
    }
} else {
    Write-Host "SKIP: Weights not found at $MRCNN_3CH_WEIGHTS" -ForegroundColor Yellow
}

# ============================================================================
# Summary
# ============================================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Testing Complete!" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Results saved to:" -ForegroundColor Yellow
Write-Host "  1. $OUTPUT_BASE/mask2former_gray_$Timestamp"
Write-Host "  2. $OUTPUT_BASE/mask2former_3ch_$Timestamp"
Write-Host "  3. $OUTPUT_BASE/maskrcnn_gray_$Timestamp"
Write-Host "  4. $OUTPUT_BASE/maskrcnn_3ch_$Timestamp"
Write-Host ""
Write-Host "Check results.json in each folder for metrics." -ForegroundColor Yellow
Write-Host ""
