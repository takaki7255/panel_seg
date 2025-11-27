# ========================================
# Automatic Model Evaluation Script
# ========================================

$ErrorActionPreference = "Continue"

# Configuration
$modelsDir = "panel_models"
$testDataset = "frame_dataset/test100_dataset"
$resultsBaseDir = "evaluation_results"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$resultsDir = "$resultsBaseDir/$timestamp"

# Create results directory
New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Automatic Model Evaluation Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Models Directory: $modelsDir" -ForegroundColor Yellow
Write-Host "Test Dataset: $testDataset" -ForegroundColor Yellow
Write-Host "Preprocessed Data: " -ForegroundColor Yellow -NoNewline
if (Test-Path "frame_dataset/test100_preprocessed") {
    Write-Host "frame_dataset/test100_preprocessed" -ForegroundColor Green
} elseif (Test-Path "frame_dataset/1000_preprocessed") {
    Write-Host "frame_dataset/1000_preprocessed" -ForegroundColor Green
} else {
    Write-Host "Not found (LSD/SDF models cannot be evaluated)" -ForegroundColor Red
}
Write-Host "Results Directory: $resultsDir" -ForegroundColor Yellow
Write-Host ""

# Get model file list
$modelFiles = Get-ChildItem -Path $modelsDir -Filter "*.pt" -File | Sort-Object Name

if ($modelFiles.Count -eq 0) {
    Write-Host "Error: No model files found in $modelsDir" -ForegroundColor Red
    exit 1
}

Write-Host "Models found: $($modelFiles.Count)" -ForegroundColor Green
Write-Host ""

# Evaluation results summary
$summaryFile = "$resultsDir/evaluation_summary.txt"
$csvFile = "$resultsDir/evaluation_results.csv"

# CSV header
"Model,TestScript,Status,ExecutionTime,OutputFile" | Out-File -FilePath $csvFile -Encoding UTF8

# Summary header
@"
========================================
Model Evaluation Summary
Execution Time: $(Get-Date -Format "yyyy/MM/dd HH:mm:ss")
========================================

"@ | Out-File -FilePath $summaryFile -Encoding UTF8

$successCount = 0
$failCount = 0
$totalModels = $modelFiles.Count

foreach ($modelFile in $modelFiles) {
    $modelName = $modelFile.BaseName
    $modelPath = $modelFile.FullName
    
    Write-Host "----------------------------------------" -ForegroundColor Cyan
    Write-Host "[$($modelFiles.IndexOf($modelFile) + 1)/$totalModels] Evaluating: $modelName" -ForegroundColor Cyan
    Write-Host "----------------------------------------" -ForegroundColor Cyan
    
    # Determine test script from model name
    $testScript = $null
    $modelType = $null
    
    # Check preprocessed dataset (prioritize test100_preprocessed)
    $preprocessedDataset = "frame_dataset/test100_preprocessed"
    if (-not (Test-Path $preprocessedDataset)) {
        $preprocessedDataset = "frame_dataset/1000_preprocessed"
    }
    
    if ($modelName -match "unetgraylsdsdf") {
        $testScript = "test_unet_gray_lsd_sdf.py"
        $modelType = "unetgraylsdsdf"
        $datasetRoot = $preprocessedDataset  # Requires LSD+SDF
    }
    elseif ($modelName -match "unetgraylsd") {
        $testScript = "test_unet_gray_lsd.py"
        $modelType = "unetgraylsd"
        $datasetRoot = $preprocessedDataset  # Requires LSD
    }
    elseif ($modelName -match "unetgray") {
        $testScript = "test_unet_gray.py"
        $modelType = "unetgray"
        $datasetRoot = $testDataset  # Raw images only
    }
    elseif ($modelName -match "segformer") {
        $testScript = "test_segformer.py"
        $modelType = "segformer"
        $datasetRoot = $preprocessedDataset  # Requires LSD+SDF
    }
    else {
        Write-Host "  Warning: Cannot determine model type: $modelName" -ForegroundColor Yellow
        Write-Host "  Skipping" -ForegroundColor Yellow
        Write-Host ""
        "$modelName,Unknown,Skipped,0,N/A" | Out-File -FilePath $csvFile -Append -Encoding UTF8
        continue
    }
    
    Write-Host "  Model Type: $modelType" -ForegroundColor White
    Write-Host "  Test Script: $testScript" -ForegroundColor White
    Write-Host "  Dataset: $datasetRoot" -ForegroundColor White
    
    # Check test script exists
    if (-not (Test-Path $testScript)) {
        Write-Host "  Error: Test script not found: $testScript" -ForegroundColor Red
        Write-Host ""
        $failCount++
        "$modelName,$testScript,Failed (Script Not Found),0,N/A" | Out-File -FilePath $csvFile -Append -Encoding UTF8
        continue
    }
    
    # Check dataset exists
    if (-not (Test-Path $datasetRoot)) {
        Write-Host "  Error: Dataset not found: $datasetRoot" -ForegroundColor Red
        Write-Host ""
        $failCount++
        "$modelName,$testScript,Failed (Dataset Not Found),0,N/A" | Out-File -FilePath $csvFile -Append -Encoding UTF8
        continue
    }
    
    # Output filename
    $outputLog = "$resultsDir/${modelName}_evaluation.log"
    
    # Run test
    Write-Host "  Starting execution..." -ForegroundColor Green
    $startTime = Get-Date
    
    try {
        # Execute test command and save to log
        # For test100_dataset (flat structure), omit --split argument
        $condaRunArgs = @("run", "-n", "myenv", "--no-capture-output", "python", $testScript, "--model", $modelPath, "--root", $datasetRoot)
        
        # Add --split only if not using test100_dataset (which has flat structure)
        if ($datasetRoot -notmatch "test100") {
            $condaRunArgs += @("--split", "test")
        }
        
        $command = "conda run -n myenv --no-capture-output python $testScript --model $modelPath --root $datasetRoot"
        if ($datasetRoot -notmatch "test100") {
            $command += " --split test"
        }
        Write-Host "  Command: $command" -ForegroundColor Gray
        
        # Execute command and capture output
        $outputLines = @()
        & conda @condaRunArgs 2>&1 | ForEach-Object {
            $line = $_.ToString()
            Write-Host $line
            $outputLines += $line
        }
        
        $outputLines | Out-File -FilePath $outputLog -Encoding UTF8
        $exitCode = $LASTEXITCODE
        
        $endTime = Get-Date
        $duration = ($endTime - $startTime).TotalSeconds
        
        if ($exitCode -eq 0) {
            Write-Host "  Success (${duration}s)" -ForegroundColor Green
            $successCount++
            $status = "Success"
            
            # Extract important metrics from results (show last few lines)
            $lastLines = $outputLines | Select-Object -Last 10
            foreach ($line in $lastLines) {
                if ($line -match "(IoU|Dice|Precision|Recall|F1|mIoU)") {
                    Write-Host "    $line" -ForegroundColor Cyan
                }
            }
        }
        else {
            Write-Host "  Failed (exitcode: $exitCode)" -ForegroundColor Red
            $failCount++
            $status = "Failed (Exit Code $exitCode)"
        }
        
        "$modelName,$testScript,$status,$duration,$outputLog" | Out-File -FilePath $csvFile -Append -Encoding UTF8
        
    }
    catch {
        $endTime = Get-Date
        $duration = ($endTime - $startTime).TotalSeconds
        Write-Host "  Exception: $($_.Exception.Message)" -ForegroundColor Red
        $failCount++
        $status = "Failed (Exception)"
        "$modelName,$testScript,$status,$duration,$outputLog" | Out-File -FilePath $csvFile -Append -Encoding UTF8
    }
    
    Write-Host ""
}

# Final summary
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Evaluation Complete" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Total Models: $totalModels" -ForegroundColor White
Write-Host "Success: $successCount" -ForegroundColor Green
Write-Host "Failed: $failCount" -ForegroundColor Red
Write-Host ""
Write-Host "Results saved to: $resultsDir" -ForegroundColor Yellow
Write-Host "  - Summary: $summaryFile" -ForegroundColor Yellow
Write-Host "  - CSV: $csvFile" -ForegroundColor Yellow
Write-Host ""

# Append results to summary file
@"
Evaluation Results:
- Total Models: $totalModels
- Success: $successCount
- Failed: $failCount

See evaluation_results.csv for details.
Log files for each model are saved in the same directory.
"@ | Out-File -FilePath $summaryFile -Append -Encoding UTF8

# Open CSV file (optional)
# Start-Process $csvFile

Write-Host "Done!" -ForegroundColor Green
