@echo off
REM ========================================
REM Automatic Model Evaluation Script (Batch)
REM ========================================

echo ========================================
echo Running Model Evaluation Script
echo ========================================
echo.

REM Execute PowerShell script
powershell -ExecutionPolicy Bypass -File evaluate_all_models.ps1

echo.
echo ========================================
echo Script Execution Complete
echo ========================================
pause
