@echo off
REM ============================================================
REM PSL Training V2 - Resume Script
REM Use this AFTER training stops to resume from checkpoint
REM ============================================================

echo.
echo ============================================================
echo RESUME PSL TRAINING V2
echo ============================================================
echo.

cd /d "%~dp0backend"

set PYTHON=C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe
set EXPERIMENT=psl_v2_20251127_231748

echo Experiment: %EXPERIMENT%
echo.
echo Checking for checkpoint...
echo.

if exist "saved_models\v2\checkpoints\%EXPERIMENT%\latest_checkpoint.pth" (
    echo [OK] Checkpoint found!
    echo.
    echo Resuming training from latest checkpoint...
    echo.
    %PYTHON% training\train_pipeline_v2.py --experiment_name %EXPERIMENT% --resume latest_checkpoint.pth --epochs 100
) else (
    echo [ERROR] Checkpoint not found!
    echo.
    echo Expected location:
    echo   saved_models\v2\checkpoints\%EXPERIMENT%\latest_checkpoint.pth
    echo.
    echo Available checkpoints:
    dir /b "saved_models\v2\checkpoints\%EXPERIMENT%\*.pth" 2>nul
    echo.
    pause
    exit /b 1
)

pause

