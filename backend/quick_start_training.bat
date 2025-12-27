@echo off
REM Quick Start Training Script for ISHARAPAL Advanced Model
REM Windows Batch File

echo ================================================================================
echo ISHARAPAL - Advanced Model Training Quick Start
echo ================================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo [1/5] Checking Python version...
python --version
echo.

echo [2/5] Checking dependencies...
python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>nul
if errorlevel 1 (
    echo WARNING: PyTorch not found. Installing dependencies...
    pip install -r requirements.txt
)
echo.

echo [3/5] Checking data directory...
if not exist "data\features_temporal\train" (
    echo WARNING: Training data not found!
    echo.
    echo Please prepare your data first:
    echo   1. Place videos in: data\Pakistan Sign Language Urdu Alphabets\
    echo   2. Run: python extract_features.py
    echo.
    pause
    exit /b 1
)
echo Data directory found!
echo.

echo [4/5] Configuration:
echo   - Phase: letters_4 ^(4 signs^)
echo   - Model Size: base
echo   - Epochs: 100
echo   - Device: Auto-detect ^(GPU if available^)
echo.

echo [5/5] Starting training...
echo.
echo ================================================================================
echo.

python train_advanced_model.py ^
    --phase letters_4 ^
    --model_size base ^
    --epochs 100 ^
    --export_onnx

echo.
echo ================================================================================
echo Training completed!
echo.
echo Your trained model is in: saved_models\
echo.
echo Next steps:
echo   1. Check training_history.json for accuracy curves
echo   2. Use best_model.pth or best_model.onnx for inference
echo   3. Integrate with your app using backend\inference\optimized_predictor.py
echo.
echo ================================================================================
pause

