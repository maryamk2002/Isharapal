@echo off
REM ============================================================
REM PSL Recognition V2 - Training Script
REM Train model on ALL 40 Urdu alphabet signs
REM ============================================================

echo.
echo ============================================================
echo PSL RECOGNITION V2 - TRAINING ON 40 SIGNS
echo ============================================================
echo.

REM Change to backend directory
cd /d "%~dp0backend"

REM Python path
set PYTHON=C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe

REM Check if Python exists
if not exist "%PYTHON%" (
    echo ERROR: Python not found at %PYTHON%
    echo Please update the PYTHON path in this script.
    pause
    exit /b 1
)

echo Python: %PYTHON%
echo Working Directory: %CD%
echo.

REM Run training
echo Starting training...
echo This may take several hours on CPU.
echo Progress will be logged to backend\logs\v2\
echo.

%PYTHON% training\train_pipeline_v2.py --epochs 100 --lr 0.0005

echo.
echo ============================================================
echo Training complete!
echo Check backend\saved_models\v2\ for the trained model.
echo ============================================================
pause

