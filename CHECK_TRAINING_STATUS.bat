@echo off
REM ============================================================
REM Check Training Status
REM ============================================================

echo.
echo ============================================================
echo PSL TRAINING STATUS CHECK
echo ============================================================
echo.

cd /d "%~dp0backend"

if exist "logs\v2\training_output.log" (
    echo [LATEST EPOCH COMPLETED]
    echo --------------------------------------------------------
    powershell -Command "Get-Content 'logs\v2\training_output.log' | Select-String 'Epoch \d+/100' | Select-Object -Last 1"
    echo.
    echo [CURRENT PROGRESS]
    echo --------------------------------------------------------
    powershell -Command "Get-Content 'logs\v2\training_output.log' -Tail 3"
    echo.
    echo [CHECKPOINTS AVAILABLE]
    echo --------------------------------------------------------
    if exist "saved_models\v2\checkpoints\psl_v2_20251127_231748\latest_checkpoint.pth" (
        echo [OK] latest_checkpoint.pth exists
    ) else (
        echo [NOT FOUND] latest_checkpoint.pth
    )
    dir /b "saved_models\v2\checkpoints\psl_v2_20251127_231748\*.pth" 2>nul
) else (
    echo [ERROR] Training log not found
    echo Training may not be running or hasn't started yet.
)

echo.
echo ============================================================
pause

