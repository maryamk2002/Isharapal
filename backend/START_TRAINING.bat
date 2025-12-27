@echo off
echo ================================================================================
echo ISHARAPAL - Starting Advanced Model Training
echo ================================================================================
echo.
echo Training Configuration:
echo   - Phase: letters_4 (4 signs)
echo   - Model Size: base
echo   - Device: CPU
echo   - Export ONNX: Yes
echo   - Expected Time: 2-4 hours
echo.
echo Press Ctrl+C at any time to stop training
echo.
echo ================================================================================
echo.

python train_advanced_model.py --phase letters_4 --model_size base --device cpu --export_onnx --no_mixed_precision

echo.
echo ================================================================================
echo Training Complete!
echo.
echo Check your model in: saved_models\
echo.
pause

