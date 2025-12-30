@echo off
echo ============================================================
echo ISHARAPAL - ONNX Model Quantization
echo ============================================================
echo.
echo This will create a smaller, faster quantized model.
echo The original model will NOT be modified.
echo.
echo Output: frontend/models/psl_model_v2_int8.onnx
echo.
pause

cd /d "%~dp0"
python backend/export_quantized_onnx.py

echo.
pause

