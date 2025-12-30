#!/usr/bin/env python3
"""
ONNX Model Quantization Script for ISHARAPAL

Creates a quantized (int8) version of the ONNX model for faster inference.
The quantized model is ~3-4x smaller and ~2x faster.

Usage:
    python export_quantized_onnx.py

This script:
1. Loads the existing ONNX model
2. Applies dynamic quantization (no calibration data needed)
3. Saves a new quantized model (does NOT modify original)

Output:
    frontend/models/psl_model_v2_int8.onnx
"""

import os
import sys
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed."""
    try:
        import onnx
        import onnxruntime
        print(f"✓ ONNX version: {onnx.__version__}")
        print(f"✓ ONNX Runtime version: {onnxruntime.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nInstall required packages:")
        print("  pip install onnx onnxruntime")
        return False


def quantize_model():
    """Quantize the ONNX model using dynamic quantization."""
    
    # Paths
    base_dir = Path(__file__).resolve().parent.parent
    input_model = base_dir / "frontend" / "models" / "psl_model_v2.onnx"
    output_model = base_dir / "frontend" / "models" / "psl_model_v2_int8.onnx"
    
    # Check input exists
    if not input_model.exists():
        print(f"✗ Input model not found: {input_model}")
        print("\nMake sure the ONNX model exists at:")
        print(f"  {input_model}")
        return False
    
    print(f"\n{'='*60}")
    print("ONNX MODEL QUANTIZATION")
    print(f"{'='*60}")
    print(f"Input:  {input_model}")
    print(f"Output: {output_model}")
    
    # Get original size
    original_size = input_model.stat().st_size / (1024 * 1024)
    print(f"Original size: {original_size:.2f} MB")
    
    try:
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        print("\n⏳ Loading model...")
        model = onnx.load(str(input_model))
        
        # Verify model
        print("⏳ Verifying model...")
        onnx.checker.check_model(model)
        print("✓ Model verification passed")
        
        # Quantize
        print("\n⏳ Applying dynamic quantization (int8)...")
        quantize_dynamic(
            model_input=str(input_model),
            model_output=str(output_model),
            weight_type=QuantType.QInt8,
            optimize_model=True,
            per_channel=False,  # Simpler quantization
            reduce_range=False
        )
        
        # Check output
        if output_model.exists():
            quantized_size = output_model.stat().st_size / (1024 * 1024)
            reduction = (1 - quantized_size / original_size) * 100
            
            print(f"\n{'='*60}")
            print("QUANTIZATION COMPLETE")
            print(f"{'='*60}")
            print(f"✓ Quantized model saved: {output_model}")
            print(f"✓ Original size:   {original_size:.2f} MB")
            print(f"✓ Quantized size:  {quantized_size:.2f} MB")
            print(f"✓ Size reduction:  {reduction:.1f}%")
            print(f"\nExpected speedup: ~1.5-2x faster inference")
            
            # Verify quantized model
            print("\n⏳ Verifying quantized model...")
            quantized_model = onnx.load(str(output_model))
            onnx.checker.check_model(quantized_model)
            print("✓ Quantized model verification passed")
            
            print(f"\n{'='*60}")
            print("NEXT STEPS")
            print(f"{'='*60}")
            print("1. Test the quantized model for accuracy")
            print("2. If accuracy is acceptable, update index_browser.html to use it:")
            print("   Change: models/psl_model_v2.onnx")
            print("   To:     models/psl_model_v2_int8.onnx")
            print("")
            
            return True
        else:
            print("✗ Quantization failed - output file not created")
            return False
            
    except Exception as e:
        print(f"\n✗ Quantization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quantized_model():
    """Quick test of the quantized model."""
    
    base_dir = Path(__file__).resolve().parent.parent
    quantized_model = base_dir / "frontend" / "models" / "psl_model_v2_int8.onnx"
    
    if not quantized_model.exists():
        print("Quantized model not found. Run quantization first.")
        return False
    
    try:
        import numpy as np
        import onnxruntime as ort
        
        print("\n⏳ Testing quantized model inference...")
        
        # Create session
        session = ort.InferenceSession(
            str(quantized_model),
            providers=['CPUExecutionProvider']
        )
        
        # Get input info
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        print(f"  Input: {input_name}, shape: {input_shape}")
        
        # Create dummy input
        dummy_input = np.random.randn(1, 60, 189).astype(np.float32)
        
        # Run inference
        import time
        times = []
        for _ in range(10):
            start = time.perf_counter()
            outputs = session.run(None, {input_name: dummy_input})
            times.append((time.perf_counter() - start) * 1000)
        
        avg_time = np.mean(times[1:])  # Skip first (warm-up)
        print(f"  Output shape: {outputs[0].shape}")
        print(f"  Average inference time: {avg_time:.2f} ms")
        print("✓ Quantized model test passed!")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


if __name__ == "__main__":
    print("="*60)
    print("ISHARAPAL - ONNX Model Quantization Tool")
    print("="*60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Quantize
    if quantize_model():
        # Test
        test_quantized_model()
    else:
        sys.exit(1)

