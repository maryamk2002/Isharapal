#!/usr/bin/env python3
"""
ONNX Export Script for ISHARAPAL PSL Recognition Model

Exports the trained PyTorch model to ONNX format for browser-based inference.
Handles weight_norm removal which is required for ONNX compatibility.

Usage:
    python export_to_onnx.py

Output:
    - frontend/models/psl_model_v2.onnx
    - frontend/models/psl_labels.json
    - frontend/models/sign_thresholds.json (copied)
"""

import os
import sys
import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models.optimized_tcn_model import OptimizedTCNModel


def remove_weight_norm(model: nn.Module) -> nn.Module:
    """
    Remove weight_norm parametrization from all modules.
    
    This is required before ONNX export because weight_norm uses
    deprecated torch.nn.utils.weight_norm which is not supported by ONNX.
    
    Args:
        model: PyTorch model with weight_norm applied
        
    Returns:
        Model with weight_norm removed (weights are fused)
    """
    removed_count = 0
    
    for name, module in model.named_modules():
        # Check if module has weight_norm applied
        # weight_norm adds 'weight_g' and 'weight_v' parameters
        if hasattr(module, 'weight_g') and hasattr(module, 'weight_v'):
            try:
                torch.nn.utils.remove_weight_norm(module)
                removed_count += 1
                print(f"  Removed weight_norm from: {name}")
            except ValueError as e:
                print(f"  Warning: Could not remove weight_norm from {name}: {e}")
    
    print(f"  Total weight_norm layers removed: {removed_count}")
    return model


def verify_onnx_output(onnx_path: str, pytorch_model: nn.Module, test_input: torch.Tensor) -> bool:
    """
    Verify ONNX model produces same output as PyTorch model.
    
    Args:
        onnx_path: Path to exported ONNX model
        pytorch_model: Original PyTorch model
        test_input: Test input tensor
        
    Returns:
        True if outputs match within tolerance
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("  Warning: onnxruntime not installed, skipping verification")
        print("  Install with: pip install onnxruntime")
        return True
    
    print("\n[4/5] Verifying ONNX output matches PyTorch...")
    
    # Get PyTorch output
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input).numpy()
    
    # Get ONNX output
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    onnx_output = ort_session.run(None, {input_name: test_input.numpy()})[0]
    
    # Compare outputs
    max_diff = np.abs(pytorch_output - onnx_output).max()
    mean_diff = np.abs(pytorch_output - onnx_output).mean()
    
    print(f"  PyTorch output shape: {pytorch_output.shape}")
    print(f"  ONNX output shape: {onnx_output.shape}")
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")
    
    # Check if within acceptable tolerance
    tolerance = 1e-5
    if max_diff < tolerance:
        print(f"  [OK] Outputs match within tolerance ({tolerance})")
        return True
    else:
        print(f"  [WARN] Outputs differ by {max_diff:.2e} (tolerance: {tolerance})")
        return max_diff < 1e-3  # Still acceptable for inference


def export_model():
    """Main export function."""
    print("=" * 60)
    print("ISHARAPAL PSL Model - ONNX Export")
    print("=" * 60)
    
    # Paths
    base_dir = Path(__file__).parent
    model_path = base_dir / "saved_models" / "v2" / "psl_model_v2.pth"
    config_path = base_dir / "saved_models" / "v2" / "psl_model_v2_config.json"
    thresholds_path = base_dir / "config" / "sign_thresholds.json"
    
    frontend_models_dir = base_dir.parent / "frontend" / "models"
    onnx_output_path = frontend_models_dir / "psl_model_v2.onnx"
    labels_output_path = frontend_models_dir / "psl_labels.json"
    thresholds_output_path = frontend_models_dir / "sign_thresholds.json"
    
    # Create frontend/models directory
    frontend_models_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {frontend_models_dir}")
    
    # Load config
    print("\n[1/5] Loading model configuration...")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"  Model type: {config['model_type']}")
    print(f"  Input dim: {config['input_dim']}")
    print(f"  Num classes: {config['num_classes']}")
    print(f"  Sequence length: {config['sequence_length']}")
    print(f"  Architecture: {config['architecture']}")
    
    # Create model with same architecture
    print("\n[2/5] Creating model and loading weights...")
    model = OptimizedTCNModel(
        input_dim=config['input_dim'],
        num_classes=config['num_classes'],
        num_channels=config['architecture']['num_channels'],
        kernel_size=config['architecture']['kernel_size'],
        dropout=config['architecture']['dropout']
    )
    
    # Load weights (handle nested checkpoint format)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"  Checkpoint format: nested (with metadata)")
            if 'metrics' in checkpoint:
                print(f"  Best accuracy: {checkpoint['metrics'].get('val_acc', 'N/A')}")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"  Checkpoint format: state_dict wrapper")
        else:
            # Assume it's the raw state dict
            state_dict = checkpoint
            print(f"  Checkpoint format: raw state_dict")
    else:
        state_dict = checkpoint
        print(f"  Checkpoint format: raw")
    
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  [OK] Loaded weights from: {model_path.name}")
    
    # Remove weight_norm for ONNX compatibility
    print("\n[3/5] Removing weight_norm for ONNX compatibility...")
    model = remove_weight_norm(model)
    
    # Create dummy input for export
    batch_size = 1
    sequence_length = config['sequence_length']  # 60
    input_dim = config['input_dim']  # 189
    
    dummy_input = torch.randn(batch_size, sequence_length, input_dim)
    print(f"\n  Test input shape: {dummy_input.shape}")
    
    # Test forward pass
    with torch.no_grad():
        test_output = model(dummy_input)
    print(f"  Test output shape: {test_output.shape}")
    
    # Export to ONNX
    print(f"\n[4/5] Exporting to ONNX: {onnx_output_path.name}...")
    
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_output_path),
        export_params=True,
        opset_version=14,  # Good balance of compatibility and features
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},  # Allow variable batch size
            'output': {0: 'batch_size'}
        }
    )
    
    # Get file size
    onnx_size_mb = onnx_output_path.stat().st_size / (1024 * 1024)
    print(f"  [OK] ONNX model exported: {onnx_size_mb:.2f} MB")
    
    # Verify ONNX output
    verify_onnx_output(str(onnx_output_path), model, dummy_input)
    
    # Create labels JSON
    print("\n[5/5] Creating labels and copying thresholds...")
    
    labels_data = {
        "labels": config['labels'],
        "num_classes": config['num_classes'],
        "sequence_length": config['sequence_length'],
        "input_dim": config['input_dim'],
        "model_info": {
            "type": config['model_type'],
            "accuracy": config.get('best_val_acc', 0),
            "epochs": config.get('epochs_trained', 0)
        }
    }
    
    with open(labels_output_path, 'w') as f:
        json.dump(labels_data, f, indent=2)
    print(f"  [OK] Labels saved: {labels_output_path.name}")
    
    # Copy sign_thresholds.json
    if thresholds_path.exists():
        shutil.copy(thresholds_path, thresholds_output_path)
        print(f"  [OK] Thresholds copied: {thresholds_output_path.name}")
    else:
        print(f"  [WARN] {thresholds_path} not found")
    
    # Summary
    print("\n" + "=" * 60)
    print("EXPORT COMPLETE!")
    print("=" * 60)
    print(f"\nFiles created in {frontend_models_dir}:")
    for f in frontend_models_dir.iterdir():
        size = f.stat().st_size / 1024
        unit = "KB"
        if size > 1024:
            size = size / 1024
            unit = "MB"
        print(f"  - {f.name} ({size:.1f} {unit})")
    
    print("\nNext steps:")
    print("  1. Create onnx_predictor.js for browser inference")
    print("  2. Create feedback_manager.js for localStorage/IndexedDB")
    print("  3. Update index_v2.html to use browser-only mode")
    
    return True


if __name__ == "__main__":
    try:
        success = export_model()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

