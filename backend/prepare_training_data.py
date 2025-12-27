#!/usr/bin/env python3
"""
Split features into train/val/test sets for training.
"""

import os
import shutil
import random
from pathlib import Path
import numpy as np

# Configuration
FEATURES_DIR = Path("data/features_temporal")
LABELS = ['2-Hay', 'Alifmad', 'Aray', 'Jeem']

# Split ratios
TRAIN_RATIO = 0.70  # 70% for training
VAL_RATIO = 0.15    # 15% for validation
TEST_RATIO = 0.15   # 15% for testing

def split_data():
    """Split data into train/val/test sets."""
    
    print("=" * 80)
    print("Preparing Training Data")
    print("=" * 80)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        for label in LABELS:
            output_dir = FEATURES_DIR / split / label
            output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each label
    total_stats = {'train': 0, 'val': 0, 'test': 0}
    
    for label in LABELS:
        label_dir = FEATURES_DIR / label
        
        if not label_dir.exists():
            print(f"⚠️  Warning: {label} directory not found")
            continue
        
        # Get all .npy files
        all_files = list(label_dir.glob("*.npy"))
        
        if len(all_files) == 0:
            print(f"⚠️  Warning: No .npy files found for {label}")
            continue
        
        # Shuffle files
        random.seed(42)  # For reproducibility
        random.shuffle(all_files)
        
        # Calculate split indices
        n_total = len(all_files)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)
        
        # Split files
        train_files = all_files[:n_train]
        val_files = all_files[n_train:n_train + n_val]
        test_files = all_files[n_train + n_val:]
        
        # Update stats
        total_stats['train'] += len(train_files)
        total_stats['val'] += len(val_files)
        total_stats['test'] += len(test_files)
        
        # Copy files to splits
        print(f"\n{label}:")
        print(f"  Total: {n_total} files")
        print(f"  Train: {len(train_files)} files")
        print(f"  Val:   {len(val_files)} files")
        print(f"  Test:  {len(test_files)} files")
        
        # Copy train files
        for i, file in enumerate(train_files):
            dst = FEATURES_DIR / 'train' / label / f"{label}_train_{i:04d}.npy"
            shutil.copy2(file, dst)
        
        # Copy val files
        for i, file in enumerate(val_files):
            dst = FEATURES_DIR / 'val' / label / f"{label}_val_{i:04d}.npy"
            shutil.copy2(file, dst)
        
        # Copy test files
        for i, file in enumerate(test_files):
            dst = FEATURES_DIR / 'test' / label / f"{label}_test_{i:04d}.npy"
            shutil.copy2(file, dst)
    
    print("\n" + "=" * 80)
    print("✅ Data preparation complete!")
    print("=" * 80)
    print("\nSummary:")
    print(f"  Train: {total_stats['train']} total samples")
    print(f"  Val:   {total_stats['val']} total samples")
    print(f"  Test:  {total_stats['test']} total samples")
    print(f"  Total: {sum(total_stats.values())} samples")
    
    print("\n📁 Data structure:")
    print(f"  {FEATURES_DIR}/")
    print(f"    ├── train/")
    print(f"    │   ├── 2-Hay/ ({len(list((FEATURES_DIR / 'train' / '2-Hay').glob('*.npy')))} files)")
    print(f"    │   ├── Alifmad/ ({len(list((FEATURES_DIR / 'train' / 'Alifmad').glob('*.npy')))} files)")
    print(f"    │   ├── Aray/ ({len(list((FEATURES_DIR / 'train' / 'Aray').glob('*.npy')))} files)")
    print(f"    │   └── Jeem/ ({len(list((FEATURES_DIR / 'train' / 'Jeem').glob('*.npy')))} files)")
    print(f"    ├── val/ ({total_stats['val']} total)")
    print(f"    └── test/ ({total_stats['test']} total)")
    
    print("\n🚀 Ready to train! Run:")
    print("   python train_advanced_model.py --phase letters_4 --model_size base --device cpu --export_onnx")
    print("=" * 80)

if __name__ == "__main__":
    split_data()

