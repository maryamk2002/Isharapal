#!/usr/bin/env python3
"""
Verify dataset files exist and fix split files by removing missing entries.
"""

import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_v2 import V2_SPLITS_DIR, FEATURES_DIR


def verify_and_fix_splits(splits_dir: Path = V2_SPLITS_DIR, features_dir: Path = FEATURES_DIR):
    """Verify files exist and remove missing entries from split files."""
    
    splits_dir = Path(splits_dir)
    features_dir = Path(features_dir)
    
    print("=" * 60)
    print("VERIFYING DATASET FILES")
    print("=" * 60)
    print(f"Features directory: {features_dir}")
    print(f"Splits directory: {splits_dir}")
    print("-" * 60)
    
    # Process each split file
    for split_name in ['train', 'val', 'test']:
        split_file = splits_dir / f"{split_name}_v2.txt"
        
        if not split_file.exists():
            print(f"⚠ {split_file} not found, skipping...")
            continue
        
        print(f"\nProcessing {split_name} split...")
        
        # Read and verify files
        valid_lines = []
        missing_files = []
        total_count = 0
        
        with open(split_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) != 2:
                    continue
                
                file_path_str, label = parts
                total_count += 1
                
                # Try to resolve file path
                file_path = Path(file_path_str)
                
                # Handle absolute paths
                if file_path.is_absolute():
                    resolved_path = file_path
                else:
                    # Relative path - try multiple locations
                    resolved_path = features_dir / file_path
                    if not resolved_path.exists() and str(file_path).startswith('data/features_temporal/'):
                        rel_part = str(file_path).replace('data/features_temporal/', '')
                        resolved_path = features_dir / rel_part
                
                # Check if file exists
                if resolved_path.exists():
                    valid_lines.append(line + '\n')
                else:
                    missing_files.append((file_path_str, label))
                    if len(missing_files) <= 10:  # Show first 10
                        print(f"  Missing: {file_path_str}")
        
        # Create backup
        backup_file = splits_dir / f"{split_name}_v2_backup_before_fix.txt"
        import shutil
        if split_file.exists():
            shutil.copy2(split_file, backup_file)
        
        # Write fixed file
        with open(split_file, 'w', encoding='utf-8') as f:
            f.writelines(valid_lines)
        
        print(f"  ✓ Valid files: {len(valid_lines)}")
        print(f"  ✗ Missing files: {len(missing_files)}")
        print(f"  Total processed: {total_count}")
        
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more missing files")
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    print("\nMissing files have been removed from split files.")
    print("Training will now skip these files automatically.")


if __name__ == "__main__":
    verify_and_fix_splits()



