#!/usr/bin/env python3
"""
Fix split files for Google Colab compatibility.
Converts absolute Windows paths to relative paths.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_v2 import V2_SPLITS_DIR, FEATURES_DIR


def fix_splits_for_colab(splits_dir: Path = V2_SPLITS_DIR, features_dir: Path = FEATURES_DIR):
    """Convert absolute paths in split files to relative paths."""
    
    splits_dir = Path(splits_dir)
    features_dir = Path(features_dir)
    
    # Get absolute path of features directory for conversion
    features_abs = features_dir.resolve()
    
    print("=" * 60)
    print("FIXING SPLITS FOR COLAB COMPATIBILITY")
    print("=" * 60)
    print(f"Splits directory: {splits_dir}")
    print(f"Features directory: {features_dir}")
    print(f"Features absolute: {features_abs}")
    print("-" * 60)
    
    # Process each split file
    for split_name in ['train', 'val', 'test']:
        split_file = splits_dir / f"{split_name}_v2.txt"
        
        if not split_file.exists():
            print(f"⚠ {split_file} not found, skipping...")
            continue
        
        print(f"\nProcessing {split_name} split...")
        
        # Read and convert paths
        new_lines = []
        fixed_count = 0
        error_count = 0
        
        with open(split_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) != 2:
                    continue
                
                file_path_str, label = parts
                file_path = Path(file_path_str)
                
                # Convert absolute path to relative
                try:
                    if file_path.is_absolute():
                        # Check if it's under features directory
                        try:
                            relative_path = file_path.relative_to(features_abs)
                            # Make it relative to backend/data/features_temporal
                            # Convert to forward slashes for cross-platform compatibility
                            new_path = f"data/features_temporal/{str(relative_path).replace(chr(92), '/')}"
                            new_lines.append(f"{new_path}\t{label}\n")
                            fixed_count += 1
                        except ValueError:
                            # Path not under features directory, try to find relative
                            if str(file_path).startswith(str(features_abs)):
                                # Extract relative part
                                rel_part = str(file_path)[len(str(features_abs)):].lstrip('\\/')
                                new_path = f"data/features_temporal/{rel_part.replace(chr(92), '/')}"
                                new_lines.append(f"{new_path}\t{label}\n")
                                fixed_count += 1
                            else:
                                # Can't convert, skip or use original
                                print(f"  Warning: Cannot convert path at line {line_num}: {file_path}")
                                error_count += 1
                    else:
                        # Already relative, use as-is
                        new_lines.append(line + '\n')
                except Exception as e:
                    print(f"  Error processing line {line_num}: {e}")
                    error_count += 1
        
        # Write fixed file
        backup_file = splits_dir / f"{split_name}_v2_backup.txt"
        
        # Create backup
        import shutil
        if split_file.exists():
            shutil.copy2(split_file, backup_file)
            print(f"  Backup created: {backup_file}")
        
        # Write new file
        with open(split_file, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        
        print(f"  ✓ Fixed {fixed_count} paths")
        print(f"  ⚠ Errors: {error_count}")
        print(f"  Total lines: {len(new_lines)}")
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print("\nSplit files now use relative paths:")
    print("  data/features_temporal/<sign>/sample_XXX.npy")
    print("\nThese will work in Google Colab!")


if __name__ == "__main__":
    fix_splits_for_colab()

