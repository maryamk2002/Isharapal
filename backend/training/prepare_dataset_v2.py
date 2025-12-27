#!/usr/bin/env python3
"""
Dataset Preparation Script V2 for PSL Recognition System.
Prepares dataset for training on ALL Urdu alphabet signs (40 classes).

Features:
- Scans and verifies all extracted features
- Creates train/val/test splits
- Generates dataset statistics report
- Handles class imbalance detection
"""

import os
import sys
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from datetime import datetime

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_v2 import (
    FEATURES_DIR, V2_SPLITS_DIR, V2_LOGS_DIR,
    get_available_signs, get_dataset_stats,
    training_config_v2, dataset_config_v2
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(V2_LOGS_DIR / 'dataset_preparation.log')
    ]
)
logger = logging.getLogger(__name__)


class DatasetPreparerV2:
    """Prepare and verify dataset for v2 training."""
    
    def __init__(
        self,
        features_dir: Path = FEATURES_DIR,
        splits_dir: Path = V2_SPLITS_DIR,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        min_samples_per_class: int = 30,
        random_seed: int = 42
    ):
        self.features_dir = Path(features_dir)
        self.splits_dir = Path(splits_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.min_samples_per_class = min_samples_per_class
        self.random_seed = random_seed
        
        # Validation
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Splits must sum to 1.0"
        
        # Create output directory
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        
        # Results
        self.available_signs = []
        self.excluded_signs = []
        self.dataset_info = {}
        
    def scan_dataset(self) -> Dict[str, Any]:
        """Scan the dataset and collect statistics."""
        logger.info("=" * 60)
        logger.info("SCANNING DATASET")
        logger.info("=" * 60)
        
        if not self.features_dir.exists():
            raise FileNotFoundError(f"Features directory not found: {self.features_dir}")
        
        # Collect all sign directories
        sign_dirs = [
            d for d in self.features_dir.iterdir()
            if d.is_dir() and d.name not in ['train', 'val', 'test']
        ]
        
        stats = {
            'total_signs': 0,
            'included_signs': 0,
            'excluded_signs': 0,
            'total_samples': 0,
            'samples_per_sign': {},
            'excluded_reasons': {},
            'feature_shapes': {},
            'issues': []
        }
        
        for sign_dir in sorted(sign_dirs):
            sign_name = sign_dir.name
            npy_files = list(sign_dir.glob("*.npy"))
            num_samples = len(npy_files)
            
            # Check minimum samples
            if num_samples < self.min_samples_per_class:
                self.excluded_signs.append(sign_name)
                stats['excluded_signs'] += 1
                stats['excluded_reasons'][sign_name] = f"Too few samples ({num_samples} < {self.min_samples_per_class})"
                logger.warning(f"  ⚠ {sign_name}: Excluded - only {num_samples} samples")
                continue
            
            # Verify feature files
            valid_samples = 0
            shapes = set()
            
            for npy_file in npy_files[:10]:  # Sample first 10 files
                try:
                    data = np.load(npy_file)
                    shapes.add(data.shape)
                    if data.shape[-1] == 189:  # Expected feature dimension
                        valid_samples += 1
                except Exception as e:
                    stats['issues'].append(f"{sign_name}/{npy_file.name}: {str(e)}")
            
            if len(shapes) > 1:
                stats['issues'].append(f"{sign_name}: Inconsistent shapes - {shapes}")
            
            # Include this sign
            self.available_signs.append(sign_name)
            stats['samples_per_sign'][sign_name] = num_samples
            stats['total_samples'] += num_samples
            stats['included_signs'] += 1
            stats['feature_shapes'][sign_name] = list(shapes)[0] if shapes else None
            
            logger.info(f"  ✓ {sign_name}: {num_samples} samples")
        
        stats['total_signs'] = len(sign_dirs)
        
        # Summary
        logger.info("-" * 60)
        logger.info(f"Total signs found: {stats['total_signs']}")
        logger.info(f"Included signs: {stats['included_signs']}")
        logger.info(f"Excluded signs: {stats['excluded_signs']}")
        logger.info(f"Total samples: {stats['total_samples']}")
        
        # Class imbalance check
        if stats['samples_per_sign']:
            min_samples = min(stats['samples_per_sign'].values())
            max_samples = max(stats['samples_per_sign'].values())
            imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
            stats['imbalance_ratio'] = imbalance_ratio
            
            if imbalance_ratio > 10:
                logger.warning(f"⚠ High class imbalance detected: {imbalance_ratio:.1f}x")
                logger.warning(f"  Min samples: {min_samples}, Max samples: {max_samples}")
        
        self.dataset_info = stats
        return stats
    
    def create_splits(self) -> Tuple[List[str], List[str], List[str]]:
        """Create train/val/test splits."""
        logger.info("=" * 60)
        logger.info("CREATING DATA SPLITS")
        logger.info("=" * 60)
        
        if not self.available_signs:
            raise ValueError("No signs available. Run scan_dataset() first.")
        
        random.seed(self.random_seed)
        
        train_files = []
        val_files = []
        test_files = []
        
        for sign_name in self.available_signs:
            sign_dir = self.features_dir / sign_name
            npy_files = sorted(list(sign_dir.glob("*.npy")))
            
            # Shuffle files
            random.shuffle(npy_files)
            
            # Calculate split indices
            n_samples = len(npy_files)
            n_train = int(n_samples * self.train_ratio)
            n_val = int(n_samples * self.val_ratio)
            
            # Split files
            train_subset = npy_files[:n_train]
            val_subset = npy_files[n_train:n_train + n_val]
            test_subset = npy_files[n_train + n_val:]
            
            # Create entries with labels
            for f in train_subset:
                train_files.append((str(f), sign_name))
            for f in val_subset:
                val_files.append((str(f), sign_name))
            for f in test_subset:
                test_files.append((str(f), sign_name))
            
            logger.info(f"  {sign_name}: train={len(train_subset)}, val={len(val_subset)}, test={len(test_subset)}")
        
        # Shuffle final lists
        random.shuffle(train_files)
        random.shuffle(val_files)
        random.shuffle(test_files)
        
        logger.info("-" * 60)
        logger.info(f"Train samples: {len(train_files)}")
        logger.info(f"Val samples: {len(val_files)}")
        logger.info(f"Test samples: {len(test_files)}")
        
        return train_files, val_files, test_files
    
    def save_splits(
        self,
        train_files: List[Tuple[str, str]],
        val_files: List[Tuple[str, str]],
        test_files: List[Tuple[str, str]]
    ) -> Dict[str, Path]:
        """Save splits to files."""
        logger.info("=" * 60)
        logger.info("SAVING SPLITS")
        logger.info("=" * 60)
        
        # Create label map
        label_map = {sign: idx for idx, sign in enumerate(sorted(self.available_signs))}
        
        # Save files
        saved_files = {}
        
        # Train split
        train_path = self.splits_dir / "train_v2.txt"
        with open(train_path, 'w', encoding='utf-8') as f:
            for file_path, label in train_files:
                f.write(f"{file_path}\t{label}\n")
        saved_files['train'] = train_path
        logger.info(f"  Saved: {train_path}")
        
        # Val split
        val_path = self.splits_dir / "val_v2.txt"
        with open(val_path, 'w', encoding='utf-8') as f:
            for file_path, label in val_files:
                f.write(f"{file_path}\t{label}\n")
        saved_files['val'] = val_path
        logger.info(f"  Saved: {val_path}")
        
        # Test split
        test_path = self.splits_dir / "test_v2.txt"
        with open(test_path, 'w', encoding='utf-8') as f:
            for file_path, label in test_files:
                f.write(f"{file_path}\t{label}\n")
        saved_files['test'] = test_path
        logger.info(f"  Saved: {test_path}")
        
        # Label map
        label_map_path = self.splits_dir / "label_map_v2.json"
        with open(label_map_path, 'w', encoding='utf-8') as f:
            json.dump({
                'label_to_idx': label_map,
                'idx_to_label': {v: k for k, v in label_map.items()},
                'num_classes': len(label_map),
                'labels': sorted(self.available_signs)
            }, f, indent=2, ensure_ascii=False)
        saved_files['label_map'] = label_map_path
        logger.info(f"  Saved: {label_map_path}")
        
        # Dataset info
        info_path = self.splits_dir / "dataset_info_v2.json"
        dataset_summary = {
            'version': '2.0.0',
            'created_at': datetime.now().isoformat(),
            'num_classes': len(self.available_signs),
            'classes': sorted(self.available_signs),
            'excluded_classes': self.excluded_signs,
            'train_samples': len(train_files),
            'val_samples': len(val_files),
            'test_samples': len(test_files),
            'total_samples': len(train_files) + len(val_files) + len(test_files),
            'split_ratios': {
                'train': self.train_ratio,
                'val': self.val_ratio,
                'test': self.test_ratio
            },
            'samples_per_class': self.dataset_info.get('samples_per_sign', {}),
            'imbalance_ratio': self.dataset_info.get('imbalance_ratio', 1.0),
            'random_seed': self.random_seed
        }
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_summary, f, indent=2, ensure_ascii=False)
        saved_files['info'] = info_path
        logger.info(f"  Saved: {info_path}")
        
        return saved_files
    
    def generate_report(self) -> str:
        """Generate a human-readable report."""
        report = []
        report.append("=" * 70)
        report.append("PSL DATASET V2 - PREPARATION REPORT")
        report.append("=" * 70)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Features Directory: {self.features_dir}")
        report.append(f"Splits Directory: {self.splits_dir}")
        
        report.append("\n" + "-" * 70)
        report.append("DATASET SUMMARY")
        report.append("-" * 70)
        
        if self.dataset_info:
            report.append(f"Total signs found: {self.dataset_info.get('total_signs', 0)}")
            report.append(f"Included signs: {self.dataset_info.get('included_signs', 0)}")
            report.append(f"Excluded signs: {self.dataset_info.get('excluded_signs', 0)}")
            report.append(f"Total samples: {self.dataset_info.get('total_samples', 0)}")
            
            if self.dataset_info.get('imbalance_ratio'):
                report.append(f"Class imbalance ratio: {self.dataset_info['imbalance_ratio']:.2f}x")
        
        report.append("\n" + "-" * 70)
        report.append("INCLUDED SIGNS")
        report.append("-" * 70)
        for sign in sorted(self.available_signs):
            samples = self.dataset_info.get('samples_per_sign', {}).get(sign, 0)
            report.append(f"  {sign}: {samples} samples")
        
        if self.excluded_signs:
            report.append("\n" + "-" * 70)
            report.append("EXCLUDED SIGNS")
            report.append("-" * 70)
            for sign in self.excluded_signs:
                reason = self.dataset_info.get('excluded_reasons', {}).get(sign, 'Unknown')
                report.append(f"  {sign}: {reason}")
        
        if self.dataset_info.get('issues'):
            report.append("\n" + "-" * 70)
            report.append("ISSUES FOUND")
            report.append("-" * 70)
            for issue in self.dataset_info['issues'][:20]:  # Limit to 20
                report.append(f"  - {issue}")
        
        report.append("\n" + "=" * 70)
        
        return '\n'.join(report)
    
    def run(self) -> Dict[str, Any]:
        """Run the full preparation pipeline."""
        logger.info("Starting dataset preparation V2...")
        
        # 1. Scan dataset
        stats = self.scan_dataset()
        
        if not self.available_signs:
            logger.error("No valid signs found in dataset!")
            return {'success': False, 'error': 'No valid signs found'}
        
        # 2. Create splits
        train_files, val_files, test_files = self.create_splits()
        
        # 3. Save splits
        saved_files = self.save_splits(train_files, val_files, test_files)
        
        # 4. Generate report
        report = self.generate_report()
        report_path = self.splits_dir / "preparation_report_v2.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("\n" + report)
        
        logger.info("=" * 60)
        logger.info("PREPARATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Train: {len(train_files)} samples")
        logger.info(f"Val: {len(val_files)} samples")
        logger.info(f"Test: {len(test_files)} samples")
        logger.info(f"Classes: {len(self.available_signs)}")
        logger.info(f"Saved to: {self.splits_dir}")
        
        return {
            'success': True,
            'num_classes': len(self.available_signs),
            'classes': self.available_signs,
            'train_samples': len(train_files),
            'val_samples': len(val_files),
            'test_samples': len(test_files),
            'saved_files': saved_files
        }


def main():
    """Main entry point."""
    preparer = DatasetPreparerV2(
        features_dir=FEATURES_DIR,
        splits_dir=V2_SPLITS_DIR,
        train_ratio=training_config_v2.TRAIN_SPLIT,
        val_ratio=training_config_v2.VAL_SPLIT,
        test_ratio=training_config_v2.TEST_SPLIT,
        min_samples_per_class=dataset_config_v2.MIN_SAMPLES_PER_CLASS,
        random_seed=42
    )
    
    result = preparer.run()
    
    if result['success']:
        print("\n✓ Dataset preparation completed successfully!")
        print(f"  - Classes: {result['num_classes']}")
        print(f"  - Train: {result['train_samples']}")
        print(f"  - Val: {result['val_samples']}")
        print(f"  - Test: {result['test_samples']}")
        print(f"\nSplits saved to: {V2_SPLITS_DIR}")
        print("\nNext step: Run train_pipeline_v2.py to start training")
    else:
        print(f"\n✗ Dataset preparation failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()

