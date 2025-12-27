#!/usr/bin/env python3
"""
Dataset V2 for PSL Recognition System.
PyTorch Dataset class with augmentation support for training on 40 sign classes.
"""

import os
import sys
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_v2 import (
    V2_SPLITS_DIR, FEATURES_DIR, training_config_v2, model_config_v2
)

logger = logging.getLogger(__name__)


class DataAugmentation:
    """Data augmentation transforms for hand landmark sequences."""
    
    def __init__(
        self,
        temporal_jitter: float = 0.1,
        noise_std: float = 0.01,
        rotation_range: float = 5.0,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        prob: float = 0.5
    ):
        """
        Initialize augmentation transforms.
        
        Args:
            temporal_jitter: Max ratio of temporal shift (0.1 = 10% of sequence)
            noise_std: Standard deviation of Gaussian noise
            rotation_range: Max rotation angle in degrees
            scale_range: Min and max scale factors
            prob: Probability of applying each augmentation
        """
        self.temporal_jitter = temporal_jitter
        self.noise_std = noise_std
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.prob = prob
    
    def __call__(self, sequence: np.ndarray) -> np.ndarray:
        """Apply random augmentations to a sequence."""
        sequence = sequence.copy()
        
        # Temporal jitter (shift frames)
        if random.random() < self.prob:
            sequence = self._temporal_jitter(sequence)
        
        # Add Gaussian noise
        if random.random() < self.prob:
            sequence = self._add_noise(sequence)
        
        # Scale landmarks
        if random.random() < self.prob:
            sequence = self._scale(sequence)
        
        return sequence
    
    def _temporal_jitter(self, sequence: np.ndarray) -> np.ndarray:
        """Randomly shift sequence temporally."""
        seq_len = len(sequence)
        max_shift = int(seq_len * self.temporal_jitter)
        if max_shift == 0:
            return sequence
        
        shift = random.randint(-max_shift, max_shift)
        if shift > 0:
            # Shift right (pad at beginning)
            return np.vstack([
                np.tile(sequence[0], (shift, 1)),
                sequence[:-shift]
            ])
        elif shift < 0:
            # Shift left (pad at end)
            return np.vstack([
                sequence[-shift:],
                np.tile(sequence[-1], (-shift, 1))
            ])
        return sequence
    
    def _add_noise(self, sequence: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to landmarks."""
        noise = np.random.normal(0, self.noise_std, sequence.shape)
        return sequence + noise
    
    def _scale(self, sequence: np.ndarray) -> np.ndarray:
        """Scale landmarks (simulates hand distance variation)."""
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        # Only scale coordinate values, not padding
        return sequence * scale


class PSLDatasetV2(Dataset):
    """
    PyTorch Dataset for PSL Recognition V2.
    
    Features:
    - Loads pre-extracted .npy features
    - Supports train/val/test splits
    - Optional data augmentation
    - Handles variable-length sequences via padding/truncation
    """
    
    def __init__(
        self,
        split: str = 'train',
        splits_dir: Path = V2_SPLITS_DIR,
        sequence_length: int = 60,
        augment: bool = False,
        augmentation_config: Optional[Dict] = None
    ):
        """
        Initialize dataset.
        
        Args:
            split: One of 'train', 'val', 'test'
            splits_dir: Directory containing split files
            sequence_length: Target sequence length (pad/truncate to this)
            augment: Whether to apply augmentation (only for train)
            augmentation_config: Custom augmentation parameters
        """
        self.split = split
        self.splits_dir = Path(splits_dir)
        self.sequence_length = sequence_length
        self.augment = augment and split == 'train'  # Only augment training
        
        # Load split file
        split_file = self.splits_dir / f"{split}_v2.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        # Load label map
        label_map_file = self.splits_dir / "label_map_v2.json"
        if not label_map_file.exists():
            raise FileNotFoundError(f"Label map not found: {label_map_file}")
        
        with open(label_map_file, 'r', encoding='utf-8') as f:
            self.label_data = json.load(f)
        
        self.label_to_idx = self.label_data['label_to_idx']
        self.idx_to_label = {int(k): v for k, v in self.label_data['idx_to_label'].items()}
        self.num_classes = self.label_data['num_classes']
        self.labels = self.label_data['labels']
        
        # Load samples
        self.samples = []
        self.features_dir = FEATURES_DIR  # Store for path resolution
        
        with open(split_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        file_path, label = parts
                        if label in self.label_to_idx:
                            self.samples.append({
                                'file': file_path,
                                'label': label,
                                'label_idx': self.label_to_idx[label]
                            })
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
        
        # Initialize augmentation
        if self.augment:
            aug_config = augmentation_config or {}
            self.augmentation = DataAugmentation(
                temporal_jitter=aug_config.get('temporal_jitter', training_config_v2.TEMPORAL_JITTER),
                noise_std=aug_config.get('noise_std', training_config_v2.NOISE_STD),
                rotation_range=aug_config.get('rotation_range', training_config_v2.ROTATION_RANGE),
                scale_range=aug_config.get('scale_range', training_config_v2.SCALE_RANGE),
                prob=aug_config.get('prob', training_config_v2.AUGMENTATION_PROB)
            )
        else:
            self.augmentation = None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample."""
        sample = self.samples[idx]
        
        # Resolve file path (handle both absolute and relative paths)
        file_path_str = sample['file']
        file_path = Path(file_path_str)
        
        # Normalize path separators (handle both / and \)
        file_path_str = file_path_str.replace('\\', '/')
        
        # If relative path, resolve relative to features directory
        if not Path(file_path_str).is_absolute():
            # Remove "data/features_temporal/" prefix if present
            if file_path_str.startswith('data/features_temporal/'):
                rel_part = file_path_str.replace('data/features_temporal/', '')
                resolved_path = self.features_dir / rel_part
            elif file_path_str.startswith('features_temporal/'):
                rel_part = file_path_str.replace('features_temporal/', '')
                resolved_path = self.features_dir / rel_part
            else:
                # Direct relative path
                resolved_path = self.features_dir / file_path_str
        else:
            # Absolute path - use as-is
            resolved_path = Path(file_path_str)
        
        # Load feature file
        try:
            if not resolved_path.exists():
                # Try alternative paths for Colab compatibility
                alt_paths = [
                    Path(file_path_str),  # Original path
                    Path.cwd() / file_path_str if not Path(file_path_str).is_absolute() else None,
                    Path('/content/ISHARAPAL/backend') / file_path_str if file_path_str.startswith('data/') else None,
                ]
                
                for alt_path in alt_paths:
                    if alt_path and alt_path.exists():
                        resolved_path = alt_path
                        break
                else:
                    # File doesn't exist - return zeros (will be skipped in training)
                    logger.warning(f"File not found: {resolved_path} (original: {sample['file']})")
                    features = np.zeros((self.sequence_length, 189), dtype=np.float32)
                    return torch.FloatTensor(features), sample['label_idx']
            
            features = np.load(resolved_path)
        except Exception as e:
            logger.error(f"Error loading {sample['file']} (resolved: {resolved_path}): {e}")
            # Return zeros as fallback (will be skipped if all zeros)
            features = np.zeros((self.sequence_length, 189), dtype=np.float32)
        
        # Ensure 2D shape (seq_len, features)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Pad or truncate to target sequence length
        features = self._normalize_sequence_length(features)
        
        # Apply augmentation
        if self.augmentation is not None:
            features = self.augmentation(features)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features)
        label_idx = sample['label_idx']
        
        return features_tensor, label_idx
    
    def _normalize_sequence_length(self, sequence: np.ndarray) -> np.ndarray:
        """Pad or truncate sequence to target length."""
        current_len = len(sequence)
        target_len = self.sequence_length
        
        if current_len == target_len:
            return sequence
        
        elif current_len < target_len:
            # Pad with zeros at the beginning
            padding = np.zeros((target_len - current_len, sequence.shape[1]), dtype=sequence.dtype)
            return np.vstack([padding, sequence])
        
        else:
            # Truncate from the beginning (keep most recent frames)
            return sequence[-target_len:]
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for handling imbalanced data."""
        class_counts = np.zeros(self.num_classes)
        
        for sample in self.samples:
            class_counts[sample['label_idx']] += 1
        
        # Avoid division by zero
        class_counts = np.maximum(class_counts, 1)
        
        # Inverse frequency weighting
        weights = 1.0 / class_counts
        weights = weights / weights.sum() * self.num_classes  # Normalize
        
        return torch.FloatTensor(weights)
    
    def get_sample_weights(self) -> List[float]:
        """Get sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights().numpy()
        sample_weights = [class_weights[s['label_idx']] for s in self.samples]
        return sample_weights
    
    def get_label_name(self, idx: int) -> str:
        """Get label name from index."""
        return self.idx_to_label.get(idx, f"Unknown_{idx}")


def create_dataloaders(
    batch_size: int = 32,
    sequence_length: int = 60,
    num_workers: int = 0,
    use_weighted_sampler: bool = True,
    augment_train: bool = True,
    splits_dir: Path = V2_SPLITS_DIR
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        batch_size: Batch size for training
        sequence_length: Target sequence length
        num_workers: Number of data loading workers (0 for Windows)
        use_weighted_sampler: Use weighted sampler for class imbalance
        augment_train: Apply augmentation to training data
        splits_dir: Directory containing split files
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, info_dict)
    """
    # Create datasets
    train_dataset = PSLDatasetV2(
        split='train',
        splits_dir=splits_dir,
        sequence_length=sequence_length,
        augment=augment_train
    )
    
    val_dataset = PSLDatasetV2(
        split='val',
        splits_dir=splits_dir,
        sequence_length=sequence_length,
        augment=False
    )
    
    test_dataset = PSLDatasetV2(
        split='test',
        splits_dir=splits_dir,
        sequence_length=sequence_length,
        augment=False
    )
    
    # Create samplers
    if use_weighted_sampler:
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        train_shuffle = False  # Sampler handles shuffling
    else:
        sampler = None
        train_shuffle = True
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Info dict
    info = {
        'num_classes': train_dataset.num_classes,
        'labels': train_dataset.labels,
        'label_to_idx': train_dataset.label_to_idx,
        'idx_to_label': train_dataset.idx_to_label,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset),
        'class_weights': train_dataset.get_class_weights(),
        'sequence_length': sequence_length,
        'batch_size': batch_size
    }
    
    logger.info(f"Created dataloaders:")
    logger.info(f"  Train: {info['train_samples']} samples, {len(train_loader)} batches")
    logger.info(f"  Val: {info['val_samples']} samples, {len(val_loader)} batches")
    logger.info(f"  Test: {info['test_samples']} samples, {len(test_loader)} batches")
    logger.info(f"  Classes: {info['num_classes']}")
    
    return train_loader, val_loader, test_loader, info


# Test the dataset
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Testing PSL Dataset V2")
    print("=" * 60)
    
    # Create dataloaders
    train_loader, val_loader, test_loader, info = create_dataloaders(
        batch_size=8,
        sequence_length=60,
        use_weighted_sampler=True,
        augment_train=True
    )
    
    print(f"\nDataset Info:")
    print(f"  Classes: {info['num_classes']}")
    print(f"  Train: {info['train_samples']} samples")
    print(f"  Val: {info['val_samples']} samples")
    print(f"  Test: {info['test_samples']} samples")
    
    print(f"\nClass weights (for imbalance):")
    weights = info['class_weights']
    for i, (label, weight) in enumerate(zip(info['labels'][:5], weights[:5])):
        print(f"  {label}: {weight:.4f}")
    print(f"  ... ({len(info['labels']) - 5} more)")
    
    # Test a batch
    print(f"\nTesting batch loading...")
    for batch_idx, (features, labels) in enumerate(train_loader):
        print(f"  Batch {batch_idx + 1}:")
        print(f"    Features shape: {features.shape}")
        print(f"    Labels: {labels[:5].tolist()}...")
        
        # Verify shapes
        assert features.shape == (8, 60, 189), f"Unexpected shape: {features.shape}"
        break
    
    print("\n[OK] Dataset V2 is working correctly!")

