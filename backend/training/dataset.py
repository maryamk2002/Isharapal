#!/usr/bin/env python3
"""
Dataset class for PSL temporal sequences.
Handles loading, preprocessing, and augmentation of sign language data.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset

from training.data_augmentation import ComprehensiveAugmentation


class PSLTemporalDataset(Dataset):
    """
    Dataset for PSL temporal sequences.
    
    Features:
    - Load temporal sequences from disk
    - Apply data augmentation
    - Handle variable-length sequences
    - Support for train/val/test splits
    """
    
    def __init__(
        self,
        data_dir: Path,
        labels: List[str],
        split: str = 'train',
        target_sequence_length: int = 60,
        augmentation: bool = True,
        augmentation_config: Optional[Dict] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing temporal sequence files
            labels: List of class labels (e.g., ['2-Hay', 'Alifmad', 'Aray', 'Jeem'])
            split: 'train', 'val', or 'test'
            target_sequence_length: Target length for sequences (pad/truncate)
            augmentation: Whether to apply data augmentation
            augmentation_config: Configuration for augmentation
        """
        self.data_dir = Path(data_dir)
        self.labels = labels
        self.split = split
        self.target_sequence_length = target_sequence_length
        self.augmentation = augmentation and split == 'train'
        
        # Label to index mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # Initialize augmentation
        if self.augmentation:
            aug_config = augmentation_config or {}
            self.augmenter = ComprehensiveAugmentation(**aug_config)
        else:
            self.augmenter = None
        
        # Load data
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
        print(f"Classes: {labels}")
        print(f"Augmentation: {self.augmentation}")
    
    def _load_samples(self) -> List[Tuple[Path, int]]:
        """
        Load all samples from data directory.
        
        Returns:
            List of (file_path, label_idx) tuples
        """
        samples = []
        
        for label in self.labels:
            label_idx = self.label_to_idx[label]
            label_dir = self.data_dir / label
            
            if not label_dir.exists():
                print(f"Warning: Directory not found: {label_dir}")
                continue
            
            # Find all .npy files (temporal sequences)
            npy_files = list(label_dir.glob('*.npy'))
            
            # If no .npy files, try to find in subdirectories
            if len(npy_files) == 0:
                npy_files = list(label_dir.rglob('*.npy'))
            
            print(f"  Found {len(npy_files)} sequences for {label}")
            
            for npy_file in npy_files:
                samples.append((npy_file, label_idx))
        
        if len(samples) == 0:
            raise ValueError(f"No samples found in {self.data_dir}")
        
        return samples
    
    def _pad_or_truncate(self, sequence: np.ndarray) -> np.ndarray:
        """
        Pad or truncate sequence to target length.
        
        Args:
            sequence: Input sequence (variable_length, feature_dim)
        
        Returns:
            Fixed-length sequence (target_length, feature_dim)
        """
        current_length = len(sequence)
        feature_dim = sequence.shape[1]
        
        if current_length == self.target_sequence_length:
            return sequence
        
        elif current_length < self.target_sequence_length:
            # Pad with zeros
            padding_length = self.target_sequence_length - current_length
            padding = np.zeros((padding_length, feature_dim), dtype=sequence.dtype)
            return np.vstack([sequence, padding])
        
        else:
            # Truncate (take middle part to preserve start and end)
            start = (current_length - self.target_sequence_length) // 2
            return sequence[start:start + self.target_sequence_length]
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple of (sequence_tensor, label_tensor)
        """
        file_path, label_idx = self.samples[idx]
        
        # Load sequence from disk
        try:
            sequence = np.load(file_path).astype(np.float32)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return dummy data
            sequence = np.zeros((self.target_sequence_length, 189), dtype=np.float32)
        
        # Ensure 2D
        if sequence.ndim == 1:
            sequence = sequence.reshape(1, -1)
        
        # Apply augmentation (only during training)
        if self.augmentation and self.augmenter is not None:
            sequence = self.augmenter.augment(sequence, apply_augmentation=True)
        
        # Pad or truncate to target length
        sequence = self._pad_or_truncate(sequence)
        
        # Convert to tensors
        sequence_tensor = torch.from_numpy(sequence).float()
        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        
        return sequence_tensor, label_tensor
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced datasets.
        
        Returns:
            Tensor of class weights
        """
        class_counts = [0] * len(self.labels)
        
        for _, label_idx in self.samples:
            class_counts[label_idx] += 1
        
        # Calculate weights (inverse frequency)
        total_samples = len(self.samples)
        class_weights = [total_samples / (count * len(self.labels)) for count in class_counts]
        
        return torch.tensor(class_weights, dtype=torch.float32)


class BalancedBatchSampler:
    """
    Sampler that ensures each batch has balanced classes.
    Useful for imbalanced datasets.
    """
    
    def __init__(
        self,
        dataset: PSLTemporalDataset,
        batch_size: int,
        samples_per_class: Optional[int] = None,
    ):
        """
        Initialize balanced batch sampler.
        
        Args:
            dataset: PSL dataset
            batch_size: Batch size
            samples_per_class: Number of samples per class in each batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_classes = len(dataset.labels)
        
        if samples_per_class is None:
            samples_per_class = batch_size // self.num_classes
        self.samples_per_class = samples_per_class
        
        # Group samples by class
        self.class_indices = [[] for _ in range(self.num_classes)]
        for idx, (_, label_idx) in enumerate(dataset.samples):
            self.class_indices[label_idx].append(idx)
        
        # Calculate number of batches
        min_samples = min(len(indices) for indices in self.class_indices)
        self.num_batches = min_samples // self.samples_per_class
    
    def __iter__(self):
        """Iterate over balanced batches."""
        for _ in range(self.num_batches):
            batch_indices = []
            
            # Sample from each class
            for class_idx in range(self.num_classes):
                indices = np.random.choice(
                    self.class_indices[class_idx],
                    self.samples_per_class,
                    replace=False,
                )
                batch_indices.extend(indices)
            
            # Shuffle within batch
            np.random.shuffle(batch_indices)
            
            yield batch_indices
    
    def __len__(self):
        """Return number of batches."""
        return self.num_batches


def create_dataloaders(
    data_dir: Path,
    labels: List[str],
    batch_size: int = 32,
    target_sequence_length: int = 60,
    num_workers: int = 4,
    use_balanced_sampler: bool = False,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train, val, and test dataloaders.
    
    Args:
        data_dir: Base data directory
        labels: List of class labels
        batch_size: Batch size
        target_sequence_length: Target sequence length
        num_workers: Number of worker processes for data loading
        use_balanced_sampler: Whether to use balanced batch sampler
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = PSLTemporalDataset(
        data_dir=data_dir / 'train',
        labels=labels,
        split='train',
        target_sequence_length=target_sequence_length,
        augmentation=True,
    )
    
    val_dataset = PSLTemporalDataset(
        data_dir=data_dir / 'val',
        labels=labels,
        split='val',
        target_sequence_length=target_sequence_length,
        augmentation=False,
    )
    
    test_dataset = PSLTemporalDataset(
        data_dir=data_dir / 'test',
        labels=labels,
        split='test',
        target_sequence_length=target_sequence_length,
        augmentation=False,
    )
    
    # Create dataloaders
    if use_balanced_sampler:
        sampler = BalancedBatchSampler(train_dataset, batch_size)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


# Example usage
if __name__ == "__main__":
    print("Testing PSL Dataset...")
    
    # Example configuration
    data_dir = Path("backend/data/features_temporal")
    labels = ['2-Hay', 'Alifmad', 'Aray', 'Jeem']
    
    try:
        # Create dataset
        dataset = PSLTemporalDataset(
            data_dir=data_dir,
            labels=labels,
            split='train',
            target_sequence_length=60,
            augmentation=True,
        )
        
        print(f"\nDataset size: {len(dataset)}")
        
        # Get a sample
        if len(dataset) > 0:
            sequence, label = dataset[0]
            print(f"Sample sequence shape: {sequence.shape}")
            print(f"Sample label: {label} ({dataset.idx_to_label[label.item()]})")
        
        # Test dataloader
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=8,
            shuffle=True,
        )
        
        batch = next(iter(loader))
        sequences, labels = batch
        print(f"\nBatch sequences shape: {sequences.shape}")
        print(f"Batch labels shape: {labels.shape}")
        
        print("\n✓ Dataset test passed!")
        
    except Exception as e:
        print(f"Note: {e}")
        print("(This is expected if data directory doesn't exist yet)")

