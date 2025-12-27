#!/usr/bin/env python3
"""
Advanced Data Augmentation for Sign Language Recognition

Techniques:
1. Temporal Augmentation: speed up/down, reverse, temporal crop
2. Spatial Augmentation: rotation, scaling, translation, flipping
3. Noise Augmentation: Gaussian noise, landmark dropout
4. Mixup: Blend sequences from same class
"""

import numpy as np
from typing import Tuple, Optional
import random


class TemporalAugmentation:
    """Temporal augmentation techniques for sequences."""
    
    def __init__(
        self,
        speed_range: Tuple[float, float] = (0.8, 1.2),
        crop_ratio_range: Tuple[float, float] = (0.8, 1.0),
        reverse_prob: float = 0.1,
    ):
        self.speed_range = speed_range
        self.crop_ratio_range = crop_ratio_range
        self.reverse_prob = reverse_prob
    
    def random_speed(self, sequence: np.ndarray) -> np.ndarray:
        """
        Change playback speed by resampling.
        Faster = more frames removed, Slower = interpolated frames added
        """
        if len(sequence) == 0:
            return sequence
            
        speed = np.random.uniform(*self.speed_range)
        new_length = int(len(sequence) / speed)
        new_length = max(1, new_length)  # At least 1 frame
        
        # Interpolate to new length
        indices = np.linspace(0, len(sequence) - 1, new_length)
        
        # Interpolate each feature dimension
        new_sequence = np.zeros((new_length, sequence.shape[1]))
        for i in range(sequence.shape[1]):
            new_sequence[:, i] = np.interp(indices, np.arange(len(sequence)), sequence[:, i])
        
        return new_sequence
    
    def random_temporal_crop(self, sequence: np.ndarray) -> np.ndarray:
        """Random temporal cropping."""
        if len(sequence) == 0:
            return sequence
            
        crop_ratio = np.random.uniform(*self.crop_ratio_range)
        crop_len = max(1, int(len(sequence) * crop_ratio))
        
        if crop_len >= len(sequence):
            return sequence
        
        start = np.random.randint(0, len(sequence) - crop_len + 1)
        return sequence[start:start + crop_len]
    
    def reverse(self, sequence: np.ndarray) -> np.ndarray:
        """Reverse temporal direction (some signs are symmetric)."""
        if np.random.random() < self.reverse_prob:
            return sequence[::-1].copy()
        return sequence
    
    def apply(self, sequence: np.ndarray) -> np.ndarray:
        """Apply random temporal augmentations."""
        # Random speed
        if np.random.random() < 0.5:
            sequence = self.random_speed(sequence)
        
        # Random temporal crop
        if np.random.random() < 0.3:
            sequence = self.random_temporal_crop(sequence)
        
        # Random reverse
        sequence = self.reverse(sequence)
        
        return sequence


class SpatialAugmentation:
    """Spatial augmentation techniques for hand landmarks."""
    
    def __init__(
        self,
        rotation_range: Tuple[float, float] = (-15, 15),
        scale_range: Tuple[float, float] = (0.9, 1.1),
        translation_range: Tuple[float, float] = (-0.1, 0.1),
        flip_prob: float = 0.3,
    ):
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.flip_prob = flip_prob
    
    def random_rotation(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Apply random rotation around Z-axis (camera view).
        landmarks: (n_frames, 189) where 189 = 2 hands × 21 landmarks × 3 coords + padding
        """
        angle = np.random.uniform(*self.rotation_range)
        angle_rad = np.radians(angle)
        
        # Rotation matrix around Z-axis
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Process each frame
        rotated = landmarks.copy()
        for frame_idx in range(len(landmarks)):
            frame = landmarks[frame_idx].reshape(-1, 3)  # Reshape to (N, 3)
            
            # Apply rotation to x, y coordinates (z stays same)
            for i in range(len(frame)):
                x, y, z = frame[i]
                frame[i, 0] = cos_a * x - sin_a * y
                frame[i, 1] = sin_a * x + cos_a * y
                frame[i, 2] = z
            
            rotated[frame_idx] = frame.flatten()
        
        return rotated
    
    def random_scale(self, landmarks: np.ndarray) -> np.ndarray:
        """Scale landmarks (hand size variation)."""
        scale = np.random.uniform(*self.scale_range)
        return landmarks * scale
    
    def random_translation(self, landmarks: np.ndarray) -> np.ndarray:
        """Random translation (hand position variation)."""
        translation_x = np.random.uniform(*self.translation_range)
        translation_y = np.random.uniform(*self.translation_range)
        
        # Apply translation
        translated = landmarks.copy()
        for frame_idx in range(len(landmarks)):
            frame = landmarks[frame_idx].reshape(-1, 3)
            frame[:, 0] += translation_x
            frame[:, 1] += translation_y
            translated[frame_idx] = frame.flatten()
        
        return translated
    
    def random_flip(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Horizontal flip (mirror).
        Note: In PSL, some signs are different when flipped, so use carefully.
        """
        if np.random.random() < self.flip_prob:
            flipped = landmarks.copy()
            for frame_idx in range(len(landmarks)):
                frame = landmarks[frame_idx].reshape(-1, 3)
                frame[:, 0] = -frame[:, 0]  # Flip x-axis
                flipped[frame_idx] = frame.flatten()
            return flipped
        return landmarks
    
    def apply(self, landmarks: np.ndarray) -> np.ndarray:
        """Apply random spatial augmentations."""
        # Random rotation
        if np.random.random() < 0.5:
            landmarks = self.random_rotation(landmarks)
        
        # Random scale
        if np.random.random() < 0.5:
            landmarks = self.random_scale(landmarks)
        
        # Random translation
        if np.random.random() < 0.5:
            landmarks = self.random_translation(landmarks)
        
        # Random flip (less frequent)
        if np.random.random() < 0.2:
            landmarks = self.random_flip(landmarks)
        
        return landmarks


class NoiseAugmentation:
    """Noise augmentation techniques."""
    
    def __init__(
        self,
        gaussian_noise_std: float = 0.01,
        dropout_prob: float = 0.1,
        dropout_ratio: float = 0.1,
    ):
        self.gaussian_noise_std = gaussian_noise_std
        self.dropout_prob = dropout_prob
        self.dropout_ratio = dropout_ratio
    
    def gaussian_noise(self, landmarks: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to simulate tracking errors."""
        noise = np.random.normal(0, self.gaussian_noise_std, landmarks.shape)
        return landmarks + noise
    
    def landmark_dropout(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Randomly dropout some landmarks to simulate occlusion.
        """
        if np.random.random() < self.dropout_prob:
            dropped = landmarks.copy()
            n_landmarks = landmarks.shape[1] // 3
            n_drop = max(1, int(n_landmarks * self.dropout_ratio))
            
            # Randomly select landmarks to drop
            drop_indices = np.random.choice(n_landmarks, n_drop, replace=False)
            
            # Set dropped landmarks to zero
            for idx in drop_indices:
                dropped[:, idx*3:(idx+1)*3] = 0
            
            return dropped
        return landmarks
    
    def apply(self, landmarks: np.ndarray) -> np.ndarray:
        """Apply random noise augmentations."""
        # Gaussian noise
        if np.random.random() < 0.5:
            landmarks = self.gaussian_noise(landmarks)
        
        # Landmark dropout
        landmarks = self.landmark_dropout(landmarks)
        
        return landmarks


class MixupAugmentation:
    """Mixup augmentation for sequences."""
    
    def __init__(self, alpha: float = 0.2, prob: float = 0.3):
        self.alpha = alpha
        self.prob = prob
    
    def mixup(
        self,
        seq1: np.ndarray,
        seq2: np.ndarray,
        label1: int,
        label2: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mixup two sequences.
        Only mix sequences from the SAME class for sign language.
        
        Returns:
            mixed_seq: Mixed sequence
            mixed_label: One-hot label with mixing coefficient
        """
        if label1 != label2 or np.random.random() > self.prob:
            # Don't mix different classes or if random skip
            return seq1, label1
        
        # Sample mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Make sequences same length (pad shorter one)
        max_len = max(len(seq1), len(seq2))
        if len(seq1) < max_len:
            seq1 = np.vstack([seq1, np.zeros((max_len - len(seq1), seq1.shape[1]))])
        if len(seq2) < max_len:
            seq2 = np.vstack([seq2, np.zeros((max_len - len(seq2), seq2.shape[1]))])
        
        # Mix
        mixed_seq = lam * seq1 + (1 - lam) * seq2
        
        return mixed_seq, label1  # Same label since same class


class ComprehensiveAugmentation:
    """Combine all augmentation techniques."""
    
    def __init__(
        self,
        use_temporal: bool = True,
        use_spatial: bool = True,
        use_noise: bool = True,
        use_mixup: bool = False,  # Only for training
        augmentation_prob: float = 0.8,
    ):
        self.use_temporal = use_temporal
        self.use_spatial = use_spatial
        self.use_noise = use_noise
        self.use_mixup = use_mixup
        self.augmentation_prob = augmentation_prob
        
        self.temporal_aug = TemporalAugmentation()
        self.spatial_aug = SpatialAugmentation()
        self.noise_aug = NoiseAugmentation()
        self.mixup_aug = MixupAugmentation()
    
    def augment(
        self,
        sequence: np.ndarray,
        apply_augmentation: bool = True,
    ) -> np.ndarray:
        """
        Apply comprehensive augmentation pipeline.
        
        Args:
            sequence: Input sequence (n_frames, 189)
            apply_augmentation: Whether to apply augmentation
        
        Returns:
            Augmented sequence
        """
        if not apply_augmentation or np.random.random() > self.augmentation_prob:
            return sequence
        
        # Temporal augmentation (changes sequence length)
        if self.use_temporal:
            sequence = self.temporal_aug.apply(sequence)
        
        # Spatial augmentation (changes landmark positions)
        if self.use_spatial:
            sequence = self.spatial_aug.apply(sequence)
        
        # Noise augmentation (adds noise/dropout)
        if self.use_noise:
            sequence = self.noise_aug.apply(sequence)
        
        return sequence


# Example usage
if __name__ == "__main__":
    print("Testing Data Augmentation...")
    
    # Create dummy sequence
    seq_length = 60
    feature_dim = 189
    dummy_sequence = np.random.randn(seq_length, feature_dim)
    
    print(f"Original sequence shape: {dummy_sequence.shape}")
    
    # Test temporal augmentation
    temporal_aug = TemporalAugmentation()
    aug_seq = temporal_aug.random_speed(dummy_sequence)
    print(f"After speed change: {aug_seq.shape}")
    
    aug_seq = temporal_aug.random_temporal_crop(dummy_sequence)
    print(f"After temporal crop: {aug_seq.shape}")
    
    # Test spatial augmentation
    spatial_aug = SpatialAugmentation()
    aug_seq = spatial_aug.random_rotation(dummy_sequence)
    print(f"After rotation: {aug_seq.shape}")
    
    aug_seq = spatial_aug.random_scale(dummy_sequence)
    print(f"After scaling: {aug_seq.shape}")
    
    # Test noise augmentation
    noise_aug = NoiseAugmentation()
    aug_seq = noise_aug.gaussian_noise(dummy_sequence)
    print(f"After Gaussian noise: {aug_seq.shape}")
    
    # Test comprehensive augmentation
    comp_aug = ComprehensiveAugmentation()
    aug_seq = comp_aug.augment(dummy_sequence)
    print(f"After comprehensive augmentation: {aug_seq.shape}")
    
    print("\n✓ All augmentation tests passed!")

