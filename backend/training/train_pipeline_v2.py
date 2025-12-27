#!/usr/bin/env python3
"""
Training Pipeline V2 for PSL Recognition System.
Complete training pipeline for all 40 Urdu alphabet signs.

Features:
- Uses proven TCN architecture
- Handles class imbalance with weighted loss
- Learning rate scheduling
- Early stopping
- Checkpoint saving
- Comprehensive logging
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_v2 import (
    V2_MODELS_DIR, V2_CHECKPOINTS_DIR, V2_LOGS_DIR, V2_SPLITS_DIR,
    model_config_v2, training_config_v2, get_hardware_info
)
from training.dataset_v2 import create_dataloaders

# Try to import existing model architecture
try:
    from models.optimized_tcn_model import OptimizedTCNModel
    MODEL_CLASS = OptimizedTCNModel
    MODEL_TYPE = 'optimized_tcn'
except ImportError:
    try:
        from models.tcn_model import EnhancedTCN, create_model
        MODEL_CLASS = EnhancedTCN
        MODEL_TYPE = 'enhanced_tcn'
    except ImportError:
        MODEL_CLASS = None
        MODEL_TYPE = None

# Configure logging
def setup_logging(log_dir: Path, experiment_name: str) -> logging.Logger:
    """Setup logging for training."""
    log_file = log_dir / f"{experiment_name}.log"
    
    # Create handlers
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding='utf-8')
    ]
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        mode: str = 'max'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class TrainerV2:
    """Training manager for PSL Recognition V2."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        dataset_info: Dict[str, Any],
        device: torch.device,
        config: Dict[str, Any],
        experiment_name: str = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.dataset_info = dataset_info
        self.device = device
        self.config = config
        
        # Experiment name
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"psl_v2_{timestamp}"
        self.experiment_name = experiment_name
        
        # Create checkpoint directory
        self.checkpoint_dir = V2_CHECKPOINTS_DIR / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function with class weights
        class_weights = dataset_info['class_weights'].to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', training_config_v2.LEARNING_RATE),
            weight_decay=config.get('weight_decay', training_config_v2.WEIGHT_DECAY)
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            patience=config.get('scheduler_patience', training_config_v2.SCHEDULER_PATIENCE),
            factor=config.get('scheduler_factor', training_config_v2.SCHEDULER_FACTOR)
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('early_stopping_patience', training_config_v2.EARLY_STOPPING_PATIENCE),
            mode='max'
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        for batch_idx, (features, labels) in enumerate(pbar):
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('gradient_clip', training_config_v2.GRADIENT_CLIP_NORM)
            )
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self, loader: torch.utils.data.DataLoader = None) -> Tuple[float, float]:
        """Validate the model."""
        if loader is None:
            loader = self.val_loader
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for features, labels in loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate on test set with detailed metrics."""
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_confidences = []
        
        for features, labels in self.test_loader:
            features = features.to(self.device)
            
            outputs = self.model(features)
            probs = torch.softmax(outputs, dim=1)
            confidences, predicted = probs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_confidences.extend(confidences.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_confidences = np.array(all_confidences)
        
        # Calculate metrics
        accuracy = (all_predictions == all_labels).mean() * 100
        
        # Per-class accuracy
        per_class_acc = {}
        for label_idx, label_name in self.dataset_info['idx_to_label'].items():
            mask = all_labels == label_idx
            if mask.sum() > 0:
                class_acc = (all_predictions[mask] == all_labels[mask]).mean() * 100
                per_class_acc[label_name] = class_acc
        
        # Confusion pairs (most confused)
        confusion_pairs = []
        for i in range(len(all_labels)):
            if all_predictions[i] != all_labels[i]:
                true_label = self.dataset_info['idx_to_label'][all_labels[i]]
                pred_label = self.dataset_info['idx_to_label'][all_predictions[i]]
                confusion_pairs.append((true_label, pred_label))
        
        # Count confusion pairs
        from collections import Counter
        confusion_counts = Counter(confusion_pairs)
        top_confusions = confusion_counts.most_common(10)
        
        return {
            'test_accuracy': accuracy,
            'per_class_accuracy': per_class_acc,
            'top_confusions': top_confusions,
            'avg_confidence': float(np.mean(all_confidences)),
            'total_samples': len(all_labels)
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'config': self.config,
            'dataset_info': {
                'num_classes': self.dataset_info['num_classes'],
                'labels': self.dataset_info['labels'],
                'label_to_idx': self.dataset_info['label_to_idx']
            }
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
        
        # Save periodic checkpoint
        if epoch % 10 == 0:
            epoch_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, epoch_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model (epoch {epoch}, val_acc={self.best_val_acc:.2f}%)")
    
    def save_final_model(self):
        """Save final model for deployment."""
        # Save to v2 models directory
        model_path = V2_MODELS_DIR / 'psl_model_v2.pth'
        config_path = V2_MODELS_DIR / 'psl_model_v2_config.json'
        labels_path = V2_MODELS_DIR / 'psl_model_v2_labels.txt'
        
        # Save model weights
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_info': {
                'model_type': MODEL_TYPE,
                'input_dim': self.config.get('input_dim', 189),
                'num_classes': self.dataset_info['num_classes'],
                'sequence_length': self.config.get('sequence_length', 60)
            },
            'config': self.config,
            'labels': self.dataset_info['labels'],
            'metrics': {
                'best_val_acc': self.best_val_acc,
                'best_epoch': self.best_epoch
            },
            'saved_at': datetime.now().isoformat()
        }, model_path)
        
        # Save config
        config_to_save = {
            'model_type': MODEL_TYPE,
            'target_seq_len': self.config.get('sequence_length', 60),
            'sequence_length': self.config.get('sequence_length', 60),
            'feature_size': self.config.get('input_dim', 189),
            'input_dim': self.config.get('input_dim', 189),
            'num_classes': self.dataset_info['num_classes'],
            'best_val_acc': self.best_val_acc,
            'epochs_trained': len(self.history['train_loss']),
            'batch_size': self.config.get('batch_size', 32),
            'learning_rate': self.config.get('learning_rate', 0.0005),
            'labels': self.dataset_info['labels'],
            'architecture': {
                'num_channels': self.config.get('num_channels', [256, 256, 256, 256, 128]),
                'kernel_size': self.config.get('kernel_size', 5),
                'dropout': self.config.get('dropout', 0.4)
            }
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_to_save, f, indent=2, ensure_ascii=False)
        
        # Save labels
        with open(labels_path, 'w', encoding='utf-8') as f:
            for label in self.dataset_info['labels']:
                f.write(f"{label}\n")
        
        self.logger.info(f"Final model saved to {model_path}")
        
        return model_path, config_path, labels_path
    
    def train(self, epochs: int, start_epoch: int = 1) -> Dict[str, Any]:
        """Run full training loop."""
        self.logger.info("=" * 60)
        if start_epoch > 1:
            self.logger.info("RESUMING TRAINING")
        else:
            self.logger.info("STARTING TRAINING")
        self.logger.info("=" * 60)
        self.logger.info(f"Experiment: {self.experiment_name}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Epochs: {start_epoch} to {epochs} (total: {epochs})")
        self.logger.info(f"Classes: {self.dataset_info['num_classes']}")
        self.logger.info(f"Train samples: {self.dataset_info['train_samples']}")
        self.logger.info(f"Val samples: {self.dataset_info['val_samples']}")
        if start_epoch > 1:
            self.logger.info(f"Resuming from epoch {start_epoch} (previous epochs already completed)")
        self.logger.info("-" * 60)
        
        start_time = time.time()
        
        for epoch in range(start_epoch, epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            epoch_time = time.time() - epoch_start
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            self.history['epoch_time'].append(epoch_time)
            
            # Check for best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Log progress
            self.logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Train: {train_acc:.2f}% (loss={train_loss:.4f}) | "
                f"Val: {val_acc:.2f}% (loss={val_loss:.4f}) | "
                f"LR: {current_lr:.6f} | "
                f"Time: {epoch_time:.1f}s"
                + (" [BEST]" if is_best else "")
            )
            
            # Early stopping check
            if self.early_stopping(val_acc):
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        total_time = time.time() - start_time
        
        # Evaluate on test set
        self.logger.info("-" * 60)
        self.logger.info("EVALUATING ON TEST SET")
        self.logger.info("-" * 60)
        
        # Load best model for evaluation
        best_path = self.checkpoint_dir / 'best_model.pth'
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        eval_results = self.evaluate()
        
        self.logger.info(f"Test Accuracy: {eval_results['test_accuracy']:.2f}%")
        self.logger.info(f"Average Confidence: {eval_results['avg_confidence']:.3f}")
        
        # Log per-class accuracy (top and bottom 5)
        sorted_acc = sorted(eval_results['per_class_accuracy'].items(), key=lambda x: x[1])
        self.logger.info("Lowest performing classes:")
        for label, acc in sorted_acc[:5]:
            self.logger.info(f"  {label}: {acc:.2f}%")
        self.logger.info("Highest performing classes:")
        for label, acc in sorted_acc[-5:]:
            self.logger.info(f"  {label}: {acc:.2f}%")
        
        # Log top confusions
        if eval_results['top_confusions']:
            self.logger.info("Top confusion pairs:")
            for (true_label, pred_label), count in eval_results['top_confusions'][:5]:
                self.logger.info(f"  {true_label} -> {pred_label}: {count} times")
        
        # Save final model
        model_path, config_path, labels_path = self.save_final_model()
        
        # Save training history
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Summary
        self.logger.info("=" * 60)
        self.logger.info("TRAINING COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"Total time: {total_time/60:.1f} minutes")
        self.logger.info(f"Best epoch: {self.best_epoch}")
        self.logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        self.logger.info(f"Test accuracy: {eval_results['test_accuracy']:.2f}%")
        self.logger.info(f"Model saved to: {model_path}")
        
        return {
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'test_accuracy': eval_results['test_accuracy'],
            'eval_results': eval_results,
            'history': self.history,
            'total_time': total_time,
            'model_path': str(model_path)
        }


def create_model_v2(
    input_dim: int = 189,
    num_classes: int = 40,
    num_channels: List[int] = None,
    kernel_size: int = 5,
    dropout: float = 0.4
) -> nn.Module:
    """Create the model for training."""
    if num_channels is None:
        num_channels = [256, 256, 256, 256, 128]
    
    if MODEL_CLASS is not None and MODEL_TYPE == 'optimized_tcn':
        model = MODEL_CLASS(
            input_dim=input_dim,
            num_classes=num_classes,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
    elif MODEL_CLASS is not None:
        from models.tcn_model import create_model
        model = create_model(
            input_dim=input_dim,
            num_classes=num_classes,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            use_attention=True
        )
    else:
        raise ImportError("No model class available. Please check imports.")
    
    return model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train PSL Model V2')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (auto-detected if not set)')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--sequence_length', type=int, default=60, help='Sequence length')
    parser.add_argument('--no_augment', action='store_true', help='Disable data augmentation')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint (e.g., latest_checkpoint.pth or checkpoint_epoch_10.pth)')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Checkpoint directory path (auto-detected from experiment_name if not set)')
    args = parser.parse_args()
    
    # Get hardware info
    hw_info = get_hardware_info()
    device = torch.device(hw_info['device'])
    
    # Auto-detect batch size if not set
    if args.batch_size is None:
        args.batch_size = hw_info['recommended_batch_size']
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or f"psl_v2_{timestamp}"
    logger = setup_logging(V2_LOGS_DIR, experiment_name)
    
    logger.info("=" * 60)
    logger.info("PSL RECOGNITION V2 - TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    if hw_info['cuda_available']:
        logger.info(f"GPU: {hw_info['gpu_name']}")
        logger.info(f"GPU Memory: {hw_info['gpu_memory_gb']:.1f} GB")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Sequence length: {args.sequence_length}")
    logger.info(f"Augmentation: {not args.no_augment}")
    
    # Create dataloaders
    logger.info("-" * 60)
    logger.info("Loading dataset...")
    
    train_loader, val_loader, test_loader, dataset_info = create_dataloaders(
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        num_workers=0,  # Windows compatibility
        use_weighted_sampler=True,
        augment_train=not args.no_augment
    )
    
    # Create model
    logger.info("-" * 60)
    logger.info("Creating model...")
    
    model = create_model_v2(
        input_dim=189,
        num_classes=dataset_info['num_classes'],
        num_channels=[256, 256, 256, 256, 128],
        kernel_size=5,
        dropout=0.4
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {MODEL_TYPE}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Training config
    config = {
        'input_dim': 189,
        'num_classes': dataset_info['num_classes'],
        'sequence_length': args.sequence_length,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': training_config_v2.WEIGHT_DECAY,
        'gradient_clip': training_config_v2.GRADIENT_CLIP_NORM,
        'scheduler_patience': training_config_v2.SCHEDULER_PATIENCE,
        'scheduler_factor': training_config_v2.SCHEDULER_FACTOR,
        'early_stopping_patience': training_config_v2.EARLY_STOPPING_PATIENCE,
        'num_channels': [256, 256, 256, 256, 128],
        'kernel_size': 5,
        'dropout': 0.4
    }
    
    # Handle resume from checkpoint
    start_epoch = 1
    resume_checkpoint = None
    
    if args.resume or args.checkpoint_dir:
        # Determine checkpoint directory
        if args.checkpoint_dir:
            checkpoint_dir = Path(args.checkpoint_dir)
        else:
            # Use experiment name to find checkpoint directory
            checkpoint_dir = V2_CHECKPOINTS_DIR / experiment_name
        
        # Determine checkpoint file path
        if args.resume:
            checkpoint_path = checkpoint_dir / args.resume
        else:
            # Auto-detect: try latest_checkpoint.pth
            checkpoint_path = checkpoint_dir / 'latest_checkpoint.pth'
        
        if checkpoint_path.exists():
            logger.info("=" * 60)
            logger.info("RESUMING FROM CHECKPOINT")
            logger.info("=" * 60)
            logger.info(f"Checkpoint path: {checkpoint_path}")
            
            try:
                resume_checkpoint = torch.load(checkpoint_path, map_location=device)
                start_epoch = resume_checkpoint['epoch'] + 1
                
                logger.info(f"Checkpoint epoch: {resume_checkpoint['epoch']}")
                logger.info(f"Resuming from epoch: {start_epoch}")
                logger.info(f"Best validation accuracy so far: {resume_checkpoint.get('best_val_acc', 0.0):.2f}%")
                
                # Update config from checkpoint if available
                if 'config' in resume_checkpoint:
                    checkpoint_config = resume_checkpoint['config']
                    # Merge checkpoint config with current config (current takes precedence for some settings)
                    config.update({k: v for k, v in checkpoint_config.items() 
                                 if k not in ['batch_size', 'learning_rate']})  # Keep current batch_size and lr
                    logger.info("Config loaded from checkpoint")
                
                # Load model state
                model.load_state_dict(resume_checkpoint['model_state_dict'])
                logger.info("Model weights loaded from checkpoint")
                
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                logger.warning("Starting training from scratch")
                start_epoch = 1
                resume_checkpoint = None
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            logger.warning("Starting training from scratch")
            start_epoch = 1
    
    # Create trainer
    trainer = TrainerV2(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        dataset_info=dataset_info,
        device=device,
        config=config,
        experiment_name=experiment_name
    )
    
    # Load optimizer, scheduler, and history states if resuming
    if resume_checkpoint:
        try:
            trainer.optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
            trainer.scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
            trainer.best_val_acc = resume_checkpoint.get('best_val_acc', 0.0)
            trainer.best_epoch = resume_checkpoint.get('epoch', 0)
            trainer.history = resume_checkpoint.get('history', {
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': [],
                'learning_rate': [],
                'epoch_time': []
            })
            
            logger.info("Optimizer state loaded")
            logger.info("Scheduler state loaded")
            logger.info("Training history loaded")
            logger.info(f"Previous training history: {len(trainer.history['train_loss'])} epochs")
            logger.info("-" * 60)
        except Exception as e:
            logger.warning(f"Failed to load some checkpoint states: {e}")
            logger.warning("Continuing with default states")
    
    # Train
    results = trainer.train(epochs=args.epochs, start_epoch=start_epoch)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 60)
    print(f"Experiment: {experiment_name}")
    print(f"Best Validation Accuracy: {results['best_val_acc']:.2f}%")
    print(f"Test Accuracy: {results['test_accuracy']:.2f}%")
    print(f"Training Time: {results['total_time']/60:.1f} minutes")
    print(f"Model saved to: {results['model_path']}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()

