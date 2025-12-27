#!/usr/bin/env python3
"""
Advanced training script for PSL Recognition with SOTA techniques:
- Mixed precision training (FP16) for speed
- Cosine annealing with warmup
- Label smoothing for better generalization
- Gradient accumulation for effective larger batch sizes
- EMA (Exponential Moving Average) for stable inference
- Comprehensive logging and checkpointing
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.advanced_spatiotemporal_model import create_advanced_model
from training.dataset import PSLTemporalDataset, create_dataloaders


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing for better generalization."""
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        log_prob = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prob).sum(dim=1).mean()
        return loss


class EMA:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()
        
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class AdvancedTrainer:
    """Advanced trainer with SOTA techniques."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = 'cuda',
        save_dir: Path = None,
        test_loader: Optional[DataLoader] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader  # Optional test loader for final evaluation
        self.config = config
        self.device = device
        self.save_dir = save_dir or Path('saved_models')
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer (AdamW with decoupled weight decay)
        self.optimizer = AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # Learning rate scheduler (cosine annealing with warm restarts)
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.get('scheduler_t0', 10),
            T_mult=config.get('scheduler_tmult', 2),
            eta_min=config.get('min_lr', 1e-6),
        )
        
        # Loss function with label smoothing
        self.criterion = LabelSmoothingCrossEntropy(
            smoothing=config.get('label_smoothing', 0.1)
        )
        
        # Mixed precision training
        self.use_amp = config.get('mixed_precision', True) and device == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # EMA for model parameters
        self.ema = EMA(model, decay=config.get('ema_decay', 0.999))
        
        # Gradient accumulation
        self.accum_steps = config.get('gradient_accumulation_steps', 1)
        
        # Tracking
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
        }
        
        logger.info("=" * 80)
        logger.info("Trainer Initialized")
        logger.info("=" * 80)
        logger.info(f"Device: {device}")
        logger.info(f"Mixed precision: {self.use_amp}")
        logger.info(f"EMA decay: {config.get('ema_decay', 0.999)}")
        logger.info(f"Gradient accumulation steps: {self.accum_steps}")
        logger.info(f"Label smoothing: {config.get('label_smoothing', 0.1)}")
        logger.info("=" * 80)
    
    def train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]", ncols=100)
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Mixed precision training
            if self.use_amp:
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss = loss / self.accum_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.accum_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get('max_grad_norm', 1.0)
                    )
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    # Update EMA
                    self.ema.update()
            else:
                # Standard training (FP32)
                output = self.model(data)
                loss = self.criterion(output, target)
                loss = loss / self.accum_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get('max_grad_norm', 1.0)
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.ema.update()
            
            # Metrics
            total_loss += loss.item() * self.accum_steps
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss / (batch_idx + 1):.4f}',
                'acc': f'{100. * correct / total:.2f}%',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}',
            })
        
        return {
            'loss': total_loss / len(self.train_loader),
            'accuracy': 100. * correct / total,
            'lr': self.optimizer.param_groups[0]['lr'],
        }
    
    @torch.no_grad()
    def validate(self, epoch: int, use_ema: bool = True) -> Dict:
        """Validate the model."""
        if use_ema:
            self.ema.apply_shadow()
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Per-class metrics
        num_classes = self.config['num_classes']
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]", ncols=100)
        
        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)
            
            if self.use_amp:
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Per-class accuracy
            for i in range(len(target)):
                label = target[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
            
            pbar.set_postfix({
                'loss': f'{total_loss / (len(class_correct) + 1):.4f}',
                'acc': f'{100. * correct / total:.2f}%',
            })
        
        if use_ema:
            self.ema.restore()
        
        # Calculate per-class accuracy
        class_accuracies = []
        for i in range(num_classes):
            if class_total[i] > 0:
                acc = 100. * class_correct[i] / class_total[i]
                class_accuracies.append(acc)
            else:
                class_accuracies.append(0.0)
        
        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': 100. * correct / total,
            'class_accuracies': class_accuracies,
        }
    
    def evaluate_test_set(self, use_ema: bool = True) -> Dict:
        """
        Evaluate the model on the test set.
        
        Args:
            use_ema: Whether to use EMA weights for evaluation
            
        Returns:
            Dictionary containing test metrics (loss, accuracy, class accuracies)
        """
        if not hasattr(self, 'test_loader') or self.test_loader is None:
            logger.warning("No test loader available for evaluation")
            return {
                'loss': 0.0,
                'accuracy': 0.0,
                'class_accuracies': [0.0] * self.config['num_classes']
            }
        
        if use_ema and hasattr(self, 'ema') and self.ema is not None:
            self.ema.apply_shadow()
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Per-class metrics
        num_classes = self.config['num_classes']
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
        
        logger.info("Running evaluation on test set...")
        pbar = tqdm(self.test_loader, desc="Test Evaluation", ncols=100)
        
        with torch.no_grad():
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                if self.use_amp:
                    with autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # Per-class accuracy
                for i in range(len(target)):
                    label = target[i].item()
                    class_total[label] += 1
                    if predicted[i] == label:
                        class_correct[label] += 1
                
                pbar.set_postfix({
                    'loss': f'{total_loss / (pbar.n + 1):.4f}',
                    'acc': f'{100. * correct / total:.2f}%',
                })
        
        if use_ema and hasattr(self, 'ema') and self.ema is not None:
            self.ema.restore()
        
        # Calculate per-class accuracy
        class_accuracies = []
        for i in range(num_classes):
            if class_total[i] > 0:
                acc = 100. * class_correct[i] / class_total[i]
                class_accuracies.append(acc)
            else:
                class_accuracies.append(0.0)
        
        return {
            'loss': total_loss / len(self.test_loader),
            'accuracy': 100. * correct / total,
            'class_accuracies': class_accuracies,
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'ema_shadow': self.ema.shadow,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'history': self.train_history,
        }
        
        # Save latest checkpoint
        latest_path = self.save_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"[OK] Best model saved to {best_path}")
        
        # Save epoch checkpoint every N epochs
        if (epoch + 1) % 10 == 0:
            epoch_path = self.save_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save(checkpoint, epoch_path)
    
    def train(self, num_epochs: int):
        """Train the model for multiple epochs."""
        logger.info("=" * 80)
        logger.info(f"Starting Training for {num_epochs} Epochs")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            self.train_history['train_loss'].append(train_metrics['loss'])
            self.train_history['train_acc'].append(train_metrics['accuracy'])
            self.train_history['lr'].append(train_metrics['lr'])
            
            # Validate
            val_metrics = self.validate(epoch, use_ema=True)
            self.train_history['val_loss'].append(val_metrics['loss'])
            self.train_history['val_acc'].append(val_metrics['accuracy'])
            
            # Update learning rate
            self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            # Log metrics
            logger.info("=" * 80)
            logger.info(f"Epoch {epoch+1}/{num_epochs} Summary (Time: {epoch_time:.2f}s)")
            logger.info("-" * 80)
            logger.info(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
            logger.info(f"Val Loss:   {val_metrics['loss']:.4f} | Val Acc:   {val_metrics['accuracy']:.2f}%")
            logger.info(f"Learning Rate: {train_metrics['lr']:.6f}")
            
            # Per-class accuracies
            labels = self.config.get('labels', [f'Class {i}' for i in range(self.config['num_classes'])])
            logger.info("-" * 80)
            logger.info("Per-Class Accuracies:")
            for label, acc in zip(labels, val_metrics['class_accuracies']):
                logger.info(f"  {label:15s}: {acc:.2f}%")
            logger.info("=" * 80)
            
            # Save best model
            is_best = False
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.best_val_loss = val_metrics['loss']
                self.epochs_without_improvement = 0
                is_best = True
                logger.info(f"🎉 New best validation accuracy: {val_metrics['accuracy']:.2f}%")
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best=is_best)
            
            # Early stopping
            patience = self.config.get('early_stopping_patience', 15)
            if self.epochs_without_improvement >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                logger.info(f"No improvement for {patience} epochs")
                break
        
        # Training complete
        total_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info("Training Completed!")
        logger.info("=" * 80)
        logger.info(f"Total training time: {total_time / 3600:.2f} hours")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info("=" * 80)
        
        # Save training history
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")
        
        # Save final config
        config_path = self.save_dir / 'training_config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Training config saved to {config_path}")
        
        return self.train_history


def main():
    """Main training entry point."""
    
    # Configuration
    config = {
        # Model
        'input_dim': 189,
        'num_classes': 4,  # Start with 4, scalable to 40+
        'model_size': 'base',  # 'tiny', 'small', 'base', 'large'
        'labels': ['2-Hay', 'Alifmad', 'Aray', 'Jeem'],
        
        # Training
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'min_lr': 1e-6,
        
        # Optimization
        'mixed_precision': True,
        'gradient_accumulation_steps': 1,
        'max_grad_norm': 1.0,
        
        # Regularization
        'label_smoothing': 0.1,
        'ema_decay': 0.999,
        'dropout': 0.2,
        
        # Scheduler
        'scheduler_t0': 10,
        'scheduler_tmult': 2,
        
        # Early stopping
        'early_stopping_patience': 15,
        
        # Data
        'target_sequence_length': 60,
        'num_workers': 4,
    }
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create save directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(f'saved_models/advanced_model_{timestamp}')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    logger.info("Creating model...")
    model = create_advanced_model(
        input_dim=config['input_dim'],
        num_classes=config['num_classes'],
        model_size=config['model_size'],
    )
    
    logger.info("Model created:")
    info = model.get_model_info()
    for key, value in info.items():
        logger.info(f"  {key}: {value}")
    
    # Load data
    logger.info("Loading data...")
    data_dir = Path('backend/data/features_temporal')
    
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=data_dir,
            labels=config['labels'],
            batch_size=config['batch_size'],
            target_sequence_length=config['target_sequence_length'],
            num_workers=config['num_workers'],
        )
        
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")
        logger.info(f"Test batches: {len(test_loader)}")
        
        # Create trainer
        trainer = AdvancedTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            save_dir=save_dir,
        )
        
        # Train
        history = trainer.train(config['num_epochs'])
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

