# PSL Training Guide - V2

## Training Pipeline for Complete Urdu Alphabet Recognition

This document explains the dataset format, preprocessing, model architecture, and training process for the PSL Recognition System V2.

---

## 📁 Dataset Structure

### Raw Data Location
```
backend/data/Pakistan Sign Language Urdu Alphabets/
├── 1-Hay/       (659 samples: 615 jpg + 44 JPG + 1 mp4)
├── 2-Hay/       (87 samples: 87 mp4)
├── Ain/         (587 samples: mixed jpg/mp4)
├── Alif/        (661 samples: mixed)
├── ...
└── Zuey/        (587 samples: mixed)
```

### Extracted Features
```
backend/data/features_temporal/
├── 1-Hay/       (659 .npy files)
├── 2-Hay/       (87 .npy files)
├── ...          (Each .npy is 189-dimensional features)
└── Zuey/        (586 .npy files)
```

### Data Splits (V2)
```
backend/data/splits_v2/
├── train_v2.txt      # 16,262 samples (70%)
├── val_v2.txt        # 3,465 samples (15%)
├── test_v2.txt       # 3,531 samples (15%)
├── label_map_v2.json # Label to index mapping
└── dataset_info_v2.json # Dataset statistics
```

---

## 🔧 Feature Format

### Dimensions
- **Per frame**: 189 features
  - 126 hand landmarks (2 hands × 21 points × 3 coords)
  - 63 padding zeros (for model compatibility)

### File Format
- NumPy `.npy` files
- Shape: `(sequence_length, 189)` or `(189,)` for single frames
- Normalized coordinates (0-1 range relative to frame)

---

## 🏗️ Model Architecture

### Enhanced TCN (Temporal Convolutional Network)
```
Input: (batch, 60, 189)
  ↓
TemporalBlock_1: Conv1D(189 → 256, kernel=5) + BatchNorm + ReLU + Dropout
  ↓
TemporalBlock_2: Conv1D(256 → 256, kernel=5) + BatchNorm + ReLU + Dropout
  ↓
TemporalBlock_3: Conv1D(256 → 256, kernel=5) + BatchNorm + ReLU + Dropout
  ↓
TemporalBlock_4: Conv1D(256 → 256, kernel=5) + BatchNorm + ReLU + Dropout
  ↓
TemporalBlock_5: Conv1D(256 → 128, kernel=5) + BatchNorm + ReLU + Dropout
  ↓
GlobalAveragePooling1D
  ↓
FullyConnected: 128 → 40 classes
  ↓
Output: Softmax probabilities
```

### Parameters
- **Total**: 2,872,744
- **Trainable**: 2,872,744

---

## 📊 Data Augmentation

Applied only during training:

| Augmentation | Range | Probability |
|-------------|-------|-------------|
| Temporal Jitter | ±10% shift | 50% |
| Gaussian Noise | σ = 0.01 | 50% |
| Scale | 0.9 - 1.1× | 50% |

---

## ⚙️ Training Hyperparameters

### Default Configuration
```python
# Optimizer
optimizer = "AdamW"
learning_rate = 5e-4 (0.0005)
weight_decay = 1e-4

# Training
batch_size = 8 (CPU) / 32-64 (GPU)
epochs = 100
sequence_length = 60

# Scheduler
scheduler = "ReduceLROnPlateau"
patience = 10
factor = 0.5

# Early Stopping
patience = 15
min_delta = 1e-4

# Gradient Clipping
max_norm = 1.0
```

### Class Imbalance Handling
- Weighted CrossEntropy loss (inverse frequency weights)
- WeightedRandomSampler for balanced batches
- Class weights range: 0.6 (majority) to 4.7 (minority)

---

## 🚀 Running Training

### Step 1: Prepare Dataset
```bash
cd backend
python training/prepare_dataset_v2.py
```

This creates:
- Train/val/test splits
- Label mapping
- Dataset statistics report

### Step 2: Start Training
```bash
python training/train_pipeline_v2.py --epochs 100 --lr 0.0005
```

### Command Line Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 100 | Number of training epochs |
| `--batch_size` | Auto | Batch size (auto-detected) |
| `--lr` | 5e-4 | Learning rate |
| `--sequence_length` | 60 | Frames per sequence |
| `--no_augment` | False | Disable augmentation |
| `--experiment_name` | Auto | Experiment name for logging |

### Example Commands
```bash
# Full training with defaults
python training/train_pipeline_v2.py

# Custom settings
python training/train_pipeline_v2.py --epochs 50 --lr 0.001 --batch_size 16

# Quick test (fewer epochs)
python training/train_pipeline_v2.py --epochs 5
```

---

## 📈 Training Output

### Checkpoints
```
backend/saved_models/v2/checkpoints/<experiment_name>/
├── best_model.pth        # Best validation accuracy
├── latest_checkpoint.pth # Most recent
├── checkpoint_epoch_10.pth
├── checkpoint_epoch_20.pth
├── ...
└── training_history.json
```

### Final Model
```
backend/saved_models/v2/
├── psl_model_v2.pth          # Model weights
├── psl_model_v2_config.json  # Configuration
└── psl_model_v2_labels.txt   # Class labels
```

### Logs
```
backend/logs/v2/
├── <experiment_name>.log     # Training log
└── dataset_preparation.log   # Dataset prep log
```

---

## 📊 Expected Metrics

### Baseline Performance (4 classes, V1)
- Validation Accuracy: 97.4%
- Test Accuracy: 100%

### Target Performance (40 classes, V2)
- Validation Accuracy: 80-90%
- Test Accuracy: 80-90%

### Training Time
- **CPU**: 30-50 hours (100 epochs)
- **GPU**: 2-4 hours (100 epochs)

---

## ⚠️ Common Issues

### Out of Memory
```
Solution: Reduce batch size
python train_pipeline_v2.py --batch_size 4
```

### Slow Training on CPU
```
Expected: ~1 batch/second on CPU
Consider: Using GPU or fewer epochs
python train_pipeline_v2.py --epochs 30
```

### Class Imbalance Warning
```
Normal: Dataset has 13x imbalance (87 to 1152 samples)
Handled: Weighted loss and balanced sampling
```

### Model Not Converging
```
Try: Lower learning rate
python train_pipeline_v2.py --lr 0.0001
```

---

## 🔍 Monitoring Training

### Check Progress
```bash
# View recent logs
Get-Content backend\logs\v2\*.log -Tail 50

# Check checkpoint directory
dir backend\saved_models\v2\checkpoints\
```

### Training Metrics in Log
```
Epoch 1/100 | Train: 45.23% (loss=2.1234) | Val: 52.10% (loss=1.9876) | LR: 0.000500 | Time: 1823.5s
Epoch 2/100 | Train: 58.67% (loss=1.5432) | Val: 61.30% (loss=1.4321) | LR: 0.000500 | Time: 1801.2s [BEST]
```

---

## 📝 Post-Training

### Evaluate Model
The training script automatically evaluates on the test set and provides:
- Overall accuracy
- Per-class accuracy
- Top confusion pairs

### Deploy Model
Model is automatically saved to `backend/saved_models/v2/psl_model_v2.pth`

Update `backend/saved_models/model_registry.json` to include the new model if needed.

---

## 🔄 Resuming Training

Currently not supported directly. To continue training:
1. Load the checkpoint
2. Create model with same architecture
3. Load state dict
4. Continue training loop

Future improvement: Add `--resume` flag.

---

## 📚 References

- MediaPipe Hands: https://google.github.io/mediapipe/solutions/hands.html
- Temporal Convolutional Networks: https://arxiv.org/abs/1803.01271
- Sign Language Recognition: Various research papers on continuous SLR

