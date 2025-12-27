# PSL Recognition System V2

## Pakistan Sign Language (PSL) Translator - Complete Urdu Alphabet

**Version 2.0.0** - Production-Ready Release

This system provides real-time translation of Pakistan Sign Language (Urdu alphabet) using deep learning and computer vision.

---

## 🎯 Features

### ✅ Complete Urdu Alphabet Support
- **40 sign classes** covering all Urdu alphabet letters
- Trained on **23,258 samples** from mixed video/image dataset
- High accuracy recognition with real-time inference

### ✅ Real-Time Recognition
- WebSocket-based continuous prediction
- 32-frame sliding window for fast response
- Stability filtering to prevent flickering
- Keypoints visualization overlay on video

### ✅ Modern UI/UX
- Side-by-side layout (video + prediction panel)
- No scrolling required - everything visible
- Bilingual support (Urdu + English)
- Feedback system for user corrections

### ✅ Robust Architecture
- Enhanced TCN (Temporal Convolutional Network) model
- MediaPipe hand landmark detection
- Flask + Socket.IO backend
- Class-balanced training with weighted loss

---

## 📁 Project Structure

```
ISHARAPAL/
├── backend/
│   ├── app.py              # V1 backend (unchanged)
│   ├── app_v2.py           # V2 backend with enhancements
│   ├── config.py           # V1 configuration
│   ├── config_v2.py        # V2 configuration
│   ├── data/
│   │   ├── features_temporal/  # Extracted features (40 classes)
│   │   ├── splits_v2/          # Train/val/test splits
│   │   └── Pakistan Sign Language Urdu Alphabets/  # Raw data
│   ├── inference/
│   │   ├── predictor.py    # V1 predictor
│   │   └── predictor_v2.py # V2 predictor with improvements
│   ├── models/
│   │   └── tcn_model.py    # TCN architecture
│   ├── saved_models/
│   │   ├── v2/             # V2 model files
│   │   └── backups/        # Backup of V1 models
│   └── training/
│       ├── prepare_dataset_v2.py  # Dataset preparation
│       ├── dataset_v2.py          # PyTorch dataset
│       └── train_pipeline_v2.py   # Training pipeline
├── frontend/
│   ├── index.html          # V1 frontend
│   ├── index_v2.html       # V2 frontend
│   ├── css/
│   │   └── main_v2.css     # V2 styles
│   └── js/
│       ├── app_v2.js       # V2 main app
│       └── websocket_v2.js # V2 WebSocket handler
├── TRAIN_V2.bat            # Training script
├── START_V2.bat            # Start V2 server
└── README_V2.md            # This file
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Windows 10/11
- Webcam

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Server
**Option A: V2 Server (recommended)**
```bash
cd backend
python app_v2.py
```

**Option B: Use batch file**
```bash
START_V2.bat
```

### 3. Open the Application
Navigate to: `http://localhost:5000`

For V2 interface: `http://localhost:5000/index_v2.html`

---

## 🏋️ Training

### Using Pre-trained Model
V2 comes with a pre-trained model supporting all 40 sign classes.

### Re-training the Model

1. **Prepare Dataset**
```bash
cd backend
python training/prepare_dataset_v2.py
```

2. **Run Training**
```bash
python training/train_pipeline_v2.py --epochs 100 --lr 0.0005
```

Or use the batch file:
```bash
TRAIN_V2.bat
```

### Training Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 100 | Number of training epochs |
| `--lr` | 0.0005 | Learning rate |
| `--batch_size` | Auto | Batch size (auto-detected based on hardware) |
| `--sequence_length` | 60 | Input sequence length |
| `--no_augment` | False | Disable data augmentation |

---

## 📊 Model Performance

### Architecture
- **Model**: Enhanced TCN (Temporal Convolutional Network)
- **Parameters**: 2.87M trainable
- **Input**: 189-dimensional hand landmarks × 60 frames
- **Output**: 40 class probabilities

### V2 Improvements
| Feature | V1 | V2 |
|---------|----|----|
| Sign Classes | 4 | 40 |
| Response Time | ~3s | ~1s |
| Sliding Window | 60 frames | 32 frames |
| Stability | Basic | 3/5 voting |
| Keypoints Overlay | No | Yes |
| Feedback System | No | Yes |

---

## 🔧 Switching Between V1 and V2

### To use V1 (fallback):
```bash
cd backend
python app.py
```
Access: `http://localhost:5000`

### To use V2:
```bash
cd backend
python app_v2.py
```
Access: `http://localhost:5000/index_v2.html`

---

## 📋 Supported Signs (40 Classes)

| # | Sign Name | # | Sign Name |
|---|-----------|---|-----------|
| 1 | 1-Hay | 21 | Lam |
| 2 | 2-Hay | 22 | Meem |
| 3 | Ain | 23 | Nuun |
| 4 | Alif | 24 | Nuungh |
| 5 | Alifmad | 25 | Pay |
| 6 | Aray | 26 | Ray |
| 7 | Bay | 27 | Say |
| 8 | Byeh | 28 | Seen |
| 9 | Chay | 29 | Sheen |
| 10 | Cyeh | 30 | Suad |
| 11 | Daal | 31 | Taay |
| 12 | Dal | 32 | Tay |
| 13 | Dochahay | 33 | Tuey |
| 14 | Fay | 34 | Wao |
| 15 | Gaaf | 35 | Zaal |
| 16 | Ghain | 36 | Zaey |
| 17 | Hamza | 37 | Zay |
| 18 | Jeem | 38 | Zuad |
| 19 | Kaf | 39 | Zuey |
| 20 | Khay | 40 | Kiaf |

---

## 🐛 Troubleshooting

### Camera not detected
- Ensure webcam is connected and not in use by another application
- Grant browser permission for camera access

### Model not loading
- Check that model files exist in `backend/saved_models/v2/`
- Verify Python path and dependencies

### Slow recognition
- V2 is optimized for 15 FPS; ensure good lighting
- Keep hand clearly visible in frame

### High CPU usage
- Training is CPU-intensive; GPU recommended for faster training
- Inference runs on CPU by default

---

## 📝 API Endpoints (V2)

### WebSocket Events
| Event | Direction | Description |
|-------|-----------|-------------|
| `connect` | Server→Client | Connection established |
| `start_recognition` | Client→Server | Start recognition |
| `frame_data` | Client→Server | Send video frame |
| `frame_processed` | Server→Client | Frame processing result |
| `prediction` | Server→Client | Stable prediction |
| `feedback` | Client→Server | User feedback |

### REST API
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/model/info` | GET | Model information |
| `/api/stats` | GET | Performance statistics |
| `/api/feedback/summary` | GET | Feedback summary |

---

## 🔄 Version History

### V2.0.0 (Current)
- Complete Urdu alphabet (40 signs)
- Faster inference (32-frame window)
- Keypoints visualization
- Feedback system
- Improved UI/UX

### V1.0.0
- Initial release (4 signs)
- Basic recognition
- 60-frame window

---

## 📄 License

This project is part of a Final Year Project (FYP).

---

## 👥 Contributors

PSL Recognition System Development Team

---

## 📞 Support

For issues or questions, please check the troubleshooting section or review the code documentation.
