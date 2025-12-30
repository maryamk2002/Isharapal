# 🚀 Quick Start Guide - Enhanced PSL Recognition System

## ⚡ **Start the System (ONE COMMAND)**

```bash
START_V2.bat
```

Then open: **http://localhost:5000/index_v2.html**

---

## 🎯 **What's New in This Version**

### ✅ **1. Keypoint Filtering**
- **What:** Smooths hand landmarks to reduce webcam jitter
- **How:** Moving average over 5 frames
- **Benefit:** Cleaner, more stable keypoint visualization

### ✅ **2. Stuck Sequence Detection**
- **What:** Detects when hand isn't moving for 15 frames
- **How:** Measures movement between frames
- **Benefit:** Prevents buffer from filling with identical frames
- **User Feedback:** Shows warning "Please move your hand"

### ✅ **3. Automatic Recording**
- **What:** Auto-starts/stops recording based on sign detection
- **How:** Starts on first prediction, stops after 2s idle
- **Benefit:** Natural UX, clear segment boundaries
- **Display:** Green pulsing dot = recording, Red dot = stopped

### ✅ **4. Recording Segments**
- **What:** Tracks each completed sign with metadata
- **Info:** Label, confidence, duration, timestamp
- **Saved:** `backend/data/recordings/segments_{session}.csv`
- **Display:** Last 10 segments in UI

### ✅ **5. Performance Monitoring**
- **What:** Tracks FPS, inference time, accuracy
- **How:** Rolling averages, per-label accuracy via feedback
- **Saved:** `backend/logs/metrics/metrics_{session}.json`
- **Display:** FPS in status card

### ✅ **6. Accuracy Tracking**
- **What:** Per-label accuracy based on user feedback
- **How:** Click 👍 (correct) or 👎 (incorrect) after predictions
- **Benefit:** Identify problematic signs for retraining

---

## 📊 **What You'll See**

### **Recording Status Indicator:**
- 🔘 **Gray dot** = Idle (ready)
- 🟢 **Pulsing green dot** = Recording (sign detected)
- 🔴 **Red dot** = Stopped (2s idle timeout)

### **Segment History:**
```
┌─────────────────────────────────┐
│ Recording Segments              │
├─────────────────────────────────┤
│ Alif          94%   1.2s  12:34 │
│ Bay           87%   1.5s  12:32 │
│ Jeem          91%   1.3s  12:30 │
└─────────────────────────────────┘
```

### **Performance Metrics:**
- **FPS:** ~15 (target)
- **Inference Time:** ~25-50ms
- **Detection Rate:** ~80-90%

---

## 🎮 **How to Use**

### **1. Start Recognition:**
1. Click **"Start"** button
2. Allow webcam access
3. Wait for "System ready" message

### **2. Make Signs:**
1. Position hand in front of webcam
2. Make a sign (hold for ~1 second)
3. Watch recording indicator turn green
4. See prediction appear

### **3. Provide Feedback:**
1. After prediction appears, click:
   - 👍 **Correct** - if prediction is right
   - 👎 **Incorrect** - if prediction is wrong
2. Accuracy is tracked per-label

### **4. View Segments:**
1. Check "Recording Segments" panel
2. See completed signs with timestamps
3. Export data from `backend/data/recordings/`

---

## ⚙️ **Configuration (Optional)**

Edit `backend/config_v2.py` to adjust:

```python
# Filtering
MOVING_AVERAGE_WINDOW = 5  # Increase for more smoothing
STUCK_THRESHOLD_FRAMES = 15  # Increase to be less sensitive

# Recording
IDLE_TIMEOUT_SEC = 2.0  # Increase to wait longer before stopping
MIN_CONFIDENCE_FOR_RECORDING = 0.6  # Increase for stricter recording

# Monitoring
AUTO_SAVE_INTERVAL_SEC = 60.0  # How often to save metrics
```

---

## 📁 **Where Data is Saved**

### **Recording Segments:**
```
backend/data/recordings/
├── segments_{session_id}.csv
└── session_{session_id}_{timestamp}.json
```

### **Performance Metrics:**
```
backend/logs/metrics/
└── metrics_{session_id}_{timestamp}.json
```

### **Feedback Data:**
```
backend/data/feedback.db  (SQLite database)
```

---

## 🐛 **Troubleshooting**

### **Problem: "Stuck sequence reset" appears frequently**
- **Cause:** Hand not moving enough
- **Fix:** Increase `STUCK_MOVEMENT_THRESHOLD` to 0.01 in `config_v2.py`

### **Problem: Recording stops too quickly**
- **Cause:** Idle timeout too short
- **Fix:** Increase `IDLE_TIMEOUT_SEC` to 3.0 in `config_v2.py`

### **Problem: Keypoints still jittery**
- **Cause:** Moving average window too small
- **Fix:** Increase `MOVING_AVERAGE_WINDOW` to 7 in `config_v2.py`

### **Problem: Low FPS (<10)**
- **Cause:** System overloaded
- **Fix:** Check `backend/logs/metrics/` for bottlenecks
- **Check:** Inference time (should be <50ms)

---

## 📊 **Demo Tips**

### **Before Demo:**
1. ✅ Test all 40 signs
2. ✅ Check FPS is stable (~15)
3. ✅ Verify recording segments save correctly
4. ✅ Practice smooth sign transitions

### **During Demo:**
1. 🎤 Explain recording indicator (green = active)
2. 📊 Show segment history in real-time
3. 👍 Demonstrate feedback system
4. 📈 Highlight FPS and performance metrics

### **After Demo:**
1. 💾 Export recording segments
2. 📊 Show accuracy metrics
3. 🔍 Analyze saved data

---

## 🎯 **Key Features to Highlight**

1. **Real-Time Performance:** ~15 FPS, <1s latency
2. **Automatic Recording:** No manual start/stop needed
3. **Segment Tracking:** Every sign saved with metadata
4. **Accuracy Tracking:** Per-label accuracy via feedback
5. **Stuck Detection:** Auto-resets when hand stops moving
6. **Smooth Keypoints:** Jitter filtering for clean visualization

---

## ✅ **System Status**

| Component | Status |
|-----------|--------|
| Model (97.2% accuracy) | ✅ Working |
| MediaPipe (hand detection) | ✅ Working |
| Keypoint filtering | ✅ Implemented |
| Recording automation | ✅ Implemented |
| Performance monitoring | ✅ Implemented |
| Accuracy tracking | ✅ Implemented |
| Frontend UI | ✅ Enhanced |

---

## 🚀 **You're Ready for Demo!**

The system is fully functional and demo-ready. All enhancements are integrated and working.

**Good luck with your presentation! 🎉**

