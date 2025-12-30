# Data Flow Diagrams (DFD)
## Pakistan Sign Language Recognition System

---

## Fig 3.4. Level 0 DFD (Context Diagram)

### Description
The Level 0 DFD shows the system as a single process with external entities and major data flows.

### Diagram Structure

```
┌─────────────┐
│             │
│    USER     │
│             │
└──────┬──────┘
       │
       │ Video Stream (Hand Signs)
       │ Control Commands (Start/Stop/Reset)
       │ Feedback (Correct/Incorrect)
       │
       ↓
┌──────────────────────────────────────┐
│                                      │
│   PSL RECOGNITION SYSTEM (V2.0)      │
│   (Pakistan Sign Language            │
│    Recognition System)               │
│                                      │
└──────────────┬───────────────────────┘
               │
               │ Recognized Signs
               │ Confidence Scores
               │ System Status
               │ Real-time Feedback
               │
               ↓
       ┌───────┴──────┐
       │              │
       │    USER      │
       │              │
       └──────────────┘


External Entity: USER
Process: PSL Recognition System
Data Flows:
  Input:
    - Video Stream (webcam feed with hand signs)
    - Control Commands (start, stop, reset recognition)
    - User Feedback (correct/incorrect predictions)
  
  Output:
    - Recognized Signs (Urdu alphabet predictions)
    - Confidence Scores (prediction accuracy %)
    - System Status (ready, recognizing, processing)
    - Recognition History (recent signs)
```

### Textual Representation for Drawing

**External Entities (Rectangles):**
- USER

**Process (Circle/Rounded Rectangle):**
- 0.0 - PSL Recognition System

**Data Flows (Arrows with Labels):**

FROM USER TO SYSTEM:
1. Video Stream (Hand Signs)
2. Control Commands (Start/Stop/Reset)
3. User Feedback (Correct/Incorrect)

FROM SYSTEM TO USER:
4. Recognized Signs (Urdu Alphabet)
5. Confidence Scores (%)
6. System Status
7. Recognition History

---

## Fig 3.5. Level 1 DFD (Detailed Processes)

### Description
The Level 1 DFD breaks down the main system into detailed sub-processes showing internal data flows.

### Diagram Structure

```
                    ┌─────────────┐
                    │    USER     │
                    └──────┬──────┘
                           │
                           │ Video Stream
                           ↓
              ┌────────────────────────┐
              │   1.0                  │
              │   CAPTURE VIDEO        │
              │   (Camera Manager)     │
              └───────────┬────────────┘
                          │
                          │ Raw Frames (15 FPS)
                          ↓
              ┌────────────────────────┐
              │   2.0                  │
              │   EXTRACT LANDMARKS    │
              │   (MediaPipe Hands)    │
              └───────────┬────────────┘
                          │
                          │ Hand Keypoints (63/126 values)
                          ↓
              ┌────────────────────────┐
              │   3.0                  │
              │   BUFFER & FILTER      │
              │   (Frame Buffer)       │
              └───────────┬────────────┘
                          │
                          │ Normalized Sequence (45 frames)
                          ↓
              ┌────────────────────────┐
              │   4.0                  │
              │   RECOGNIZE SIGN       │
              │   (TCN Model)          │
              └───────────┬────────────┘
                          │
                          │ Raw Predictions + Scores
                          ↓
              ┌────────────────────────┐
              │   5.0                  │
              │   STABILIZE PREDICTION │
              │   (Voting + Thresholding)│
              └───────────┬────────────┘
                          │
                          │ Stable Prediction
                          ↓
              ┌────────────────────────┐
              │   6.0                  │
              │   UPDATE UI            │
              │   (Display Results)    │
              └───────────┬────────────┘
                          │
                          │ Visual Output
                          ↓
                    ┌──────┴──────┐
                    │    USER     │
                    └──────┬──────┘
                           │
                           │ User Feedback
                           ↓
              ┌────────────────────────┐
              │   7.0                  │
              │   STORE FEEDBACK       │
              │   (Feedback System)    │
              └────────────────────────┘
                          │
                          │ Feedback Data
                          ↓
                      [D1: Feedback DB]
                      [D2: Sign Thresholds]
```

### Detailed Process Descriptions

**Process 1.0: CAPTURE VIDEO**
- Input: Video stream from webcam
- Process: Initialize camera, capture frames at 15 FPS
- Output: Raw video frames (640×480)
- Technology: Camera.js + WebRTC

**Process 2.0: EXTRACT LANDMARKS**
- Input: Raw video frames
- Process: Detect hands, extract 21 landmarks per hand (x, y, z)
- Output: Keypoint array (63 values for 1 hand, 126 for 2 hands)
- Technology: MediaPipe Hands (detection confidence: 0.3)

**Process 3.0: BUFFER & FILTER**
- Input: Hand keypoints
- Process: 
  - Accumulate keypoints in sliding window (45 frames)
  - Normalize coordinates (0-1 range)
  - Filter noise and interpolate missing frames
- Output: Normalized sequence ready for prediction
- Technology: Frame buffer (deque), normalization algorithms

**Process 4.0: RECOGNIZE SIGN**
- Input: Normalized sequence (45 frames × 126 features)
- Process: 
  - Forward pass through Enhanced TCN model
  - 7 temporal blocks with attention
  - Softmax classification (40 classes)
- Output: Prediction probabilities for each sign
- Technology: PyTorch TCN model (1.2M parameters)

**Process 5.0: STABILIZE PREDICTION**
- Input: Raw predictions with confidence scores
- Process:
  - Majority voting (5 recent predictions, need 3/5 agreement)
  - Per-sign confidence thresholding (0.60 base)
  - 3-second cooldown for same sign
  - Clear history on sign change
- Output: Stable, reliable prediction
- Technology: Voting algorithm, temporal filtering

**Process 6.0: UPDATE UI**
- Input: Stable prediction with confidence
- Process:
  - Display Urdu sign in prediction card
  - Update confidence meter (%)
  - Add to history list
  - Show feedback buttons
  - Update buffer progress
- Output: Visual feedback to user
- Technology: JavaScript UI manager, WebSocket updates

**Process 7.0: STORE FEEDBACK**
- Input: User feedback (correct/incorrect) + prediction details
- Process:
  - Log feedback to JSON file
  - Update per-sign accuracy statistics
  - Recalculate confidence thresholds
- Output: Feedback records, updated thresholds
- Technology: Python logging, JSON storage

**Data Stores:**
- **D1: Feedback Database** (backend/data/feedback/feedback_YYYYMMDD.json)
  - Stores user feedback for accuracy tracking
  
- **D2: Sign Thresholds** (backend/config/sign_thresholds.json)
  - Dynamic confidence thresholds per sign (0.55-0.75)

- **D3: Model Registry** (backend/saved_models/model_registry.json)
  - Trained model metadata and paths

- **D4: Recognition History** (Frontend in-memory)
  - Recent signs displayed in UI

---

## Data Flow Summary

### Primary Data Pipeline
1. **Video Capture** → Raw Frames (15 FPS)
2. **Landmark Extraction** → Keypoints (63/126 values)
3. **Buffering** → Sequence (45 frames)
4. **Recognition** → Predictions (40 classes)
5. **Stabilization** → Stable Sign
6. **Display** → User Interface

### Secondary Flows
- **Control Flow**: User commands (Start/Stop/Reset) → System state changes
- **Feedback Loop**: User feedback → Storage → Threshold updates
- **Status Updates**: All processes → UI status indicators (FPS, buffer, confidence)

### Real-time Constraints
- **Frame Rate**: 15 FPS capture
- **Latency**: ~2.0s from sign start to prediction
- **Buffer Size**: 45 frames (3 seconds at 15 FPS)
- **Prediction Rate**: Every 5 frames (continuous)

---

## Technical Implementation Details

### Communication Protocol
- **Frontend ↔ Backend**: WebSocket (Socket.IO)
- **Events**: 
  - `frame_data`: Video frames (base64 encoded)
  - `prediction`: Recognition results
  - `status`: System status updates
  - `feedback`: User feedback submissions

### Data Formats
- **Video Frame**: Base64 encoded JPEG
- **Keypoints**: Float array [x1,y1,z1, x2,y2,z2, ...]
- **Prediction**: {label: string, confidence: float, timestamp: int}
- **Feedback**: {prediction: string, is_correct: bool, confidence: float}

---

## Drawing Instructions

### For Level 0 DFD:
1. Draw 1 external entity (rectangle): "User"
2. Draw 1 process (circle): "0.0 PSL Recognition System"
3. Draw 7 data flows (arrows with labels) as listed above

### For Level 1 DFD:
1. Draw 1 external entity: "User" (appears twice - source and sink)
2. Draw 7 processes (circles): 1.0 through 7.0
3. Draw 4 data stores (open rectangles): D1, D2, D3, D4
4. Connect with arrows showing data flow direction
5. Label all arrows with data descriptions

### Recommended Tools:
- Microsoft Visio
- Lucidchart
- Draw.io (free)
- PowerPoint with SmartArt

---

## Notes for Report
- Both diagrams show the **real-time, continuous recognition** nature of the system
- Emphasize the **feedback loop** for system improvement
- Highlight **multi-stage processing** (capture → extract → buffer → recognize → stabilize)
- Show **data persistence** through data stores
- Demonstrate **user interaction** at multiple points

**Document Version**: 1.0  
**Date**: December 26, 2025  
**System**: PSL Recognition System V2.0  
**40 Urdu Alphabet Signs**

