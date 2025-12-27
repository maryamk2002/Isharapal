"""
Comprehensive demo readiness test for IsharaPal PSL Recognition System.
Tests all acceptance criteria with automated frame sending.
"""
import cv2
import numpy as np
import base64
import socketio
import time
import sys
import threading
from collections import defaultdict
from datetime import datetime

class DemoReadinessTest:
    def __init__(self):
        self.sio = socketio.Client()
        self.metrics = {
            'frames_sent': 0,
            'frames_processed': 0,
            'landmarks_extracted': 0,
            'no_hands_count': 0,
            'predictions_received': 0,
            'latencies': [],
            'errors': [],
            'frozen_frame_count': 0,
        }
        self.frame_send_times = {}
        self.last_keypoints = None
        self.identical_keypoints_count = 0
        self.test_running = False
        self.connected = False
        self.recognition_started = False
        
        self.setup_handlers()
    
    def setup_handlers(self):
        @self.sio.on('connected')
        def on_connected(data):
            self.connected = True
            print(f"[OK] Connected: {data['session_id']}")
        
        @self.sio.on('recognition_started')
        def on_recognition_started(data):
            self.recognition_started = True
            print(f"[OK] Recognition started")
        
        @self.sio.on('frame_processed')
        def on_frame_processed(data):
            self.metrics['frames_processed'] += 1
            
            # Calculate latency
            frame_id = data.get('timestamp')
            if frame_id in self.frame_send_times:
                latency = (time.time() - self.frame_send_times[frame_id]) * 1000
                self.metrics['latencies'].append(latency)
                del self.frame_send_times[frame_id]
            
            status = data.get('status', 'unknown')
            
            if data.get('keypoints'):
                self.metrics['landmarks_extracted'] += 1
                
                # Check for frozen frames
                current_kp = str(data['keypoints'][:20])  # First 20 values
                if current_kp == self.last_keypoints:
                    self.identical_keypoints_count += 1
                    if self.identical_keypoints_count > 10:
                        self.metrics['frozen_frame_count'] += 1
                else:
                    self.identical_keypoints_count = 0
                self.last_keypoints = current_kp
            
            if status == 'no_hands':
                self.metrics['no_hands_count'] += 1
            
            if data.get('prediction'):
                self.metrics['predictions_received'] += 1
                pred = data['prediction']
                print(f"[PREDICTION] {pred['label']} ({pred['confidence']:.2%})")
        
        @self.sio.on('error')
        def on_error(data):
            error_msg = data.get('message', str(data))
            self.metrics['errors'].append(error_msg)
            print(f"[ERROR] {error_msg}")
    
    def create_test_frame(self, frame_type='open_palm', hand_count=2):
        """Create realistic test frames with hand-like patterns"""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 240  # Gray background
        
        if frame_type == 'open_palm':
            # Draw open palm(s)
            if hand_count >= 1:
                # Right hand
                cv2.circle(frame, (320, 240), 60, (200, 150, 100), -1)
                for angle in [-30, 0, 30, 60, 90]:
                    x = int(320 + 90 * np.cos(np.radians(angle - 90)))
                    y = int(240 + 90 * np.sin(np.radians(angle - 90)))
                    cv2.circle(frame, (x, y), 18, (180, 130, 90), -1)
            
            if hand_count >= 2:
                # Left hand
                cv2.circle(frame, (180, 240), 50, (200, 150, 100), -1)
                for angle in [-30, 0, 30, 60, 90]:
                    x = int(180 + 80 * np.cos(np.radians(angle - 90)))
                    y = int(240 + 80 * np.sin(np.radians(angle - 90)))
                    cv2.circle(frame, (x, y), 15, (180, 130, 90), -1)
        
        elif frame_type == 'pointing':
            # Single pointing finger
            cv2.circle(frame, (320, 280), 40, (200, 150, 100), -1)
            cv2.rectangle(frame, (310, 150), (330, 280), (180, 130, 90), -1)
        
        elif frame_type == 'fist':
            # Closed fist
            cv2.circle(frame, (320, 240), 50, (200, 150, 100), -1)
        
        # Add noise for realism
        noise = np.random.randint(-10, 10, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return frame
    
    def encode_frame(self, frame):
        """Encode frame as base64 JPEG"""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{jpg_as_text}"
    
    def send_frame(self, frame_data):
        """Send frame with timestamp for latency tracking"""
        timestamp = time.time()
        self.frame_send_times[timestamp] = timestamp
        self.sio.emit('frame_data', {'frame': frame_data, 'timestamp': timestamp})
        self.metrics['frames_sent'] += 1
    
    def test_startup(self):
        """A. Start-up checks"""
        print("\n" + "="*70)
        print("TEST A: START-UP CHECKS")
        print("="*70)
        
        try:
            print("Connecting to backend...")
            self.sio.connect('http://localhost:5000', wait_timeout=10)
            time.sleep(1)
            
            if not self.connected:
                print("[FAIL] Connection failed")
                return False
            
            print("Starting recognition...")
            self.sio.emit('start_recognition', {})
            time.sleep(1)
            
            if not self.recognition_started:
                print("[FAIL] Recognition failed to start")
                return False
            
            print("[PASS] Start-up successful")
            return True
            
        except Exception as e:
            print(f"[FAIL] Start-up error: {e}")
            return False
    
    def test_streaming(self, duration=60):
        """B. Streaming & detection smoke test"""
        print("\n" + "="*70)
        print(f"TEST B: STREAMING TEST ({duration}s)")
        print("="*70)
        
        # Ensure frame_send_times exists
        if not hasattr(self, 'frame_send_times'):
            self.frame_send_times = {}
        
        start_time = time.time()
        frame_count = 0
        fps = 15  # Realistic FPS
        frame_interval = 1.0 / fps
        
        test_sequence = [
            ('open_palm', 2, 10),   # Both hands, 10s
            ('open_palm', 1, 10),   # Single hand, 10s
            ('pointing', 1, 10),    # Pointing, 10s
            ('fist', 1, 10),        # Fist, 10s
        ]
        
        current_test = 0
        test_start = time.time()
        frame_type, hand_count, test_duration = test_sequence[current_test]
        
        print(f"Starting test: {frame_type} with {hand_count} hand(s)")
        
        try:
            while time.time() - start_time < duration:
                # Check if we should switch test
                if time.time() - test_start >= test_duration:
                    current_test += 1
                    if current_test < len(test_sequence):
                        frame_type, hand_count, test_duration = test_sequence[current_test]
                        test_start = time.time()
                        print(f"\nSwitching to: {frame_type} with {hand_count} hand(s)")
                
                # Create and send frame
                frame = self.create_test_frame(frame_type, hand_count)
                encoded = self.encode_frame(frame)
                self.send_frame(encoded)
                
                frame_count += 1
                
                # Progress update every 5s
                if frame_count % (fps * 5) == 0:
                    elapsed = time.time() - start_time
                    print(f"Progress: {elapsed:.1f}s - Sent: {self.metrics['frames_sent']}, "
                          f"Processed: {self.metrics['frames_processed']}, "
                          f"Landmarks: {self.metrics['landmarks_extracted']}")
                
                time.sleep(frame_interval)
            
            # Wait for final responses
            time.sleep(2)
            
            print(f"\n[COMPLETE] Streaming test finished")
            return True
            
        except Exception as e:
            print(f"[FAIL] Streaming error: {e}")
            return False
    
    def test_single_hand(self):
        """D. One-hand tolerance test"""
        print("\n" + "="*70)
        print("TEST D: SINGLE HAND TOLERANCE")
        print("="*70)
        
        landmarks_before = self.metrics['landmarks_extracted']
        
        print("Testing right hand only (20s)...")
        start_time = time.time()
        while time.time() - start_time < 20:
            frame = self.create_test_frame('open_palm', hand_count=1)
            encoded = self.encode_frame(frame)
            self.send_frame(encoded)
            time.sleep(1.0 / 15)
        
        time.sleep(1)
        
        landmarks_extracted = self.metrics['landmarks_extracted'] - landmarks_before
        detection_rate = landmarks_extracted / (20 * 15) * 100
        
        print(f"Single hand detection rate: {detection_rate:.1f}%")
        
        if detection_rate < 30:
            print(f"[FAIL] Single hand detection too low: {detection_rate:.1f}%")
            return False
        
        print(f"[PASS] Single hand detection: {detection_rate:.1f}%")
        return True
    
    def analyze_results(self):
        """C. Acceptance thresholds"""
        print("\n" + "="*70)
        print("TEST C: ACCEPTANCE THRESHOLDS")
        print("="*70)
        
        passed = True
        
        # 1. Landmark extraction rate
        if self.metrics['frames_sent'] > 0:
            detection_rate = (self.metrics['landmarks_extracted'] / self.metrics['frames_sent']) * 100
            print(f"1. Detection rate: {detection_rate:.1f}% ({self.metrics['landmarks_extracted']}/{self.metrics['frames_sent']})")
            
            if self.metrics['landmarks_extracted'] < 30:
                print(f"   [FAIL] Need at least 30 landmarks in 60s, got {self.metrics['landmarks_extracted']}")
                passed = False
            else:
                print(f"   [PASS] Sufficient landmark extractions")
        
        # 2. Frame data errors
        frame_errors = sum(1 for e in self.metrics['errors'] if 'Frame data too short' in str(e))
        print(f"2. Frame data errors: {frame_errors}")
        if frame_errors > 0:
            print(f"   [FAIL] Found {frame_errors} 'Frame data too short' errors")
            passed = False
        else:
            print(f"   [PASS] No frame data errors")
        
        # 3. Frozen frames
        print(f"3. Frozen frame incidents: {self.metrics['frozen_frame_count']}")
        if self.metrics['frozen_frame_count'] > 5:
            print(f"   [FAIL] Too many frozen frame incidents")
            passed = False
        else:
            print(f"   [PASS] Acceptable frozen frame count")
        
        # 4. Latency
        if self.metrics['latencies']:
            median_latency = np.median(self.metrics['latencies'])
            p95_latency = np.percentile(self.metrics['latencies'], 95)
            print(f"4. Latency: median={median_latency:.1f}ms, p95={p95_latency:.1f}ms")
            
            if median_latency > 600:
                print(f"   [WARN] Median latency > 600ms target")
            else:
                print(f"   [PASS] Latency within target")
        else:
            print(f"4. Latency: [SKIP] No latency data")
        
        # 5. Processing rate
        if self.metrics['frames_sent'] > 0:
            processing_rate = (self.metrics['frames_processed'] / self.metrics['frames_sent']) * 100
            print(f"5. Processing rate: {processing_rate:.1f}%")
            
            if processing_rate < 80:
                print(f"   [WARN] Processing rate below 80%")
            else:
                print(f"   [PASS] Good processing rate")
        
        return passed
    
    def print_report(self, overall_pass):
        """Print final report"""
        print("\n" + "="*70)
        print("FINAL TEST REPORT")
        print("="*70)
        
        status = "PASSED" if overall_pass else "FAILED"
        print(f"\nSTATUS: {status}\n")
        
        print("METRICS COLLECTED:")
        print(f"  Total frames sent: {self.metrics['frames_sent']}")
        print(f"  Total frames processed: {self.metrics['frames_processed']}")
        print(f"  Total landmark extractions: {self.metrics['landmarks_extracted']}")
        print(f"  No hands count: {self.metrics['no_hands_count']}")
        print(f"  Predictions received: {self.metrics['predictions_received']}")
        print(f"  Frozen frame incidents: {self.metrics['frozen_frame_count']}")
        
        if self.metrics['latencies']:
            print(f"  Median latency: {np.median(self.metrics['latencies']):.1f}ms")
            print(f"  Max latency: {np.max(self.metrics['latencies']):.1f}ms")
            print(f"  Min latency: {np.min(self.metrics['latencies']):.1f}ms")
        
        if self.metrics['errors']:
            print(f"\nERRORS ENCOUNTERED: {len(self.metrics['errors'])}")
            for i, error in enumerate(self.metrics['errors'][:5], 1):
                print(f"  {i}. {error}")
        
        if self.metrics['frames_sent'] > 0:
            detection_rate = (self.metrics['landmarks_extracted'] / self.metrics['frames_sent']) * 100
            print(f"\nDETECTION RATE: {detection_rate:.1f}%")
        
        print("\nCOMMANDS TO RUN:")
        print("  Terminal 1: cd backend && python app_v2.py")
        print("  Terminal 2: cd frontend && python -m http.server 3000")
        print("  Browser: http://localhost:3000/index_v2.html (Ctrl+Shift+R)")
        
        if overall_pass:
            print("\n3-STEP DEMO CHECKLIST:")
            print("  1. Hard refresh browser (Ctrl+Shift+R)")
            print("  2. Click 'Start Recognition'")
            print("  3. Show hand signs - predictions appear in ~1s")
        else:
            print("\nIMMEDIATE MITIGATION:")
            if self.metrics['landmarks_extracted'] < 30:
                print("  - Lower MediaPipe confidence thresholds further")
                print("  - Increase frame processing timeout")
            if self.metrics['frozen_frame_count'] > 5:
                print("  - Adjust frozen frame detection threshold")
        
        print("="*70)
    
    def cleanup(self):
        """Cleanup connections"""
        try:
            if self.sio.connected:
                self.sio.disconnect()
        except:
            pass
    
    def run_full_test(self):
        """Run all tests"""
        overall_pass = True
        
        try:
            # Test A: Startup
            if not self.test_startup():
                overall_pass = False
                return overall_pass
            
            # Test B: Streaming
            if not self.test_streaming(duration=60):
                overall_pass = False
            
            # Test D: Single hand
            if not self.test_single_hand():
                overall_pass = False
            
            # Test C: Analysis
            if not self.analyze_results():
                overall_pass = False
            
        except KeyboardInterrupt:
            print("\n\n[INTERRUPTED] Test stopped by user")
            overall_pass = False
        except Exception as e:
            print(f"\n\n[ERROR] Test failed: {e}")
            import traceback
            traceback.print_exc()
            overall_pass = False
        finally:
            self.cleanup()
        
        self.print_report(overall_pass)
        return overall_pass

def main():
    print("\n" + "="*70)
    print("ISHARAPAL DEMO READINESS TEST")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNOTE: Backend must be running on http://localhost:5000")
    print("="*70)
    
    test = DemoReadinessTest()
    passed = test.run_full_test()
    
    sys.exit(0 if passed else 1)

if __name__ == '__main__':
    main()

