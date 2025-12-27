#!/usr/bin/env python3
"""
Automated performance test suite for PSL recognition.
Tests 5 representative signs for speed, accuracy, and stability.
"""

import time
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any
import sys
import logging

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from inference.predictor_v2 import RealTimePredictorV2
from models.model_manager import ModelManager
from config_v2 import inference_config_v2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test signs (diverse difficulty: 2 rare, 2 common, 1 medium)
TEST_SIGNS = ['Jeem', 'Aray', 'Gaaf', 'Alif', 'Bay']

class PerformanceTestSuite:
    """Automated performance testing for PSL recognition."""
    
    def __init__(self):
        self.test_data_dir = Path(__file__).parent / "data" / "test_sequences"
        self.results_dir = Path(__file__).parent / "test_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model and predictor
        self.predictor = None
        self._init_model()
    
    def _init_model(self):
        """Initialize model and predictor."""
        logger.info("Initializing model...")
        models_dir = Path(__file__).parent / "saved_models" / "v2"
        model_manager = ModelManager(models_dir=models_dir)
        
        import torch
        device = torch.device('cpu')
        
        best_model = model_manager.get_best_model()
        if not best_model:
            raise RuntimeError("No model found!")
        
        model, config, labels = model_manager.load_model(best_model, device=device)
        
        if model is None:
            raise RuntimeError("Failed to load model")
        
        # Create predictor
        self.predictor = RealTimePredictorV2(
            model=model,
            config=config,
            labels=labels,
            sliding_window_size=inference_config_v2.SLIDING_WINDOW_SIZE,
            min_prediction_frames=inference_config_v2.MIN_FRAMES_FOR_PREDICTION,
            stability_votes=5,
            stability_threshold=3,
            min_confidence=0.55,
            reset_on_no_hands=True
        )
        
        logger.info(f"✓ Model loaded: {len(labels)} classes")
    
    def load_test_sequences(self) -> Dict[str, np.ndarray]:
        """
        Load pre-recorded .npy sequences for testing.
        
        Returns:
            Dict mapping sign_name -> sequence array
        """
        sequences = {}
        
        for sign in TEST_SIGNS:
            sign_dir = Path(__file__).parent / "data" / "features_temporal" / sign
            
            if not sign_dir.exists():
                logger.warning(f"Sign directory not found: {sign_dir}")
                continue
            
            # Get first .npy file from this sign
            npy_files = list(sign_dir.glob("*.npy"))
            
            if not npy_files:
                logger.warning(f"No .npy files found for {sign}")
                continue
            
            # Load first sequence
            sequence_path = npy_files[0]
            try:
                sequence = np.load(sequence_path)
                sequences[sign] = sequence
                logger.info(f"✓ Loaded {sign}: shape {sequence.shape}")
            except Exception as e:
                logger.error(f"Failed to load {sign}: {e}")
        
        return sequences
    
    def copy_test_sequences(self):
        """Copy test sequences to test_sequences folder for consistency."""
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
        
        for sign in TEST_SIGNS:
            source_dir = Path(__file__).parent / "data" / "features_temporal" / sign
            if not source_dir.exists():
                logger.warning(f"Source not found: {source_dir}")
                continue
            
            npy_files = list(source_dir.glob("*.npy"))
            if not npy_files:
                logger.warning(f"No sequences for {sign}")
                continue
            
            # Copy first file
            source_file = npy_files[0]
            dest_file = self.test_data_dir / f"{sign}.npy"
            
            if not dest_file.exists():
                import shutil
                shutil.copy(source_file, dest_file)
                logger.info(f"✓ Copied {sign} test sequence")
    
    def run_prediction_test(self, sign_name: str, sequence: np.ndarray) -> Dict[str, Any]:
        """
        Run sequence through predictor, measure performance.
        
        Args:
            sign_name: Expected sign label
            sequence: Keypoint sequence (frames, 189)
        
        Returns:
            Dict with metrics: time_to_first, accuracy, confidence, etc.
        """
        # Clear predictor
        self.predictor.clear_buffer()
        
        # Feed frames one by one
        start_time = time.time()
        first_prediction_time = None
        predictions = []
        
        for i, frame in enumerate(sequence):
            self.predictor.add_frame(frame)
            
            result = self.predictor.predict_current(return_all_predictions=True)
            
            if result and result.get('ready') and result.get('is_stable'):
                if first_prediction_time is None:
                    first_prediction_time = time.time() - start_time
                
                predictions.append({
                    'frame': i,
                    'label': result.get('stable_prediction'),
                    'confidence': result.get('stable_confidence'),
                    'is_new': result.get('is_new', False)
                })
        
        total_time = time.time() - start_time
        
        # Analyze results
        correct = False
        final_confidence = 0.0
        
        if predictions:
            # Check if any stable prediction matches expected
            for pred in predictions:
                if pred['label'] == sign_name:
                    correct = True
                    final_confidence = pred['confidence']
                    break
        
        return {
            'sign': sign_name,
            'correct': correct,
            'time_to_first_prediction': first_prediction_time,
            'total_time': total_time,
            'confidence': final_confidence,
            'num_predictions': len(predictions),
            'predictions': predictions,
            'frames_processed': len(sequence)
        }
    
    def run_all_tests(self) -> List[Dict]:
        """Run tests on all test signs."""
        logger.info("=" * 60)
        logger.info("RUNNING PERFORMANCE TEST SUITE")
        logger.info("=" * 60)
        
        # Load sequences
        sequences = self.load_test_sequences()
        
        if not sequences:
            logger.error("No test sequences found!")
            return []
        
        logger.info(f"\nTesting {len(sequences)} signs: {list(sequences.keys())}")
        logger.info("")
        
        # Run tests
        results = []
        
        for sign_name, sequence in sequences.items():
            logger.info(f"Testing: {sign_name}...")
            result = self.run_prediction_test(sign_name, sequence)
            results.append(result)
            
            # Print result
            status = "✓ PASS" if result['correct'] else "✗ FAIL"
            time_str = f"{result['time_to_first_prediction']:.2f}s" if result['time_to_first_prediction'] else "N/A"
            conf_str = f"{result['confidence']:.2%}" if result['correct'] else "N/A"
            
            logger.info(f"  {status} | Time: {time_str} | Confidence: {conf_str}")
        
        return results
    
    def generate_report(self, results: List[Dict]) -> None:
        """Print formatted report with metrics."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("=" * 60)
        
        if not results:
            logger.info("No results to report")
            return
        
        # Calculate metrics
        total = len(results)
        correct = sum(1 for r in results if r['correct'])
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        times = [r['time_to_first_prediction'] for r in results if r['time_to_first_prediction']]
        avg_time = sum(times) / len(times) if times else 0
        min_time = min(times) if times else 0
        max_time = max(times) if times else 0
        
        confidences = [r['confidence'] for r in results if r['correct']]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        logger.info(f"\nAccuracy: {correct}/{total} ({accuracy:.1f}%)")
        logger.info(f"First Prediction Time:")
        logger.info(f"  Average: {avg_time:.2f}s")
        logger.info(f"  Min: {min_time:.2f}s")
        logger.info(f"  Max: {max_time:.2f}s")
        logger.info(f"Average Confidence: {avg_confidence:.2%}")
        
        logger.info("\nPer-Sign Breakdown:")
        for r in results:
            status = "✓" if r['correct'] else "✗"
            time_str = f"{r['time_to_first_prediction']:.2f}s" if r['time_to_first_prediction'] else "N/A"
            conf_str = f"{r['confidence']:.2%}" if r['correct'] else "N/A"
            logger.info(f"  {status} {r['sign']:12s} | {time_str:8s} | {conf_str}")
        
        logger.info("=" * 60)
    
    def save_results(self, results: List[Dict], label: str = "test") -> Path:
        """
        Save results to JSON file.
        
        Args:
            results: Test results
            label: Label for filename (e.g., 'baseline', 'day2')
        
        Returns:
            Path to saved file
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"performance_{label}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Calculate summary metrics
        total = len(results)
        correct = sum(1 for r in results if r['correct'])
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        times = [r['time_to_first_prediction'] for r in results if r['time_to_first_prediction']]
        avg_time = sum(times) / len(times) if times else 0
        
        confidences = [r['confidence'] for r in results if r['correct']]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        data = {
            'timestamp': timestamp,
            'label': label,
            'summary': {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'avg_time_to_first_prediction': avg_time,
                'avg_confidence': avg_confidence
            },
            'results': results
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"\n✓ Results saved: {filepath}")
        return filepath


def main():
    """Run test suite."""
    suite = PerformanceTestSuite()
    
    # Copy test sequences (first time only)
    suite.copy_test_sequences()
    
    # Run tests
    results = suite.run_all_tests()
    
    # Generate report
    suite.generate_report(results)
    
    # Save results as baseline
    suite.save_results(results, label="baseline")


if __name__ == '__main__':
    main()

