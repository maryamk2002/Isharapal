#!/usr/bin/env python3
"""
Analyze feedback database to compute optimal per-sign confidence thresholds.
Generates sign_thresholds.json configuration file.
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from feedback_system import FeedbackDatabase
from config_v2 import ALL_URDU_SIGNS
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_per_sign_accuracy() -> dict:
    """
    Query feedback DB, compute per-sign accuracy and recommended thresholds.
    
    Logic:
    - High accuracy signs (>90%) → raise threshold to 0.60-0.65
    - Low accuracy signs (<80%) → lower threshold to 0.45-0.55
    - Medium accuracy → use default 0.55
    - No feedback yet → use default 0.55
    
    Returns:
        {
            'Alif': {'accuracy': 0.95, 'samples': 50, 'recommended_threshold': 0.58},
            'Jeem': {'accuracy': 0.78, 'samples': 12, 'recommended_threshold': 0.48},
            ...
        }
    """
    try:
        db = FeedbackDatabase()
    except Exception as e:
        logger.warning(f"Failed to connect to feedback database: {e}")
        logger.info("Using default thresholds for all signs")
        # Return defaults for all signs
        return {label: {
            'accuracy': None,
            'samples': 0,
            'recommended_threshold': 0.55
        } for label in ALL_URDU_SIGNS}
    
    sign_stats = {}
    
    for label in ALL_URDU_SIGNS:
        try:
            # Get accuracy stats from feedback DB
            accuracy_data = db.get_accuracy_by_sign(label)
            
            if accuracy_data is None:
                # No feedback yet, use default
                recommended = 0.55
                accuracy = None
                samples = 0
            else:
                accuracy = accuracy_data.get('accuracy', 0.0)
                samples = accuracy_data.get('total', 0)
                
                if samples < 5:
                    # Not enough samples, use default
                    recommended = 0.55
                elif accuracy > 0.90:
                    # Very accurate sign, can raise threshold
                    recommended = 0.60 + (accuracy - 0.90) * 0.5  # 0.60-0.65
                elif accuracy < 0.80:
                    # Problematic sign, lower threshold
                    recommended = 0.55 - (0.80 - accuracy) * 0.3  # 0.45-0.55
                else:
                    # Normal sign (80-90% accuracy)
                    recommended = 0.55
            
            sign_stats[label] = {
                'accuracy': accuracy,
                'samples': samples,
                'recommended_threshold': round(recommended, 2)
            }
            
        except Exception as e:
            logger.warning(f"Failed to analyze {label}: {e}")
            sign_stats[label] = {
                'accuracy': None,
                'samples': 0,
                'recommended_threshold': 0.55
            }
    
    return sign_stats


def generate_threshold_config(output_path: Path) -> None:
    """
    Generate sign_thresholds.json config file.
    
    Args:
        output_path: Path to save JSON file
    """
    logger.info("Analyzing sign performance from feedback database...")
    stats = analyze_per_sign_accuracy()
    
    # Extract just thresholds for config
    thresholds = {
        label: data['recommended_threshold']
        for label, data in stats.items()
    }
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_path, 'w') as f:
        json.dump(thresholds, f, indent=2)
    
    logger.info(f"\n✓ Generated: {output_path}")
    logger.info(f"  Total signs: {len(thresholds)}")
    logger.info(f"  Threshold range: {min(thresholds.values()):.2f} - {max(thresholds.values()):.2f}")
    
    # Print signs with non-default thresholds
    non_default = {k: v for k, v in thresholds.items() if v != 0.55}
    if non_default:
        logger.info(f"\n  Signs with adjusted thresholds ({len(non_default)}):")
        for sign, threshold in sorted(non_default.items(), key=lambda x: x[1]):
            accuracy = stats[sign]['accuracy']
            samples = stats[sign]['samples']
            acc_str = f"{accuracy:.1%}" if accuracy is not None else "N/A"
            logger.info(f"    {sign:12s} → {threshold:.2f} (accuracy: {acc_str}, n={samples})")
    else:
        logger.info("  All signs using default threshold (0.55) - no feedback data yet")


def main():
    """Generate sign thresholds configuration."""
    # Output path
    config_dir = Path(__file__).parent.parent / 'config'
    output_path = config_dir / 'sign_thresholds.json'
    
    # Generate config
    generate_threshold_config(output_path)
    
    logger.info("\n✓ Configuration generated successfully!")
    logger.info(f"  The predictor will load thresholds from: {output_path}")


if __name__ == '__main__':
    main()

