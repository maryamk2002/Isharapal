#!/usr/bin/env python3
"""
Feedback System for PSL Recognition
Stores user feedback to improve model accuracy over time
"""

import sqlite3
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class FeedbackDatabase:
    """
    SQLite database for storing user feedback on predictions.
    
    Schema:
    - feedback: Main feedback table
    - statistics: Aggregated statistics per sign
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize feedback database.
        
        Args:
            db_path: Path to SQLite database file (default: backend/data/feedback.db)
        """
        if db_path is None:
            db_path = Path(__file__).parent / "data" / "feedback.db"
        
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        logger.info(f"Feedback database initialized: {self.db_path}")
    
    def _init_database(self):
        """Create database tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Main feedback table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    predicted_label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    is_correct BOOLEAN NOT NULL,
                    timestamp REAL NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Index for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_predicted_label 
                ON feedback(predicted_label)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON feedback(timestamp)
            ''')
            
            # Statistics table (aggregated per sign)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS statistics (
                    sign_label TEXT PRIMARY KEY,
                    total_predictions INTEGER DEFAULT 0,
                    correct_predictions INTEGER DEFAULT 0,
                    incorrect_predictions INTEGER DEFAULT 0,
                    avg_confidence REAL DEFAULT 0.0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def add_feedback(
        self,
        predicted_label: str,
        confidence: float,
        is_correct: bool,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add feedback entry to database.
        
        Args:
            predicted_label: The label that was predicted
            confidence: Confidence score of the prediction
            is_correct: Whether the prediction was correct
            session_id: Optional session identifier
            metadata: Optional metadata dictionary
        
        Returns:
            ID of the inserted feedback entry
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                metadata_json = json.dumps(metadata) if metadata else None
                timestamp = datetime.now().timestamp()
                
                cursor.execute('''
                    INSERT INTO feedback 
                    (session_id, predicted_label, confidence, is_correct, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    session_id,
                    predicted_label,
                    confidence,
                    int(is_correct),
                    timestamp,
                    metadata_json
                ))
                
                feedback_id = cursor.lastrowid
                conn.commit()
                
                # Update statistics
                self._update_statistics(predicted_label, confidence, is_correct)
                
                logger.info(
                    f"Feedback added: {predicted_label} = {'Correct' if is_correct else 'Incorrect'} "
                    f"(confidence: {confidence:.3f})"
                )
                
                return feedback_id
        
        except Exception as e:
            logger.error(f"Failed to add feedback: {e}")
            raise
    
    def _update_statistics(self, label: str, confidence: float, is_correct: bool):
        """Update aggregated statistics for a sign."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get current statistics
                cursor.execute('''
                    SELECT total_predictions, correct_predictions, 
                           incorrect_predictions, avg_confidence
                    FROM statistics
                    WHERE sign_label = ?
                ''', (label,))
                
                row = cursor.fetchone()
                
                if row:
                    # Update existing statistics
                    total = row['total_predictions'] + 1
                    correct = row['correct_predictions'] + (1 if is_correct else 0)
                    incorrect = row['incorrect_predictions'] + (0 if is_correct else 1)
                    
                    # Update average confidence (exponential moving average)
                    old_avg = row['avg_confidence']
                    new_avg = old_avg * 0.9 + confidence * 0.1
                    
                    cursor.execute('''
                        UPDATE statistics
                        SET total_predictions = ?,
                            correct_predictions = ?,
                            incorrect_predictions = ?,
                            avg_confidence = ?,
                            last_updated = CURRENT_TIMESTAMP
                        WHERE sign_label = ?
                    ''', (total, correct, incorrect, new_avg, label))
                
                else:
                    # Insert new statistics
                    cursor.execute('''
                        INSERT INTO statistics
                        (sign_label, total_predictions, correct_predictions, 
                         incorrect_predictions, avg_confidence)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        label,
                        1,
                        1 if is_correct else 0,
                        0 if is_correct else 1,
                        confidence
                    ))
                
                conn.commit()
        
        except Exception as e:
            logger.error(f"Failed to update statistics: {e}")
    
    def get_statistics(self, label: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get statistics for signs.
        
        Args:
            label: Optional specific sign label (if None, returns all)
        
        Returns:
            List of statistics dictionaries
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if label:
                    cursor.execute('''
                        SELECT * FROM statistics
                        WHERE sign_label = ?
                    ''', (label,))
                else:
                    cursor.execute('''
                        SELECT * FROM statistics
                        ORDER BY total_predictions DESC
                    ''')
                
                rows = cursor.fetchall()
                
                statistics = []
                for row in rows:
                    accuracy = (
                        row['correct_predictions'] / row['total_predictions']
                        if row['total_predictions'] > 0 else 0.0
                    )
                    
                    statistics.append({
                        'sign_label': row['sign_label'],
                        'total_predictions': row['total_predictions'],
                        'correct_predictions': row['correct_predictions'],
                        'incorrect_predictions': row['incorrect_predictions'],
                        'accuracy': accuracy,
                        'avg_confidence': row['avg_confidence'],
                        'last_updated': row['last_updated']
                    })
                
                return statistics
        
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return []
    
    def get_feedback_history(
        self,
        limit: int = 100,
        label: Optional[str] = None,
        is_correct: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Get feedback history with optional filtering.
        
        Args:
            limit: Maximum number of entries to return
            label: Optional filter by label
            is_correct: Optional filter by correctness
        
        Returns:
            List of feedback entries
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                query = 'SELECT * FROM feedback WHERE 1=1'
                params = []
                
                if label:
                    query += ' AND predicted_label = ?'
                    params.append(label)
                
                if is_correct is not None:
                    query += ' AND is_correct = ?'
                    params.append(int(is_correct))
                
                query += ' ORDER BY created_at DESC LIMIT ?'
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                feedback = []
                for row in rows:
                    metadata = json.loads(row['metadata']) if row['metadata'] else {}
                    
                    feedback.append({
                        'id': row['id'],
                        'session_id': row['session_id'],
                        'predicted_label': row['predicted_label'],
                        'confidence': row['confidence'],
                        'is_correct': bool(row['is_correct']),
                        'timestamp': row['timestamp'],
                        'metadata': metadata,
                        'created_at': row['created_at']
                    })
                
                return feedback
        
        except Exception as e:
            logger.error(f"Failed to get feedback history: {e}")
            return []
    
    def get_confused_signs(self, min_errors: int = 5) -> List[Tuple[str, int, float]]:
        """
        Get signs that are frequently predicted incorrectly.
        
        Args:
            min_errors: Minimum number of errors to include
        
        Returns:
            List of (sign_label, error_count, error_rate) tuples
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT 
                        sign_label,
                        incorrect_predictions,
                        CAST(incorrect_predictions AS FLOAT) / total_predictions as error_rate
                    FROM statistics
                    WHERE incorrect_predictions >= ?
                    ORDER BY error_rate DESC, incorrect_predictions DESC
                ''', (min_errors,))
                
                rows = cursor.fetchall()
                
                return [
                    (row['sign_label'], row['incorrect_predictions'], row['error_rate'])
                    for row in rows
                ]
        
        except Exception as e:
            logger.error(f"Failed to get confused signs: {e}")
            return []
    
    def export_feedback(self, output_path: Path, format: str = 'json'):
        """
        Export feedback data for analysis or retraining.
        
        Args:
            output_path: Path to output file
            format: Export format ('json' or 'csv')
        """
        try:
            feedback = self.get_feedback_history(limit=10000)
            
            if format == 'json':
                with open(output_path, 'w') as f:
                    json.dump(feedback, f, indent=2)
            
            elif format == 'csv':
                import csv
                
                with open(output_path, 'w', newline='') as f:
                    if feedback:
                        writer = csv.DictWriter(f, fieldnames=feedback[0].keys())
                        writer.writeheader()
                        writer.writerows(feedback)
            
            logger.info(f"Feedback exported to {output_path}")
        
        except Exception as e:
            logger.error(f"Failed to export feedback: {e}")
            raise
    
    def get_summary(self) -> Dict[str, Any]:
        """Get overall feedback summary."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Total feedback count
                cursor.execute('SELECT COUNT(*) as total FROM feedback')
                total = cursor.fetchone()['total']
                
                # Correct/incorrect counts
                cursor.execute('''
                    SELECT 
                        SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct,
                        SUM(CASE WHEN is_correct = 0 THEN 1 ELSE 0 END) as incorrect
                    FROM feedback
                ''')
                row = cursor.fetchone()
                correct = row['correct'] or 0
                incorrect = row['incorrect'] or 0
                
                # Overall accuracy
                accuracy = correct / total if total > 0 else 0.0
                
                # Number of unique signs with feedback
                cursor.execute('SELECT COUNT(DISTINCT sign_label) as signs FROM statistics')
                signs = cursor.fetchone()['signs']
                
                return {
                    'total_feedback': total,
                    'correct': correct,
                    'incorrect': incorrect,
                    'overall_accuracy': accuracy,
                    'signs_with_feedback': signs
                }
        
        except Exception as e:
            logger.error(f"Failed to get summary: {e}")
            return {}


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("FEEDBACK SYSTEM TEST")
    print("=" * 60)
    
    # Create database
    db = FeedbackDatabase()
    
    # Add some test feedback
    print("\n Adding test feedback...")
    db.add_feedback("Alifmad", 0.95, True, "test_session_1")
    db.add_feedback("Jeem", 0.87, True, "test_session_1")
    db.add_feedback("Aray", 0.72, False, "test_session_1")  # Incorrect
    db.add_feedback("2-Hay", 0.68, False, "test_session_1")  # Incorrect
    db.add_feedback("Alifmad", 0.91, True, "test_session_2")
    
    # Get summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("-" * 60)
    summary = db.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Get statistics
    print("\n" + "=" * 60)
    print("STATISTICS PER SIGN:")
    print("-" * 60)
    stats = db.get_statistics()
    for stat in stats:
        print(f"\n {stat['sign_label']}:")
        print(f"    Total: {stat['total_predictions']}")
        print(f"    Correct: {stat['correct_predictions']}")
        print(f"    Incorrect: {stat['incorrect_predictions']}")
        print(f"    Accuracy: {stat['accuracy']:.2%}")
        print(f"    Avg Confidence: {stat['avg_confidence']:.3f}")
    
    # Get confused signs
    print("\n" + "=" * 60)
    print("SIGNS WITH ERRORS:")
    print("-" * 60)
    confused = db.get_confused_signs(min_errors=1)
    for sign, errors, rate in confused:
        print(f"  {sign}: {errors} errors ({rate:.1%} error rate)")
    
    print("\n" + "=" * 60)
    print("✓ Feedback system test complete!")
    print("=" * 60)

