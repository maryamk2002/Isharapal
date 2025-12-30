#!/usr/bin/env python3
"""Quick script to check SQLite database contents"""
import sqlite3
from pathlib import Path

db_path = Path('backend/data/feedback.db')
print(f"Database path: {db_path.absolute()}")
print(f"Database exists: {db_path.exists()}")

if db_path.exists():
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Show tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [t[0] for t in cursor.fetchall()]
    print(f"\nTables: {tables}")
    
    # Show feedback count
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  {table}: {count} rows")
    
    # Show sample feedback
    if 'feedback' in tables:
        cursor.execute("SELECT * FROM feedback LIMIT 5")
        rows = cursor.fetchall()
        print(f"\nSample feedback entries:")
        for row in rows:
            print(f"  {row}")
    
    conn.close()
else:
    print("Database not found!")

