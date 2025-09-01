"""
Shared Run ID Manager for consistent ID generation across all logging tables.
"""

import sqlite3
from pathlib import Path


class RunIDManager:
    """
    Manages shared run IDs across all logging tables to ensure consistent incrementing.
    """
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            project_root = Path(__file__).parent.parent
            db_path = project_root / "logs" / "prompts" / "prompt_logs.db"
        
        self.db_path = str(db_path)
        self._init_run_id_table()
    
    def _init_run_id_table(self):
        """Initialize the run_id_counter table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create run_id_counter table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS run_id_counter (
                    id INTEGER PRIMARY KEY,
                    last_run_id INTEGER NOT NULL DEFAULT 0
                )
            """)
            
            # Initialize with current max ID from existing tables if empty
            cursor.execute("SELECT COUNT(*) FROM run_id_counter")
            if cursor.fetchone()[0] == 0:
                # Find max ID across all tables
                cursor.execute("""
                    SELECT COALESCE(MAX(max_id), 0) as current_max FROM (
                        SELECT MAX(id) as max_id FROM prompt_logs
                        UNION ALL
                        SELECT MAX(id) as max_id FROM tabular_preprocessing_logs
                        UNION ALL
                        SELECT MAX(id) as max_id FROM preprocessing_logs
                    )
                """)
                current_max = cursor.fetchone()[0] or 0
                
                cursor.execute("""
                    INSERT INTO run_id_counter (id, last_run_id) VALUES (1, ?)
                """, (current_max,))
                
                print(f"🆔 Initialized run ID counter starting from {current_max}")
    
    def get_next_run_id(self) -> int:
        """
        Get the next available run ID and increment the counter atomically.
        
        Returns:
            int: The next run ID to use
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Atomically increment and get the next ID
            cursor.execute("""
                UPDATE run_id_counter 
                SET last_run_id = last_run_id + 1 
                WHERE id = 1
            """)
            
            cursor.execute("""
                SELECT last_run_id FROM run_id_counter WHERE id = 1
            """)
            
            next_id = cursor.fetchone()[0]
            return next_id
    
    def get_current_run_id(self) -> int:
        """Get the current run ID without incrementing."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT last_run_id FROM run_id_counter WHERE id = 1")
            result = cursor.fetchone()
            return result[0] if result else 0


# Usage example:
# run_manager = RunIDManager()
# run_id = run_manager.get_next_run_id()  # Returns 65, 66, 67, etc.
# 
# # Use the same run_id for all logging in this run:
# tabular_logger.log_with_run_id(run_id, ...)
# prompt_logger.log_with_run_id(run_id, ...)
