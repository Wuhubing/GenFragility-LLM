import sqlite3
from datetime import datetime

class StateDB:
    def __init__(self, db_path="logs/pipeline_state.sqlite"):
        self.db_path = db_path
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS run_state (
                    run_id TEXT PRIMARY KEY,
                    target_id TEXT,
                    seed INTEGER,
                    anchor_config TEXT,
                    status TEXT,
                    failure_reason TEXT,
                    retry_count INTEGER DEFAULT 0,
                    started_at DATETIME,
                    completed_at DATETIME,
                    output_path TEXT
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_status ON run_state(status)')

    def is_completed(self, run_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status FROM run_state WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            return row is not None and row[0] == 'completed'

    def mark_started(self, run_id: str, target_id: str, seed: int, anchor_config: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO run_state 
                (run_id, target_id, seed, anchor_config, status, started_at) 
                VALUES (?, ?, ?, ?, 'running', ?)
            ''', (run_id, target_id, seed, anchor_config, datetime.now()))

    def mark_completed(self, run_id: str, output_path: str = ""):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE run_state 
                SET status = 'completed', completed_at = ?, output_path = ?
                WHERE run_id = ?
            ''', (datetime.now(), output_path, run_id))

    def mark_failed(self, run_id: str, reason: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE run_state 
                SET status = 'failed', failure_reason = ?, retry_count = retry_count + 1
                WHERE run_id = ?
            ''', (str(reason), run_id))

    def get_failed_runs(self, max_retries=3):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT run_id FROM run_state WHERE status = 'failed' AND retry_count < ?", (max_retries,))
            return [row[0] for row in cursor.fetchall()]

if __name__ == "__main__":
    db = StateDB("logs/test_state.sqlite")
    db.mark_started("test_run", "T01", 42, "baseline")
    assert not db.is_completed("test_run")
    db.mark_completed("test_run", "out.jsonl")
    assert db.is_completed("test_run")
    print("StateDB tests passed.")
