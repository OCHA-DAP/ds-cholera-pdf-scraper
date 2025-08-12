"""
Prompt Logging System for LLM workflows.
Captures and organizes prompts, responses, and metrics for reproducibility and debugging.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class PromptLogger:
    """
    Logs LLM interactions with prompts, responses, and performance metrics.
    """

    def __init__(self, log_dir: str = None, use_sqlite: bool = True):
        """
        Initialize PromptLogger.

        Args:
            log_dir: Directory for log storage
            use_sqlite: Whether to use SQLite DB (True) or JSONL files (False)
        """
        if log_dir is None:
            # Default to logs directory in project root
            project_root = Path(__file__).parent.parent
            log_dir = project_root / "logs" / "prompts"

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.use_sqlite = use_sqlite

        if self.use_sqlite:
            self.db_path = self.log_dir / "prompt_logs.db"
            self._init_database()
        else:
            self.jsonl_path = self.log_dir / "prompt_logs.jsonl"

    def _init_database(self):
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create logs table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS prompt_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    prompt_type TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_parameters TEXT NOT NULL,
                    system_prompt TEXT NOT NULL,
                    user_prompt TEXT NOT NULL,
                    raw_response TEXT NOT NULL,
                    parsed_success BOOLEAN NOT NULL,
                    records_extracted INTEGER,
                    parsing_errors TEXT,
                    execution_time_seconds REAL,
                    prompt_metadata TEXT,
                    custom_metrics TEXT
                )
            """
            )

            # Create index for efficient querying
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_prompt_type_version 
                ON prompt_logs(prompt_type, prompt_version)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON prompt_logs(timestamp)
            """
            )

            conn.commit()
            print(f"✅ SQLite database initialized: {self.db_path}")

    def log_llm_call(
        self,
        prompt_metadata: Dict[str, Any],
        model_name: str,
        model_parameters: Dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        raw_response: str,
        parsed_success: bool,
        records_extracted: Optional[int] = None,
        parsing_errors: Optional[str] = None,
        execution_time_seconds: Optional[float] = None,
        custom_metrics: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Log a complete LLM interaction.

        Args:
            prompt_metadata: Metadata from PromptManager
            model_name: Name and version of the model used
            model_parameters: All model parameters (temperature, max_tokens, etc.)
            system_prompt: Complete system prompt sent
            user_prompt: Complete user prompt sent
            raw_response: Raw response from model
            parsed_success: Whether parsing was successful
            records_extracted: Number of records successfully extracted
            parsing_errors: Any parsing error messages
            execution_time_seconds: Time taken for the call
            custom_metrics: Additional metrics specific to the task

        Returns:
            str: Log entry ID or timestamp
        """
        timestamp = datetime.now().isoformat()

        log_entry = {
            "timestamp": timestamp,
            "prompt_type": prompt_metadata.get("prompt_type"),
            "prompt_version": prompt_metadata.get("version"),
            "model_name": model_name,
            "model_parameters": model_parameters,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": raw_response,
            "parsed_success": parsed_success,
            "records_extracted": records_extracted,
            "parsing_errors": parsing_errors,
            "execution_time_seconds": execution_time_seconds,
            "prompt_metadata": prompt_metadata,
            "custom_metrics": custom_metrics or {},
        }

        if self.use_sqlite:
            return self._log_to_sqlite(log_entry)
        else:
            return self._log_to_jsonl(log_entry)

    def _log_to_sqlite(self, log_entry: Dict[str, Any]) -> str:
        """Log entry to SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO prompt_logs (
                    timestamp, prompt_type, prompt_version, model_name, 
                    model_parameters, system_prompt, user_prompt, raw_response,
                    parsed_success, records_extracted, parsing_errors, 
                    execution_time_seconds, prompt_metadata, custom_metrics
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    log_entry["timestamp"],
                    log_entry["prompt_type"],
                    log_entry["prompt_version"],
                    log_entry["model_name"],
                    json.dumps(log_entry["model_parameters"]),
                    log_entry["system_prompt"],
                    log_entry["user_prompt"],
                    log_entry["raw_response"],
                    log_entry["parsed_success"],
                    log_entry["records_extracted"],
                    log_entry["parsing_errors"],
                    log_entry["execution_time_seconds"],
                    json.dumps(log_entry["prompt_metadata"]),
                    json.dumps(log_entry["custom_metrics"]),
                ),
            )

            log_id = cursor.lastrowid
            conn.commit()

            print(
                f"📝 Logged LLM call (ID: {log_id}) - {log_entry['prompt_type']} v{log_entry['prompt_version']}"
            )
            return str(log_id)

    def _log_to_jsonl(self, log_entry: Dict[str, Any]) -> str:
        """Log entry to JSONL file."""
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        print(
            f"📝 Logged LLM call to JSONL - {log_entry['prompt_type']} v{log_entry['prompt_version']}"
        )
        return log_entry["timestamp"]

    def query_logs(
        self,
        prompt_type: Optional[str] = None,
        prompt_version: Optional[str] = None,
        model_name: Optional[str] = None,
        parsed_success: Optional[bool] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Query logged entries with filters.

        Returns:
            List of log entries matching the criteria
        """
        if not self.use_sqlite:
            return self._query_jsonl(
                prompt_type, prompt_version, model_name, parsed_success, limit
            )

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            cursor = conn.cursor()

            # Build query with filters
            where_clauses = []
            params = []

            if prompt_type:
                where_clauses.append("prompt_type = ?")
                params.append(prompt_type)

            if prompt_version:
                where_clauses.append("prompt_version = ?")
                params.append(prompt_version)

            if model_name:
                where_clauses.append("model_name = ?")
                params.append(model_name)

            if parsed_success is not None:
                where_clauses.append("parsed_success = ?")
                params.append(parsed_success)

            where_sql = ""
            if where_clauses:
                where_sql = "WHERE " + " AND ".join(where_clauses)

            query = f"""
                SELECT * FROM prompt_logs 
                {where_sql}
                ORDER BY timestamp DESC 
                LIMIT ?
            """
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            # Convert to list of dicts
            results = []
            for row in rows:
                result = dict(row)
                # Parse JSON fields
                result["model_parameters"] = json.loads(result["model_parameters"])
                result["prompt_metadata"] = json.loads(result["prompt_metadata"])
                result["custom_metrics"] = json.loads(result["custom_metrics"])
                results.append(result)

            return results

    def _query_jsonl(
        self,
        prompt_type: Optional[str] = None,
        prompt_version: Optional[str] = None,
        model_name: Optional[str] = None,
        parsed_success: Optional[bool] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query JSONL logs with filters."""
        if not self.jsonl_path.exists():
            return []

        results = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if len(results) >= limit:
                    break

                try:
                    entry = json.loads(line.strip())

                    # Apply filters
                    if prompt_type and entry.get("prompt_type") != prompt_type:
                        continue
                    if prompt_version and entry.get("prompt_version") != prompt_version:
                        continue
                    if model_name and entry.get("model_name") != model_name:
                        continue
                    if (
                        parsed_success is not None
                        and entry.get("parsed_success") != parsed_success
                    ):
                        continue

                    results.append(entry)

                except json.JSONDecodeError:
                    continue

        # Sort by timestamp (most recent first)
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return results[:limit]

    def get_performance_summary(
        self, prompt_type: Optional[str] = None, prompt_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get performance summary statistics for logged calls.

        Returns:
            Dict with aggregated metrics
        """
        logs = self.query_logs(
            prompt_type=prompt_type, prompt_version=prompt_version, limit=1000
        )

        if not logs:
            return {"message": "No logs found for the specified criteria"}

        # Calculate metrics
        total_calls = len(logs)
        successful_parses = sum(1 for log in logs if log.get("parsed_success", False))
        total_records = sum(
            log.get("records_extracted", 0) or 0
            for log in logs
            if log.get("records_extracted")
        )

        avg_records_per_call = total_records / total_calls if total_calls > 0 else 0
        success_rate = successful_parses / total_calls if total_calls > 0 else 0

        # Execution time stats (if available)
        execution_times = [
            log.get("execution_time_seconds")
            for log in logs
            if log.get("execution_time_seconds")
        ]
        avg_execution_time = (
            sum(execution_times) / len(execution_times) if execution_times else None
        )

        return {
            "prompt_type": prompt_type,
            "prompt_version": prompt_version,
            "total_calls": total_calls,
            "successful_parses": successful_parses,
            "parsing_success_rate": round(success_rate * 100, 2),
            "total_records_extracted": total_records,
            "avg_records_per_call": round(avg_records_per_call, 2),
            "avg_execution_time_seconds": (
                round(avg_execution_time, 2) if avg_execution_time else None
            ),
            "date_range": {
                "earliest": min(log.get("timestamp", "") for log in logs),
                "latest": max(log.get("timestamp", "") for log in logs),
            },
        }


# Example usage
if __name__ == "__main__":
    # Initialize logger
    logger = PromptLogger()

    # Example log entry
    logger.log_llm_call(
        prompt_metadata={"prompt_type": "health_data_extraction", "version": "v2.0.0"},
        model_name="gpt-4o",
        model_parameters={"temperature": 0, "max_tokens": 16000},
        system_prompt="Test system prompt",
        user_prompt="Test user prompt",
        raw_response='[{"Country": "Test", "Event": "Test"}]',
        parsed_success=True,
        records_extracted=1,
        execution_time_seconds=2.5,
    )

    print("\n✅ Prompt Logger setup complete!")
