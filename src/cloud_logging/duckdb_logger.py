"""
DuckDB-based logger for cloud storage.

Writes logs to Parquet files that can be queried directly from Azure Blob Storage
without downloading. Each run creates a new Parquet file to avoid locking issues.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class DuckDBLogger:
    """
    Logger that writes to Parquet files for DuckDB querying.

    Each log entry is written to a local Parquet file that can be uploaded to blob.
    Multiple files can be queried together using DuckDB's read_parquet('*.parquet').
    """

    def __init__(self, output_dir: str = None):
        """
        Initialize DuckDB logger.

        Args:
            output_dir: Directory for local Parquet files (default: logs/parquet)
        """
        if output_dir is None:
            # Default to logs/parquet directory
            project_root = Path(__file__).parent.parent
            output_dir = project_root / "logs" / "parquet"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Separate directories for each table type
        self.prompt_logs_dir = self.output_dir / "prompt_logs"
        self.preprocessing_logs_dir = self.output_dir / "tabular_preprocessing_logs"

        self.prompt_logs_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessing_logs_dir.mkdir(parents=True, exist_ok=True)

    def _get_next_id(self) -> int:
        """
        Get next sequential ID by reading max ID from ALL existing parquet files.
        Checks both prompt_logs and preprocessing_logs to maintain global sequence.

        Returns:
            int: Next available ID
        """
        max_id = 0

        # Check prompt logs parquet files
        if self.prompt_logs_dir.exists():
            for parquet_file in self.prompt_logs_dir.glob("run_*.parquet"):
                try:
                    df = pd.read_parquet(parquet_file)
                    if len(df) > 0 and 'id' in df.columns:
                        file_max_id = df['id'].max()
                        if pd.notna(file_max_id):
                            max_id = max(max_id, int(file_max_id))
                except Exception:
                    continue

        # Check preprocessing logs parquet files
        if self.preprocessing_logs_dir.exists():
            for parquet_file in self.preprocessing_logs_dir.glob("run_*.parquet"):
                try:
                    df = pd.read_parquet(parquet_file)
                    if len(df) > 0 and 'id' in df.columns:
                        file_max_id = df['id'].max()
                        if pd.notna(file_max_id):
                            max_id = max(max_id, int(file_max_id))
                except Exception:
                    continue

        # Check legacy SQLite database (both tables)
        try:
            import sqlite3
            from pathlib import Path
            sqlite_path = Path(__file__).parent.parent.parent / "logs" / "prompts" / "prompt_logs.db"
            if sqlite_path.exists():
                conn = sqlite3.connect(sqlite_path)
                cursor = conn.cursor()

                # Check prompt_logs table
                cursor.execute("SELECT MAX(id) FROM prompt_logs")
                result = cursor.fetchone()
                if result[0]:
                    max_id = max(max_id, result[0])

                # Check tabular_preprocessing_logs table
                cursor.execute("SELECT MAX(id) FROM tabular_preprocessing_logs")
                result = cursor.fetchone()
                if result[0]:
                    max_id = max(max_id, result[0])

                conn.close()
        except Exception:
            # SQLite not available or no database, that's fine
            pass

        return max_id + 1

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
        preprocessing_id: Optional[int] = None,
        git_commit_hash: Optional[str] = None,
    ) -> str:
        """
        Log an LLM call to a Parquet file.

        Args:
            prompt_metadata: Metadata from PromptManager
            model_name: Name and version of the model used
            model_parameters: All model parameters
            system_prompt: Complete system prompt sent
            user_prompt: Complete user prompt sent
            raw_response: Raw response from model
            parsed_success: Whether parsing was successful
            records_extracted: Number of records successfully extracted
            parsing_errors: Any parsing error messages
            execution_time_seconds: Time taken for the call
            custom_metrics: Additional metrics
            preprocessing_id: Link to preprocessing log entry
            git_commit_hash: Git commit hash for reproducibility

        Returns:
            str: Run ID (sequential integer)
        """
        # Use sequential ID (continues from SQLite if available)
        run_id = self._get_next_id()
        timestamp = datetime.now()

        # Create log entry
        log_entry = {
            "id": run_id,
            "timestamp": timestamp,
            "prompt_type": prompt_metadata.get("prompt_type"),
            "prompt_version": prompt_metadata.get("version"),
            "model_name": model_name,
            "model_parameters": json.dumps(model_parameters),
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": raw_response,
            "parsed_success": parsed_success,
            "records_extracted": records_extracted,
            "parsing_errors": parsing_errors,
            "execution_time_seconds": execution_time_seconds,
            "prompt_metadata": json.dumps(prompt_metadata),
            "custom_metrics": json.dumps(custom_metrics or {}),
            "preprocessing_id": preprocessing_id,
            "git_commit_hash": git_commit_hash,
        }

        # Convert to DataFrame
        df = pd.DataFrame([log_entry])

        # Write to Parquet file
        output_path = self.prompt_logs_dir / f"run_{run_id}.parquet"
        df.to_parquet(
            output_path,
            engine='pyarrow',
            compression='zstd',
            index=False
        )

        print(f"📝 Logged LLM call to Parquet (ID: {run_id}) - {prompt_metadata.get('prompt_type')} v{prompt_metadata.get('version')}")
        print(f"   File: {output_path.name}")

        return str(run_id)

    def log_preprocessing_result(
        self,
        pdf_path: str,
        preprocessing_method: str,
        success: bool,
        records_extracted: Optional[int] = None,
        countries_detected: Optional[int] = None,
        events_detected: Optional[int] = None,
        execution_time_seconds: Optional[float] = None,
        csv_output_path: Optional[str] = None,
        metadata_json_path: Optional[str] = None,
        table_summary: Optional[Dict[str, Any]] = None,
        extraction_metadata: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        source_pdf_hash: Optional[str] = None,
    ) -> str:
        """
        Log preprocessing result to a Parquet file.

        Args:
            pdf_path: Path to the processed PDF
            preprocessing_method: Type of preprocessing
            success: Whether preprocessing succeeded
            records_extracted: Number of records extracted
            countries_detected: Number of countries detected
            events_detected: Number of events detected
            execution_time_seconds: Processing time
            csv_output_path: Path to output CSV
            metadata_json_path: Path to metadata JSON
            table_summary: Summary statistics
            extraction_metadata: Technical metadata
            error_message: Error message if failed
            source_pdf_hash: Hash for deduplication

        Returns:
            Log entry ID as string
        """
        # Use sequential ID (same counter as prompt logs)
        run_id = self._get_next_id()
        timestamp = datetime.now()

        # Create log entry
        log_entry = {
            "id": run_id,
            "timestamp": timestamp,
            "pdf_path": pdf_path,
            "preprocessing_method": preprocessing_method,
            "success": success,
            "records_extracted": records_extracted,
            "countries_detected": countries_detected,
            "events_detected": events_detected,
            "execution_time_seconds": execution_time_seconds,
            "csv_output_path": csv_output_path,
            "metadata_json_path": metadata_json_path,
            "table_summary": json.dumps(table_summary) if table_summary else None,
            "extraction_metadata": json.dumps(extraction_metadata) if extraction_metadata else None,
            "error_message": error_message,
            "source_pdf_hash": source_pdf_hash,
        }

        # Convert to DataFrame
        df = pd.DataFrame([log_entry])

        # Write to Parquet file
        output_path = self.preprocessing_logs_dir / f"run_{run_id}.parquet"
        df.to_parquet(
            output_path,
            engine='pyarrow',
            compression='zstd',
            index=False
        )

        print(f"📝 Logged preprocessing result to Parquet (ID: {run_id}) - {preprocessing_method}")
        print(f"   File: {output_path.name}")

        return str(run_id)

    def get_local_parquet_path(self, run_id: str, log_type: str = "prompt") -> Path:
        """
        Get the path to a local Parquet file.

        Args:
            run_id: Run ID
            log_type: Type of log ("prompt" or "preprocessing")

        Returns:
            Path to the Parquet file
        """
        if log_type == "prompt":
            return self.prompt_logs_dir / f"run_{run_id}.parquet"
        elif log_type == "preprocessing":
            return self.preprocessing_logs_dir / f"run_{run_id}.parquet"
        else:
            raise ValueError(f"Unknown log_type: {log_type}")


# Example usage
if __name__ == "__main__":
    import tempfile

    # Initialize logger with temp directory
    logger = DuckDBLogger(output_dir=tempfile.mkdtemp())

    # Test logging an LLM call
    run_id = logger.log_llm_call(
        prompt_metadata={
            "prompt_type": "health_data_extraction",
            "version": "v1.4.7",
        },
        model_name="gpt-5",
        model_parameters={"temperature": 0, "max_tokens": 16384},
        system_prompt="Test system prompt",
        user_prompt="Test user prompt",
        raw_response='[{"Country": "Test"}]',
        parsed_success=True,
        records_extracted=1,
        execution_time_seconds=2.5,
        git_commit_hash="abc123",
    )

    print(f"\n✅ Test successful! Run ID: {run_id}")
    print(f"📁 Parquet files created in: {logger.output_dir}")
