#!/usr/bin/env python3
"""
Migrate a single run from SQLite to DuckDB parquet format.
Useful for moving recent runs that accidentally logged to SQLite.

Usage:
    python scripts/migrate_single_run.py --run-id 202
"""

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

# Set to use duckdb before imports
import os
os.environ['LOG_BACKEND'] = 'duckdb'

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.config import Config
from src.cloud_logging import DuckDBLogger


def migrate_run(run_id: int, upload_to_blob: bool = False):
    """
    Migrate a specific run from SQLite to DuckDB parquet.

    Args:
        run_id: The run ID to migrate
        upload_to_blob: Whether to upload to blob after creating parquet
    """
    # Connect to SQLite
    sqlite_path = Path(__file__).parent.parent / "logs" / "prompts" / "prompt_logs.db"

    if not sqlite_path.exists():
        print(f"❌ SQLite database not found: {sqlite_path}")
        return False

    print(f"📂 Reading run {run_id} from SQLite...")

    conn = sqlite3.connect(sqlite_path)

    # Read the specific run
    df = pd.read_sql(
        f"SELECT * FROM prompt_logs WHERE id = {run_id}",
        conn
    )

    conn.close()

    if len(df) == 0:
        print(f"❌ No records found for run ID {run_id}")
        return False

    print(f"✅ Found {len(df)} record(s) for run {run_id}")
    print(f"   Model: {df.iloc[0]['model_name']}")
    print(f"   Prompt: {df.iloc[0]['prompt_version']}")
    print(f"   Timestamp: {df.iloc[0]['timestamp']}")
    print()

    # Save to parquet
    parquet_dir = Config.get_duckdb_logs_dir() / "prompt_logs"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    output_path = parquet_dir / f"run_{run_id}.parquet"

    df.to_parquet(
        output_path,
        engine='pyarrow',
        compression='zstd',
        index=False
    )

    print(f"✅ Saved to: {output_path}")

    if upload_to_blob:
        print(f"\n📤 Uploading to blob...")
        import ocha_stratus as stratus

        stage = os.getenv("STAGE", "dev")
        proj_dir = Config.BLOB_PROJ_DIR
        blob_path = f"{proj_dir}/processed/logs/prompt_logs/run_{run_id}.parquet"

        stratus.upload_parquet_to_blob(
            df=df,
            blob_name=blob_path,
            stage=stage,
            container_name=Config.BLOB_CONTAINER,
        )

        print(f"✅ Uploaded to: {blob_path}")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Migrate a single run from SQLite to DuckDB parquet"
    )
    parser.add_argument(
        "--run-id",
        type=int,
        required=True,
        help="The run ID to migrate"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to blob storage after creating parquet"
    )

    args = parser.parse_args()

    success = migrate_run(args.run_id, upload_to_blob=args.upload)

    if success:
        print(f"\n🎉 Migration complete!")
    else:
        print(f"\n❌ Migration failed")
        sys.exit(1)
