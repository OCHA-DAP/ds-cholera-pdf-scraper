#!/usr/bin/env python3
"""
Migrate existing SQLite database to Parquet files for DuckDB querying.

This is a one-time migration script that converts:
- logs/prompts/prompt_logs.db (SQLite)
→ logs/parquet/prompt_logs/historical.parquet
→ logs/parquet/tabular_preprocessing_logs/historical.parquet

After migration, the Parquet files can be:
1. Uploaded to Azure Blob Storage
2. Queried directly from blob using DuckDB (no download needed!)

Usage:
    python scripts/migrate_sqlite_to_parquet.py
    python scripts/migrate_sqlite_to_parquet.py --upload  # Also upload to blob
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import duckdb
from src.config import Config


def migrate_sqlite_to_parquet(
    sqlite_path: str = None,
    output_dir: str = None,
    upload_to_blob: bool = False,
):
    """
    Migrate SQLite database to Parquet files.

    Args:
        sqlite_path: Path to SQLite database (default: logs/prompts/prompt_logs.db)
        output_dir: Output directory for Parquet files (default: logs/parquet)
        upload_to_blob: Whether to upload to Azure Blob Storage
    """
    # Set defaults
    if sqlite_path is None:
        sqlite_path = Config.PROJECT_ROOT / "logs" / "prompts" / "prompt_logs.db"
    else:
        sqlite_path = Path(sqlite_path)

    if output_dir is None:
        output_dir = Config.PROJECT_ROOT / "logs" / "parquet"
    else:
        output_dir = Path(output_dir)

    # Check if SQLite database exists
    if not sqlite_path.exists():
        print(f"❌ SQLite database not found: {sqlite_path}")
        print("   Nothing to migrate.")
        return

    print("=" * 60)
    print("SQLite → Parquet Migration")
    print("=" * 60)
    print(f"📂 Source: {sqlite_path}")
    print(f"📂 Destination: {output_dir}")
    print()

    # Create output directories
    prompt_logs_dir = output_dir / "prompt_logs"
    preprocessing_logs_dir = output_dir / "tabular_preprocessing_logs"
    prompt_logs_dir.mkdir(parents=True, exist_ok=True)
    preprocessing_logs_dir.mkdir(parents=True, exist_ok=True)

    # Use pandas to read from SQLite and DuckDB to write Parquet
    # This is more reliable than DuckDB's SQLite extension
    import pandas as pd
    import sqlite3

    # Initialize DuckDB
    con = duckdb.connect()

    # Connect to SQLite
    print(f"🔗 Connecting to SQLite database: {sqlite_path}")
    sqlite_conn = sqlite3.connect(sqlite_path)

    # Check what tables exist
    cursor = sqlite_conn.cursor()
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
    """)
    available_tables = [row[0] for row in cursor.fetchall()]
    print(f"📋 Found tables: {', '.join(available_tables)}")
    print()

    # Migrate table 1: prompt_logs
    if 'prompt_logs' in available_tables:
        print("📦 Migrating prompt_logs...")

        # Read from SQLite into pandas
        df = pd.read_sql("SELECT * FROM prompt_logs ORDER BY id", sqlite_conn)
        row_count = len(df)
        print(f"   Found {row_count} rows")

        if row_count > 0:
            # Convert timestamp column to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Write to Parquet
            output_path = prompt_logs_dir / "historical.parquet"
            df.to_parquet(output_path, engine='pyarrow', compression='zstd', index=False)

            # Verify
            file_size = output_path.stat().st_size / 1024 / 1024  # MB
            print(f"   ✅ Migrated {row_count} rows")
            print(f"   💾 Output: {output_path.name} ({file_size:.2f} MB)")
        else:
            print("   ⚠️ No rows to migrate")
    else:
        print("⚠️ prompt_logs table not found, skipping...")

    print()

    # Migrate table 2: tabular_preprocessing_logs
    if 'tabular_preprocessing_logs' in available_tables:
        print("📦 Migrating tabular_preprocessing_logs...")

        # Read from SQLite into pandas
        df = pd.read_sql("SELECT * FROM tabular_preprocessing_logs ORDER BY id", sqlite_conn)
        row_count = len(df)
        print(f"   Found {row_count} rows")

        if row_count > 0:
            # Convert timestamp column to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Write to Parquet
            output_path = preprocessing_logs_dir / "historical.parquet"
            df.to_parquet(output_path, engine='pyarrow', compression='zstd', index=False)

            # Verify
            file_size = output_path.stat().st_size / 1024 / 1024  # MB
            print(f"   ✅ Migrated {row_count} rows")
            print(f"   💾 Output: {output_path.name} ({file_size:.2f} MB)")
        else:
            print("   ⚠️ No rows to migrate")
    else:
        print("⚠️ tabular_preprocessing_logs table not found, skipping...")

    # Close SQLite connection
    sqlite_conn.close()

    print()
    print("=" * 60)
    print("✅ Migration Complete!")
    print("=" * 60)
    print()
    print("📁 Parquet files created:")
    print(f"   {prompt_logs_dir}")
    print(f"   {preprocessing_logs_dir}")
    print()

    # Show how to query the files
    print("💡 Query example (local):")
    print("   python")
    print("   >>> import duckdb")
    print("   >>> con = duckdb.connect()")
    print(f"   >>> df = con.execute(\"SELECT * FROM read_parquet('{prompt_logs_dir}/*.parquet')\").df()")
    print("   >>> print(df.head())")
    print()

    # Upload to blob if requested
    if upload_to_blob:
        print("☁️  Uploading to Azure Blob Storage...")
        upload_parquet_to_blob(output_dir)
    else:
        print("💡 To upload to blob storage, run:")
        print("   python scripts/migrate_sqlite_to_parquet.py --upload")


def upload_parquet_to_blob(parquet_dir: Path):
    """Upload Parquet files to Azure Blob Storage using stratus."""
    try:
        import ocha_stratus as stratus
        import pandas as pd
    except ImportError:
        print("❌ ocha-stratus or pandas not available, cannot upload to blob")
        return

    stage = Config.STAGE
    container = Config.BLOB_CONTAINER
    proj_dir = Config.BLOB_PROJ_DIR

    # Upload prompt_logs
    prompt_logs_dir = parquet_dir / "prompt_logs"
    if prompt_logs_dir.exists():
        for parquet_file in prompt_logs_dir.glob("*.parquet"):
            blob_path = f"{proj_dir}/processed/logs/prompt_logs/{parquet_file.name}"
            print(f"   📤 Uploading {parquet_file.name}...")

            # Read parquet file as DataFrame
            df = pd.read_parquet(parquet_file)

            # Use stratus's upload_parquet_to_blob function
            stratus.upload_parquet_to_blob(
                df=df,
                blob_name=blob_path,
                stage=stage,
                container_name=container,
            )
            print(f"      ✅ {blob_path}")

    # Upload tabular_preprocessing_logs
    preprocessing_logs_dir = parquet_dir / "tabular_preprocessing_logs"
    if preprocessing_logs_dir.exists():
        for parquet_file in preprocessing_logs_dir.glob("*.parquet"):
            blob_path = f"{proj_dir}/processed/logs/tabular_preprocessing_logs/{parquet_file.name}"
            print(f"   📤 Uploading {parquet_file.name}...")

            # Read parquet file as DataFrame
            df = pd.read_parquet(parquet_file)

            # Use stratus's upload_parquet_to_blob function
            stratus.upload_parquet_to_blob(
                df=df,
                blob_name=blob_path,
                stage=stage,
                container_name=container,
            )
            print(f"      ✅ {blob_path}")

    print()
    print("✅ Upload complete!")
    print()
    print("💡 Query from blob (no download needed!):")
    print()
    print("   Method 1: Use DuckDBCloudQuery helper")
    print("   >>> from src.cloud_logging import DuckDBCloudQuery")
    print("   >>> query = DuckDBCloudQuery()")
    print("   >>> df = query.get_latest_runs(n=10)")
    print("   >>> print(df)")
    print()
    print("   Method 2: Direct DuckDB query")
    print("   >>> import duckdb")
    print("   >>> from src.cloud_logging import DuckDBCloudQuery")
    print("   >>> query = DuckDBCloudQuery()  # Handles auth")
    print(f"   >>> blob_url = query.get_blob_url('processed/logs/prompt_logs/*.parquet')")
    print("   >>> df = query.execute_query(f\"SELECT * FROM read_parquet('{blob_url}')\")")
    print()
    print(f"   Note: Requires DSCI_AZ_BLOB_{stage.upper()}_SAS_WRITE in environment")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate SQLite database to Parquet files for DuckDB"
    )
    parser.add_argument(
        "--sqlite-path",
        type=str,
        help="Path to SQLite database (default: logs/prompts/prompt_logs.db)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for Parquet files (default: logs/parquet)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to Azure Blob Storage after migration",
    )

    args = parser.parse_args()

    migrate_sqlite_to_parquet(
        sqlite_path=args.sqlite_path,
        output_dir=args.output_dir,
        upload_to_blob=args.upload,
    )


if __name__ == "__main__":
    main()
