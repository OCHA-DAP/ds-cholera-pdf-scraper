#!/usr/bin/env python3
"""
Create a new table specifically for tabular preprocessing results.
This separates tabular data extraction from text preprocessing.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict


def create_tabular_preprocessing_table():
    """Create a new table for tabular preprocessing results."""

    db_path = Path("logs/prompts/prompt_logs.db")

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Create new table for tabular preprocessing
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tabular_preprocessing_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                pdf_path TEXT NOT NULL,
                preprocessing_method TEXT NOT NULL,  -- 'pdfplumber', 'table-focused', etc.
                success BOOLEAN NOT NULL,
                records_extracted INTEGER,
                countries_detected INTEGER,
                events_detected INTEGER, 
                execution_time_seconds REAL,
                
                -- File outputs (linked by ID)
                csv_output_path TEXT,      -- e.g. "outputs/preprocessing/tables_{id}.csv" 
                metadata_json_path TEXT,   -- e.g. "outputs/preprocessing/metadata_{id}.json"
                
                -- Core data (redundant with files but useful for queries)
                table_summary JSON,        -- Basic stats: {records, countries, events, etc.}
                extraction_metadata JSON,  -- Technical details: {method, confidence, pages, etc.}
                
                -- Error handling
                error_message TEXT,
                
                -- Linkage
                source_pdf_hash TEXT       -- For deduplication across runs
            )
        """
        )

        print("✅ Created tabular_preprocessing_logs table")

        # Show the schema
        cursor.execute("PRAGMA table_info(tabular_preprocessing_logs)")
        columns = cursor.fetchall()

        print("\n📋 Table Schema:")
        for col in columns:
            print(f"  {col[1]:25} {col[2]:15} {'PRIMARY KEY' if col[5] else ''}")

        conn.commit()


def create_file_organization_structure():
    """Create the organized file structure."""

    base_outputs = Path("outputs")

    # Create organized directories
    dirs_to_create = [
        "preprocessing/tables",  # CSV files with extracted table data
        "preprocessing/metadata",  # JSON files with extraction metadata
        "llm_extractions/standard",  # Standard LLM extraction results
        "llm_extractions/corrected",  # LLM-corrected results (table-focused pipeline)
    ]

    for dir_path in dirs_to_create:
        full_path = base_outputs / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created: {full_path}")


def example_usage():
    """Show how the new system would work."""

    print("\n🎯 NEW SYSTEM USAGE EXAMPLE")
    print("=" * 50)

    example_log = {
        "id": 1,
        "pdf_path": "/path/to/Week_28.pdf",
        "preprocessing_method": "pdfplumber",
        "records_extracted": 118,
        "countries_detected": 47,
        "events_detected": 25,
        "csv_output_path": "outputs/preprocessing/tables/tables_1.csv",
        "metadata_json_path": "outputs/preprocessing/metadata/metadata_1.json",
        "table_summary": {
            "records": 118,
            "countries": 47,
            "events": 25,
            "top_events": {"Cholera": 19, "Measles": 22},
        },
        "extraction_metadata": {
            "method": "surveillance_reconstruction",
            "pages_processed": [9, 10, 11, 12, 13, 14, 15],
            "confidence": 0.95,
        },
    }

    print("Example record:")
    for key, value in example_log.items():
        print(f"  {key}: {value}")

    print("\n🔗 LINKING EXAMPLE")
    print("-" * 30)
    print("1. Preprocessing run creates:")
    print("   • tabular_preprocessing_logs(id=1)")
    print("   • outputs/preprocessing/tables/tables_1.csv")
    print("   • outputs/preprocessing/metadata/metadata_1.json")
    print("")
    print("2. LLM extraction references preprocessing_id=1:")
    print("   • prompt_logs(id=50, preprocessing_id=1)")
    print("   • outputs/llm_extractions/standard/extraction_50.csv")
    print("")
    print("3. Perfect traceability: PDF → tables_1.csv → extraction_50.csv")


if __name__ == "__main__":
    print("🚀 Creating New Tabular Preprocessing System")
    print("=" * 60)

    # Create the database table
    create_tabular_preprocessing_table()

    # Create file organization
    create_file_organization_structure()

    # Show usage example
    example_usage()

    print("\n✅ Setup complete! Ready for new organized preprocessing.")
