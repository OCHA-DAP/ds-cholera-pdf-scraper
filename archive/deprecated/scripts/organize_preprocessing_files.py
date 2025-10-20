#!/usr/bin/env python3
"""
Migrate existing files to the new organized structure.
This is optional - you can start fresh with new runs.
"""

import re
import shutil
from pathlib import Path


def migrate_existing_files():
    """Move existing files to organized structure."""

    outputs_dir = Path("outputs")

    # Current files in outputs/
    files = list(outputs_dir.glob("*.csv")) + list(outputs_dir.glob("*.json"))

    preprocessing_files = []
    llm_extraction_files = []
    surveillance_corrected_files = []
    other_files = []

    # Categorize files
    for file in files:
        name = file.name

        if name.startswith("preprocessing_"):
            preprocessing_files.append(file)
        elif name.startswith("extraction_"):
            llm_extraction_files.append(file)
        elif name.startswith("surveillance_corrected_"):
            surveillance_corrected_files.append(file)
        else:
            other_files.append(file)

    print("📊 FILE CATEGORIZATION")
    print("=" * 40)
    print(f"Preprocessing files: {len(preprocessing_files)}")
    print(f"LLM extraction files: {len(llm_extraction_files)}")
    print(f"Surveillance corrected files: {len(surveillance_corrected_files)}")
    print(f"Other files: {len(other_files)}")

    # Show what would be moved (don't actually move for safety)
    print("\n🔄 PROPOSED MIGRATIONS")
    print("=" * 40)

    print("\n📁 Preprocessing files → outputs/preprocessing/tables/")
    for file in preprocessing_files[:5]:  # Show first 5
        new_name = file.name.replace("preprocessing_", "tables_").replace(
            "_original_data", ""
        )
        print(f"  {file.name} → {new_name}")
    if len(preprocessing_files) > 5:
        print(f"  ... and {len(preprocessing_files) - 5} more")

    print("\n📁 LLM extractions → outputs/llm_extractions/standard/")
    for file in llm_extraction_files[:5]:  # Show first 5
        new_name = file.name  # Keep same name
        print(f"  {file.name} → {new_name}")
    if len(llm_extraction_files) > 5:
        print(f"  ... and {len(llm_extraction_files) - 5} more")

    print("\n📁 Surveillance corrected → outputs/llm_extractions/corrected/")
    for file in surveillance_corrected_files[:5]:  # Show first 5
        new_name = file.name  # Keep same name
        print(f"  {file.name} → {new_name}")
    if len(surveillance_corrected_files) > 5:
        print(f"  ... and {len(surveillance_corrected_files) - 5} more")


def show_new_naming_convention():
    """Show the new naming convention."""

    print("\n🎯 NEW NAMING CONVENTION")
    print("=" * 50)

    print("📋 TABULAR PREPROCESSING:")
    print("  Database: tabular_preprocessing_logs(id)")
    print("  Files:")
    print("    • outputs/preprocessing/tables/tables_{id}.csv")
    print("    • outputs/preprocessing/metadata/metadata_{id}.json")
    print("")

    print("📋 LLM EXTRACTIONS:")
    print("  Database: prompt_logs(id, preprocessing_id)")
    print("  Files:")
    print("    • outputs/llm_extractions/standard/extraction_{id}.csv")
    print("    • outputs/llm_extractions/corrected/corrected_{id}.csv")
    print("")

    print("🔗 LINKING EXAMPLE:")
    print("  1. PDF processed → tabular_preprocessing_logs(id=5)")
    print("     Files: tables_5.csv, metadata_5.json")
    print("  2. LLM extraction → prompt_logs(id=42, preprocessing_id=5)")
    print("     File: extraction_42.csv")
    print("  3. Perfect chain: PDF → tables_5.csv → extraction_42.csv")


def create_logger_helper():
    """Create a helper class for the new logging system."""

    content = '''"""
Enhanced PromptLogger with tabular preprocessing support.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd


class TabularPreprocessingLogger:
    """Logger specifically for tabular preprocessing results."""
    
    def __init__(self, db_path: str = "logs/prompts/prompt_logs.db"):
        self.db_path = Path(db_path)
        self.outputs_dir = Path("outputs")
    
    def log_tabular_preprocessing(
        self,
        pdf_path: str,
        preprocessing_method: str,  # 'pdfplumber', 'table-focused', etc.
        surveillance_df: pd.DataFrame,
        extraction_metadata: Dict[str, Any],
        execution_time_seconds: float,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> int:
        """
        Log tabular preprocessing results with organized file storage.
        
        Returns:
            preprocessing_id: ID for linking to LLM extractions
        """
        timestamp = datetime.now().isoformat()
        
        # Calculate summary stats
        records_extracted = len(surveillance_df)
        countries_detected = surveillance_df["Country"].nunique() if "Country" in surveillance_df.columns else 0
        events_detected = surveillance_df["Event"].nunique() if "Event" in surveillance_df.columns else 0
        
        # Log to database first to get ID
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO tabular_preprocessing_logs (
                    timestamp, pdf_path, preprocessing_method, success,
                    records_extracted, countries_detected, events_detected,
                    execution_time_seconds, table_summary, extraction_metadata,
                    error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, pdf_path, preprocessing_method, success,
                records_extracted, countries_detected, events_detected,
                execution_time_seconds, 
                json.dumps({
                    "records": records_extracted,
                    "countries": countries_detected,
                    "events": events_detected,
                    "top_events": surveillance_df["Event"].value_counts().head(5).to_dict() if "Event" in surveillance_df.columns else {}
                }),
                json.dumps(extraction_metadata),
                error_message
            ))
            
            preprocessing_id = cursor.lastrowid
            
            # Save files with ID-based naming
            csv_path = f"outputs/preprocessing/tables/tables_{preprocessing_id}.csv"
            metadata_path = f"outputs/preprocessing/metadata/metadata_{preprocessing_id}.json"
            
            # Update database with file paths
            cursor.execute("""
                UPDATE tabular_preprocessing_logs 
                SET csv_output_path = ?, metadata_json_path = ?
                WHERE id = ?
            """, (csv_path, metadata_path, preprocessing_id))
            
            conn.commit()
        
        # Save files
        surveillance_df.to_csv(csv_path, index=False)
        
        with open(metadata_path, 'w') as f:
            json.dump({
                "preprocessing_id": preprocessing_id,
                "timestamp": timestamp,
                "pdf_path": pdf_path,
                "method": preprocessing_method,
                "extraction_metadata": extraction_metadata,
                "summary": {
                    "records": records_extracted,
                    "countries": countries_detected,
                    "events": events_detected
                }
            }, f, indent=2)
        
        print(f"📝 Logged tabular preprocessing (ID: {preprocessing_id})")
        print(f"   📊 {records_extracted} records, {countries_detected} countries, {events_detected} events")
        print(f"   📁 Files: {csv_path}, {metadata_path}")
        
        return preprocessing_id


# Usage example:
# logger = TabularPreprocessingLogger()
# prep_id = logger.log_tabular_preprocessing(
#     pdf_path="/path/to/file.pdf",
#     preprocessing_method="pdfplumber",
#     surveillance_df=extracted_df,
#     extraction_metadata={"method": "surveillance_reconstruction", "pages": [9,10,11]},
#     execution_time_seconds=2.5
# )
# 
# # Then link to LLM extraction:
# llm_logger = PromptLogger()
# llm_id = llm_logger.log_llm_call(..., preprocessing_id=prep_id)
'''

    helper_path = Path("src/tabular_preprocessing_logger.py")
    with open(helper_path, "w") as f:
        f.write(content)

    print(f"📄 Created helper: {helper_path}")


if __name__ == "__main__":
    print("🔄 File Migration Analysis")
    print("=" * 50)

    # Analyze existing files
    migrate_existing_files()

    # Show new system
    show_new_naming_convention()

    # Create helper
    create_logger_helper()

    print("\\n⚠️  NOTE: This script shows what WOULD be migrated.")
    print("   Run with --execute flag to actually move files.")
    print("   Or start fresh with new preprocessing runs.")
