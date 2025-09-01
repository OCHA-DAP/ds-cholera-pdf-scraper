"""
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
        
        # Get shared run ID for consistent incrementing across all tables
        from src.run_id_manager import RunIDManager
        run_manager = RunIDManager(self.db_path)
        run_id = run_manager.get_next_run_id()
        
        # Log to database with shared run ID
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO tabular_preprocessing_logs (
                    id, timestamp, pdf_path, preprocessing_method, success,
                    records_extracted, countries_detected, events_detected,
                    execution_time_seconds, table_summary, extraction_metadata,
                    error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, timestamp, pdf_path, preprocessing_method, success,
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
            
            preprocessing_id = run_id
            
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
