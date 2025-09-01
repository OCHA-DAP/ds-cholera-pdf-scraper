"""
Simplified surveillance preprocessing that focuses on table extraction.
Bypasses potentially slow PDF segmentation step.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

from src.config import Config
from src.preprocess.table_detection_surveillance import TableDetectionEngine

logger = logging.getLogger(__name__)


def process_surveillance_bulletin(
    pdf_path: str, output_dir: str = None
) -> Dict[str, Any]:
    """Process a WHO surveillance bulletin focusing on table extraction."""

    output_dir = Path(output_dir) if output_dir else Config.OUTPUTS_DIR
    output_dir.mkdir(exist_ok=True)

    logger.info(f"Processing surveillance bulletin: {pdf_path}")

    # Initialize table detector
    table_detector = TableDetectionEngine()

    try:
        # Extract surveillance tables
        table_result = table_detector.detect_tables(pdf_path)

        if not table_result.get("detected_tables"):
            return {
                "success": False,
                "error": "No surveillance tables detected",
                "pdf_path": pdf_path,
            }

        # Get surveillance data
        main_table = table_result["detected_tables"][0]
        surveillance_df = main_table["data"]

        # Create summary including the DataFrame for LLM processing
        surveillance_summary = {
            "records": len(surveillance_df),
            "countries": surveillance_df["Country"].nunique(),
            "events": surveillance_df["Event"].nunique(),
            "event_types": surveillance_df["Event"].value_counts().head(10).to_dict(),
            "grade_distribution": surveillance_df["Grade"].value_counts().to_dict(),
            "top_countries": surveillance_df["Country"]
            .value_counts()
            .head(10)
            .to_dict(),
            "data": surveillance_df,  # Include DataFrame for LLM processing
        }

        # Log tabular preprocessing results using new organized system
        import time
        from src.tabular_preprocessing_logger import TabularPreprocessingLogger

        logger_db = TabularPreprocessingLogger()

        # Calculate processing time (placeholder for now) 
        processing_time = 1.0  # TODO: Add actual timing

        # Log to new tabular preprocessing table with organized file storage
        preprocessing_log_id = logger_db.log_tabular_preprocessing(
            pdf_path=pdf_path,
            preprocessing_method="pdfplumber",
            surveillance_df=surveillance_df,
            extraction_metadata=table_result["extraction_metadata"],
            execution_time_seconds=processing_time,
            success=True,
            error_message=None,
        )

        logger.info(f"Processing completed: {len(surveillance_df)} records extracted")
        logger.info(f"Results logged to database with ID: {preprocessing_log_id}")

        return {
            "success": True,
            "pdf_path": pdf_path,
            "surveillance_data": surveillance_summary,
            "output_files": {},  # No intermediate files saved
            "table_detection_metadata": table_result["extraction_metadata"],
            "preprocessing_log_id": preprocessing_log_id,  # Link to database record
        }

    except Exception as e:
        # Log failed preprocessing to database
        from src.prompt_logger import PromptLogger

        logger_db = PromptLogger()
        preprocessing_log_id = logger_db.log_preprocessing_result(
            pdf_path=pdf_path,
            preprocessing_type="pdfplumber_simple",
            success=False,
            records_extracted=0,
            execution_time_seconds=0.0,
            raw_result={"error": str(e)},
            error_message=str(e),
        )

        logger.error(f"Processing failed: {e}")
        logger.info(f"Error logged to database with ID: {preprocessing_log_id}")
        return {
            "success": False,
            "error": str(e),
            "pdf_path": pdf_path,
            "preprocessing_log_id": preprocessing_log_id,
        }


if __name__ == "__main__":
    # Test with the WHO bulletin
    from src.config import Config

    pdf_path = str(
        Path(Config.LOCAL_DIR_BASE)
        / "Cholera - General"
        / "WHO_bulletins_historical"
        / "Week_28__7_-_13_July_2025.pdf"
    )

    print("🔄 Processing WHO surveillance bulletin...")

    try:
        results = process_surveillance_bulletin(pdf_path)

        if results["success"]:
            print("✅ Processing completed successfully!")
            print(f"📊 Records extracted: {results['surveillance_data']['records']}")
            print(f"🌍 Countries: {results['surveillance_data']['countries']}")
            print(f"🦠 Events: {results['surveillance_data']['events']}")
            print(f"💾 Files saved:")
            for name, path in results["output_files"].items():
                print(f"   • {name}: {path}")

            print(f"\n🔝 Top events:")
            for event, count in list(
                results["surveillance_data"]["event_types"].items()
            )[:5]:
                print(f"   • {event}: {count}")

        else:
            print(f"❌ Processing failed: {results.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
