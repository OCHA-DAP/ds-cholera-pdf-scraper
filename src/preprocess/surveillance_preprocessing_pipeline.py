"""
Main preprocessing pipeline for WHO surveillance bulletins.
Integrates surveillance table reconstruction with existing pipeline structure.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.config import Config
from src.preprocess.pdf_segmentation import PDFSegmentationEngine
from src.preprocess.table_detection_surveillance import TableDetectionEngine

logger = logging.getLogger(__name__)


class SurveillancePreprocessingPipeline:
    """Complete preprocessing pipeline for WHO surveillance bulletins."""

    def __init__(self, output_dir: str = None):
        """Initialize the preprocessing pipeline."""

        self.output_dir = Path(output_dir) if output_dir else Config.OUTPUTS_DIR
        self.output_dir.mkdir(exist_ok=True)

        # Initialize components
        self.pdf_segmenter = PDFSegmentationEngine()
        self.table_detector = TableDetectionEngine()

        logger.info("Surveillance preprocessing pipeline initialized")

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process a WHO surveillance bulletin PDF."""

        logger.info(f"Processing WHO bulletin: {pdf_path}")

        # Step 1: PDF segmentation (for completeness, though surveillance
        # reconstruction doesn't strictly need it)
        try:
            segmentation_result = self.pdf_segmenter.segment_pdf(pdf_path)
            logger.info("PDF segmentation completed")
        except Exception as e:
            logger.warning(f"PDF segmentation failed: {e}")
            segmentation_result = {}

        # Step 2: Surveillance table detection and reconstruction
        try:
            table_result = self.table_detector.detect_tables(pdf_path)
            logger.info("Surveillance table detection completed")
        except Exception as e:
            logger.error(f"Table detection failed: {e}")
            return {"success": False, "error": str(e), "pdf_path": pdf_path}

        # Step 3: Extract surveillance data
        surveillance_data = self._extract_surveillance_data(table_result)

        # Step 4: Save results
        results = {
            "success": True,
            "pdf_path": pdf_path,
            "segmentation": segmentation_result,
            "table_detection": table_result,
            "surveillance_data": surveillance_data,
            "output_files": self._save_results(
                pdf_path, table_result, surveillance_data
            ),
        }

        logger.info(
            f"Processing completed: {surveillance_data['records']} records extracted"
        )

        return results

    def _extract_surveillance_data(
        self, table_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract structured surveillance data from table detection results."""

        if not table_result.get("detected_tables"):
            return {"records": 0, "countries": 0, "events": 0, "data": None}

        # Get the main surveillance table
        main_table = table_result["detected_tables"][0]
        surveillance_df = main_table["data"]

        return {
            "records": len(surveillance_df),
            "countries": surveillance_df["Country"].nunique(),
            "events": surveillance_df["Event"].nunique(),
            "event_types": surveillance_df["Event"].value_counts().to_dict(),
            "grade_distribution": surveillance_df["Grade"].value_counts().to_dict(),
            "top_countries": surveillance_df["Country"]
            .value_counts()
            .head(10)
            .to_dict(),
            "data": surveillance_df,
        }

    def _save_results(
        self,
        pdf_path: str,
        table_result: Dict[str, Any],
        surveillance_data: Dict[str, Any],
    ) -> Dict[str, str]:
        """Save processing results to files."""

        pdf_name = Path(pdf_path).stem
        output_files = {}

        # Keep surveillance DataFrame in memory - no intermediate CSV
        # Only final LLM output will be saved as CSV
        if surveillance_data["data"] is not None:
            logger.info(
                f"Surveillance data ready for LLM processing: {len(surveillance_data['data'])} records"
            )

        # Save full results as JSON (excluding DataFrame for JSON serialization)
        json_data = {
            "pdf_path": pdf_path,
            "table_detection": {
                k: v
                for k, v in table_result.items()
                if k != "detected_tables" or not table_result["detected_tables"]
            },
            "surveillance_summary": {
                k: v for k, v in surveillance_data.items() if k != "data"
            },
        }

        # Add table metadata without the actual DataFrame
        if table_result.get("detected_tables"):
            json_data["table_detection"]["table_metadata"] = [
                {k: v for k, v in table.items() if k != "data"}
                for table in table_result["detected_tables"]
            ]

        json_path = self.output_dir / f"{pdf_name}_processing_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        output_files["results_json"] = str(json_path)

        return output_files


def process_who_bulletin(pdf_path: str, output_dir: str = None) -> Dict[str, Any]:
    """Main function to process a WHO surveillance bulletin."""

    pipeline = SurveillancePreprocessingPipeline(output_dir)
    return pipeline.process_pdf(pdf_path)


if __name__ == "__main__":
    # Test with the WHO bulletin
    from src.config import Config

    pdf_path = str(
        Path(Config.LOCAL_DIR_BASE)
        / "Cholera - General"
        / "WHO_bulletins_historical"
        / "Week_28__7_-_13_July_2025.pdf"
    )

    print("🔄 Running surveillance preprocessing pipeline...")

    try:
        results = process_who_bulletin(pdf_path)

        if results["success"]:
            print("✅ Processing completed successfully!")
            print(f"📊 Records extracted: {results['surveillance_data']['records']}")
            print(f"🌍 Countries: {results['surveillance_data']['countries']}")
            print(f"🦠 Events: {results['surveillance_data']['events']}")
            print(f"💾 Output files: {list(results['output_files'].keys())}")
        else:
            print(f"❌ Processing failed: {results.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        import traceback

        traceback.print_exc()
