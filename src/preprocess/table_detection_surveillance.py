"""
Table detection using surveillance table reconstruction for WHO bulletins.
Specifically designed for multi-page surveillance tables with interspersed text.
Replaces TATR approach with purpose-built WHO bulletin table extraction.
"""

import logging
from typing import Any, Dict, List

import pandas as pd

from src.preprocess.surveillance_table_reconstructor import (
    SurveillanceTableReconstructor,
)

logger = logging.getLogger(__name__)


class TableDetectionEngine:
    """Surveillance-aware table detection for WHO bulletins."""

    def __init__(self, confidence_threshold: float = 0.95):
        """Initialize surveillance table detection."""
        self.confidence_threshold = confidence_threshold
        self.reconstructor = None
        logger.info("Surveillance table detection engine initialized")

    def detect_tables(
        self, pdf_path: str, table_regions: List[Dict] = None
    ) -> Dict[str, Any]:
        """Detect and reconstruct the main surveillance table from PDF."""

        logger.info(f"Detecting surveillance tables in PDF: {pdf_path}")

        # Initialize reconstructor
        self.reconstructor = SurveillanceTableReconstructor(pdf_path)

        # Extract the main surveillance table
        surveillance_df = self.reconstructor.extract_surveillance_table()

        if surveillance_df.empty:
            logger.warning("No surveillance tables detected")
            return {
                "detected_tables": [],
                "total_tables": 0,
                "extraction_metadata": {
                    "method": "surveillance_reconstruction",
                    "pdf_path": pdf_path,
                    "success": False,
                },
            }

        # Convert to standardized table detection format
        table_info = {
            "table_id": "main_surveillance_table",
            "type": "surveillance_table",
            "pages": list(range(9, 16)),  # Main surveillance table spans pages 9-15
            "records": len(surveillance_df),
            "countries": surveillance_df["Country"].nunique(),
            "events": surveillance_df["Event"].nunique(),
            "confidence": self.confidence_threshold,
            "data": surveillance_df,
            "structure": {
                "columns": list(surveillance_df.columns),
                "rows": len(surveillance_df),
                "data_types": {
                    col: str(surveillance_df[col].dtype)
                    for col in surveillance_df.columns
                },
            },
            "bbox": None,  # N/A for reconstructed tables
            "extraction_method": "surveillance_reconstruction",
        }

        logger.info(f"Surveillance table detected: {len(surveillance_df)} records")
        logger.info(f"Countries: {surveillance_df['Country'].nunique()}")
        logger.info(f"Events: {surveillance_df['Event'].nunique()}")

        return {
            "detected_tables": [table_info],
            "total_tables": 1,
            "extraction_metadata": {
                "method": "surveillance_reconstruction",
                "pdf_path": pdf_path,
                "success": True,
                "records_extracted": len(surveillance_df),
                "countries": surveillance_df["Country"].nunique(),
                "events": surveillance_df["Event"].nunique(),
            },
        }

    def extract_table_content(
        self, pdf_path: str, table_regions: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Extract table content using surveillance reconstruction."""

        detection_result = self.detect_tables(pdf_path, table_regions)

        if not detection_result["detected_tables"]:
            return []

        # Convert surveillance data to content format
        table_contents = []
        for table_info in detection_result["detected_tables"]:
            content = {
                "table_id": table_info["table_id"],
                "content": table_info["data"],
                "format": "dataframe",
                "structure": table_info["structure"],
                "extraction_method": "surveillance_reconstruction",
            }
            table_contents.append(content)

        return table_contents


class TableDetector(TableDetectionEngine):
    """Backward compatibility alias."""

    def __init__(self, confidence_threshold: float = 0.95):
        super().__init__(confidence_threshold)

    def detect_tables_in_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Detect tables in PDF (legacy interface)."""

        result = self.detect_tables(pdf_path)
        return result.get("detected_tables", [])


def detect_tables(pdf_path: str) -> List[Dict[str, Any]]:
    """Main function to detect surveillance tables in a PDF."""

    detector = TableDetector()
    return detector.detect_tables_in_pdf(pdf_path)


def detect_tables_with_metadata(pdf_path: str) -> Dict[str, Any]:
    """Detect tables with full metadata."""

    engine = TableDetectionEngine()
    return engine.detect_tables(pdf_path)
