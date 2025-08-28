"""
Preprocessing Manager - Orchestrates the preprocessing pipeline.
Provides unified interface for all preprocessing operations.
"""

import time
from typing import Any, Dict, List, Optional

# Try absolute import first, fall back to local import if needed
try:
    from src.config import Config
except ImportError:
    from config import Config

# Optional imports for fallback compatibility
try:
    import PyPDF2

    PYPDF2_AVAILABLE = True
except ImportError:
    PyPDF2 = None
    PYPDF2_AVAILABLE = False


class PreprocessingManager:
    """
    Orchestrates the preprocessing pipeline with optional components.
    Maintains compatibility with existing LLM extraction workflow.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize preprocessing manager with configuration.

        Args:
            config: Optional configuration override
        """
        self.config = config or self._get_default_config()
        self.segmentation_engine = None
        self.table_detection_engine = None
        self.table_stitching_engine = None
        self.narrative_linking_engine = None

        # Initialize engines based on config
        self._initialize_engines()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default preprocessing configuration."""
        return {
            "enable_preprocessing": getattr(Config, "ENABLE_PREPROCESSING", False),
            "preprocessing_mode": getattr(Config, "PREPROCESSING_MODE", "hybrid"),
            "table_detection_enabled": getattr(Config, "TABLE_DETECTION_ENABLED", True),
            "narrative_linking_enabled": getattr(
                Config, "NARRATIVE_LINKING_ENABLED", False
            ),
            "table_confidence_threshold": getattr(
                Config, "TABLE_CONFIDENCE_THRESHOLD", 0.8
            ),
        }

    def _initialize_engines(self):
        """Initialize preprocessing engines based on configuration."""
        if not self.config.get("enable_preprocessing", False):
            return

        # Initialize segmentation engine (always needed if preprocessing enabled)
        from src.preprocess.pdf_segmentation import PDFSegmentationEngine

        self.segmentation_engine = PDFSegmentationEngine()

        # Initialize table detection if enabled
        if self.config.get("table_detection_enabled", True):
            from src.preprocess.table_detection import TableDetectionEngine

            self.table_detection_engine = TableDetectionEngine(
                confidence_threshold=self.config.get("table_confidence_threshold", 0.8)
            )

            from src.preprocess.table_stitching import TableStitchingEngine

            self.table_stitching_engine = TableStitchingEngine()

        # Initialize narrative linking if enabled
        if self.config.get("narrative_linking_enabled", False):
            from src.preprocess.narrative_linking import NarrativeLinkingEngine

            self.narrative_linking_engine = NarrativeLinkingEngine()

    def preprocess_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Run complete preprocessing pipeline on a PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with preprocessed data and metadata
        """
        if not self.config.get("enable_preprocessing", False):
            return {"preprocessing_enabled": False, "raw_text_path": pdf_path}

        start_time = time.time()
        result = {
            "pdf_path": pdf_path,
            "preprocessing_enabled": True,
            "preprocessing_config": self.config.copy(),
            "processing_times": {},
            "components": {},
        }

        try:
            # Step 1: PDF Segmentation
            if self.segmentation_engine:
                seg_start = time.time()
                segmentation_result = self.segmentation_engine.segment_pdf(pdf_path)
                result["components"]["segmentation"] = segmentation_result
                result["processing_times"]["segmentation"] = time.time() - seg_start

            # Step 2: Table Detection
            if self.table_detection_engine and "segmentation" in result["components"]:
                table_start = time.time()
                table_result = self.table_detection_engine.detect_tables(
                    pdf_path,
                    result["components"]["segmentation"].get("table_regions", []),
                )
                result["components"]["table_detection"] = table_result
                result["processing_times"]["table_detection"] = (
                    time.time() - table_start
                )

                # Step 3: Table Stitching
                if self.table_stitching_engine and table_result.get("tables"):
                    stitch_start = time.time()
                    stitched_result = self.table_stitching_engine.stitch_tables(
                        table_result["tables"]
                    )
                    result["components"]["table_stitching"] = stitched_result
                    result["processing_times"]["table_stitching"] = (
                        time.time() - stitch_start
                    )

            # Step 4: Narrative Linking (optional)
            if self.narrative_linking_engine and "segmentation" in result["components"]:
                narrative_start = time.time()
                narrative_result = self.narrative_linking_engine.link_narrative(
                    result["components"]["segmentation"].get("text_blocks", [])
                )
                result["components"]["narrative_linking"] = narrative_result
                result["processing_times"]["narrative_linking"] = (
                    time.time() - narrative_start
                )

            result["total_processing_time"] = time.time() - start_time
            result["success"] = True

        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            result["total_processing_time"] = time.time() - start_time

        return result

    def format_for_llm(self, preprocessed_data: Dict[str, Any]) -> str:
        """
        Format preprocessed data for LLM consumption.

        Args:
            preprocessed_data: Output from preprocess_pdf()

        Returns:
            Formatted text for LLM input
        """
        if not preprocessed_data.get("preprocessing_enabled", False):
            # Fallback to raw text extraction
            return self._extract_raw_text(preprocessed_data["raw_text_path"])

        components = preprocessed_data.get("components", {})
        formatted_sections = []

        # Add structured table data if available
        if "table_stitching" in components:
            tables = components["table_stitching"].get("stitched_tables", [])
            if tables:
                formatted_sections.append("=== STRUCTURED TABLES ===")
                for i, table in enumerate(tables):
                    formatted_sections.append(f"Table {i+1}:")
                    formatted_sections.append(table.get("html", ""))
                    formatted_sections.append("")

        # Add segmented text blocks
        if "segmentation" in components:
            text_blocks = components["segmentation"].get("text_blocks", [])
            if text_blocks:
                formatted_sections.append("=== TEXT CONTENT ===")
                for block in text_blocks:
                    formatted_sections.append(block.get("text", ""))
                formatted_sections.append("")

        # Add narrative corrections if available
        if "narrative_linking" in components:
            corrections = components["narrative_linking"].get("corrections", [])
            if corrections:
                formatted_sections.append("=== NARRATIVE CORRECTIONS ===")
                for correction in corrections:
                    formatted_sections.append(correction.get("text", ""))
                formatted_sections.append("")

        return "\n".join(formatted_sections)

    def _extract_raw_text(self, pdf_path: str) -> str:
        """Fallback to raw text extraction when preprocessing is disabled."""
        if not PYPDF2_AVAILABLE:
            return f"Could not extract text from {pdf_path} - PyPDF2 not available"

        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            # Ultimate fallback
            return f"Could not extract text from {pdf_path}: {str(e)}"

    def get_preprocessing_metadata(
        self, preprocessed_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract metadata for logging and evaluation.

        Args:
            preprocessed_data: Output from preprocess_pdf()

        Returns:
            Metadata dictionary for database logging
        """
        if not preprocessed_data.get("preprocessing_enabled", False):
            return {"preprocessing_enabled": False}

        metadata = {
            "preprocessing_enabled": True,
            "success": preprocessed_data.get("success", False),
            "total_processing_time": preprocessed_data.get("total_processing_time", 0),
            "processing_times": preprocessed_data.get("processing_times", {}),
            "config": preprocessed_data.get("preprocessing_config", {}),
        }

        # Add component-specific metadata
        components = preprocessed_data.get("components", {})

        if "segmentation" in components:
            seg_data = components["segmentation"]
            metadata.update(
                {
                    "text_blocks_extracted": len(seg_data.get("text_blocks", [])),
                    "table_regions_detected": len(seg_data.get("table_regions", [])),
                }
            )

        if "table_detection" in components:
            table_data = components["table_detection"]
            metadata.update(
                {
                    "tables_detected": len(table_data.get("tables", [])),
                    "average_table_confidence": self._calculate_average_confidence(
                        table_data.get("tables", [])
                    ),
                }
            )

        if "table_stitching" in components:
            stitch_data = components["table_stitching"]
            metadata.update(
                {
                    "tables_stitched": len(stitch_data.get("stitched_tables", [])),
                }
            )

        return metadata

    def _calculate_average_confidence(self, tables: List[Dict]) -> float:
        """Calculate average confidence score for detected tables."""
        if not tables:
            return 0.0

        confidences = [table.get("confidence", 0.0) for table in tables]
        return sum(confidences) / len(confidences)
