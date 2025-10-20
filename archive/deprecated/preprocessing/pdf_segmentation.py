"""
PDF Segmentation Engine - Uses PyMuPDF (fitz) for layout analysis.
Extracts text blocks, bounding boxes, and identifies table regions.
Based on recommendation: PyMuPDF (fitz) for extracting text blocks, bounding boxes, page geometry.
"""

import logging
import time
from typing import Any, Dict, List

# Required imports - fail if not available
try:
    import fitz  # PyMuPDF
except ImportError as e:
    raise ImportError(
        "PyMuPDF (fitz) is required for PDF segmentation. "
        "Install with: pip install PyMuPDF"
    ) from e

logger = logging.getLogger(__name__)


class PDFSegmentationEngine:
    """
    PDF segmentation using PyMuPDF (fitz) for layout analysis.
    Identifies text blocks, table regions, and document structure.
    Follows recommended approach: PyMuPDF for text blocks and bounding boxes.
    """

    def __init__(self):
        """Initialize the PDF segmentation engine with PyMuPDF."""
        self.fitz = fitz
        logger.info("PDFSegmentationEngine initialized with PyMuPDF")

    def segment_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Segment PDF using PyMuPDF layout analysis.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with segmented content including text blocks and table regions
        """
        try:
            doc = self.fitz.open(pdf_path)
            result = {
                "text_blocks": [],
                "table_regions": [],
                "pages": len(doc),
                "metadata": {"method": "pymupdf", "version": self.fitz.__version__},
            }

            for page_num, page in enumerate(doc):
                page_data = self._segment_page_with_pymupdf(page, page_num)
                result["text_blocks"].extend(page_data["text_blocks"])
                result["table_regions"].extend(page_data["table_regions"])

            doc.close()
            logger.info(
                f"Segmented {result['pages']} pages, "
                f"found {len(result['text_blocks'])} text blocks, "
                f"{len(result['table_regions'])} table regions"
            )
            return result

        except Exception as e:
            logger.error(f"PyMuPDF segmentation failed: {e}")
            raise

    def _segment_page_with_pymupdf(self, page, page_num: int) -> Dict[str, Any]:
        """
        Segment page using PyMuPDF's detailed layout analysis.

        Args:
            page: PyMuPDF page object
            page_num: Page number

        Returns:
            Page segmentation data with text blocks and table regions
        """
        text_blocks = []
        table_regions = []

        # Get text blocks with detailed layout information
        blocks = page.get_text("dict")["blocks"]

        # Also get tables using PyMuPDF's native table detection as backup
        try:
            pymupdf_tables = page.find_tables()
            for table in pymupdf_tables:
                table_bbox = table.bbox
                table_regions.append(
                    {
                        "bbox": list(table_bbox),
                        "page": page_num,
                        "type": "table_pymupdf",
                        "confidence": 0.8,  # PyMuPDF table detection confidence
                        "method": "pymupdf_native",
                    }
                )
        except Exception as e:
            logger.debug(f"PyMuPDF table detection failed on page {page_num}: {e}")

        for block in blocks:
            if "lines" in block:  # Text block
                text_content = ""
                for line in block["lines"]:
                    for span in line["spans"]:
                        text_content += span["text"] + " "

                if text_content.strip():
                    text_blocks.append(
                        {
                            "text": text_content.strip(),
                            "bbox": block["bbox"],
                            "page": page_num,
                            "type": "text",
                            "font_info": self._extract_font_info(block),
                        }
                    )

            elif "image" in block:  # Image block - might contain tables
                # Mark image regions as potential table candidates for TATR
                table_regions.append(
                    {
                        "bbox": block["bbox"],
                        "page": page_num,
                        "type": "image_table_candidate",
                        "confidence": 0.3,  # Lower confidence for image regions
                        "method": "pymupdf_image",
                    }
                )

        return {"text_blocks": text_blocks, "table_regions": table_regions}

    def _extract_font_info(self, block) -> Dict[str, Any]:
        """Extract font information from text block for classification."""
        font_info = {"sizes": [], "flags": [], "fonts": []}

        for line in block.get("lines", []):
            for span in line.get("spans", []):
                font_info["sizes"].append(span.get("size", 0))
                font_info["flags"].append(span.get("flags", 0))
                font_info["fonts"].append(span.get("font", ""))

        return {
            "avg_size": (
                sum(font_info["sizes"]) / len(font_info["sizes"])
                if font_info["sizes"]
                else 0
            ),
            "dominant_font": (
                max(set(font_info["fonts"]), key=font_info["fonts"].count)
                if font_info["fonts"]
                else ""
            ),
            "has_bold": any(flag & 2**4 for flag in font_info["flags"]),  # Bold flag
        }

    def _fallback_segmentation(self, pdf_path: str) -> Dict[str, Any]:
        """
        Fallback segmentation using simple text extraction.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Basic segmentation data
        """
        if not PYPDF2_AVAILABLE:
            logger.error("PyPDF2 not available for fallback segmentation")
            return {
                "text_blocks": [
                    {
                        "text": f"Could not process {pdf_path}",
                        "page": 0,
                        "type": "error",
                    }
                ],
                "table_regions": [],
                "pages": 0,
                "metadata": {"error": "No PDF processing libraries available"},
            }

        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text_blocks = []

                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        text_blocks.append(
                            {
                                "text": text.strip(),
                                "bbox": [0, 0, 100, 100],  # Placeholder bbox
                                "page": page_num,
                                "type": "text",
                            }
                        )

                return {
                    "text_blocks": text_blocks,
                    "table_regions": [],
                    "pages": len(reader.pages),
                    "metadata": {"fallback": True},
                }

        except Exception as e:
            logger.error(f"Fallback segmentation failed: {e}")
            return {
                "text_blocks": [
                    {
                        "text": f"Could not process {pdf_path}",
                        "page": 0,
                        "type": "error",
                    }
                ],
                "table_regions": [],
                "pages": 0,
                "metadata": {"error": str(e)},
            }
