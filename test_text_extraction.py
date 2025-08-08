#!/usr/bin/env python3
"""
Test script to try text-based extraction instead of vision model.
"""

import json
import logging
import os
import sys
from pathlib import Path

import pdfplumber
import PyPDF2

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("test_text_extraction.log"),
        ],
    )


def extract_text_with_pdfplumber(pdf_path: Path) -> str:
    """Extract text from PDF using pdfplumber (better for tables)."""
    logger = logging.getLogger(__name__)
    text_content = ""

    with pdfplumber.open(pdf_path) as pdf:
        logger.info(f"PDF has {len(pdf.pages)} pages")

        for page_num, page in enumerate(pdf.pages, 1):
            logger.info(f"Processing page {page_num}")
            page_text = page.extract_text()
            if page_text:
                text_content += f"\n\n=== PAGE {page_num} ===\n\n"
                text_content += page_text

                # Also try to extract tables specifically
                tables = page.extract_tables()
                if tables:
                    logger.info(f"Found {len(tables)} tables on page {page_num}")
                    for i, table in enumerate(tables):
                        text_content += f"\n\n--- TABLE {i+1} ON PAGE {page_num} ---\n"
                        for row in table:
                            if row:  # Skip empty rows
                                text_content += (
                                    " | ".join(
                                        str(cell) if cell else "" for cell in row
                                    )
                                    + "\n"
                                )

    return text_content


def main():
    """Main execution function."""
    setup_logging(verbose=True)
    logger = logging.getLogger(__name__)

    # Use the same PDF path as the vision model test
    pdf_path = Path(
        "/Users/zackarno/Library/CloudStorage/GoogleDrive-Zachary.arno@humdata.org/Shared drives/Data Science/CERF Anticipatory Action/Cholera - General/WHO_bulletins_historical/Week_28__7_-_13_July_2025.pdf"
    )

    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return

    logger.info(f"Extracting text from PDF: {pdf_path}")
    logger.info(f"File size: {pdf_path.stat().st_size / 1024:.1f} KB")

    try:
        # Extract text content
        text_content = extract_text_with_pdfplumber(pdf_path)

        # Save extracted text for inspection
        output_path = Path("extracted_text.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text_content)

        logger.info(f"Extracted text saved to: {output_path}")
        logger.info(f"Total text length: {len(text_content):,} characters")

        # Show a preview
        print("\n" + "=" * 60)
        print("TEXT EXTRACTION PREVIEW (first 2000 chars):")
        print("=" * 60)
        print(text_content[:2000])
        print("...")
        print("=" * 60)

        # Count potential table rows (lines with country names)
        lines = text_content.split("\n")
        potential_data_lines = []

        countries = [
            "Angola",
            "Benin",
            "Burkina Faso",
            "Burundi",
            "Cameroon",
            "Chad",
            "Democratic Republic of the Congo",
            "Mozambique",
            "Niger",
            "Nigeria",
            "Somalia",
            "South Sudan",
            "Sudan",
            "Uganda",
            "Zimbabwe",
        ]

        for line_num, line in enumerate(lines, 1):
            for country in countries:
                if country in line:
                    potential_data_lines.append((line_num, line.strip()))
                    break

        logger.info(
            f"Found {len(potential_data_lines)} lines potentially containing country data"
        )

        if potential_data_lines:
            print("\nPOTENTIAL DATA LINES:")
            for line_num, line in potential_data_lines[:10]:  # Show first 10
                print(f"Line {line_num}: {line}")
            if len(potential_data_lines) > 10:
                print(f"... and {len(potential_data_lines) - 10} more lines")

    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        import traceback

        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
