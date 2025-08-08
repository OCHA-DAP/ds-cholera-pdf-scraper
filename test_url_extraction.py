#!/usr/bin/env python3
"""
Test script to download a PDF from URL and test LLM extraction.
"""

import json
import logging
import os
import sys
from pathlib import Path

import requests

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_extract import LLMExtractor


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("test_url_extraction.log"),
        ],
    )


def download_pdf_from_url(url: str, filename: str) -> Path:
    """Download PDF from URL to local file."""
    logger = logging.getLogger(__name__)
    local_path = Path(filename)

    logger.info(f"Downloading PDF from: {url}")

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(
            f"Downloaded {local_path.stat().st_size / 1024:.1f} KB to {local_path}"
        )
        return local_path

    except Exception as e:
        logger.error(f"Failed to download PDF: {e}")
        raise


def main():
    """Main function."""
    setup_logging(verbose=True)
    logger = logging.getLogger(__name__)

    # Test with the current PDF (Week 28, 2025) - local file
    pdf_path = Path(
        "/Users/zackarno/Library/CloudStorage/GoogleDrive-Zachary.arno@humdata.org/Shared drives/Data Science/CERF Anticipatory Action/Cholera - General/WHO_bulletins_historical/Week_28__7_-_13_July_2025.pdf"
    )

    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        return

    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return

    try:
        logger.info(f"Using local PDF: {pdf_path}")
        logger.info(f"File size: {pdf_path.stat().st_size / 1024:.1f} KB")

        # Initialize extractor
        logger.info("Initializing LLM extractor...")
        extractor = LLMExtractor(api_key=api_key, temperature=0.1)

        # Process the PDF with CSV output
        logger.info("Processing PDF with LLM...")
        result = extractor.process_pdf_file(pdf_path, save_csv=True)

        # Display results
        print("\n" + "=" * 60)
        print("✅ EXTRACTION COMPLETED!")
        print("=" * 60)

        # Pretty print results
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # Save to file
        output_file = Path("url_extraction_result.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {output_file}")

        # Summary
        if "records" in result:
            num_records = len(result["records"])
            logger.info(f"📊 Extracted {num_records} record(s)")

            if num_records > 0:
                sample = result["records"][0]
                logger.info("📋 Sample record fields:")
                for field in sample.keys():
                    logger.info(f"  • {field}")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback

        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
