#!/usr/bin/env python3
"""
Test script for LLM-based PDF extraction with a single file.

This script allows you to test the LLM extraction functionality
with a single PDF file from your local storage.
"""

import json
import logging
import os
import sys
from pathlib import Path

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
            logging.FileHandler("test_extraction.log"),
        ],
    )


def test_single_pdf(pdf_path: str, verbose: bool = False) -> None:
    """
    Test LLM extraction with a single PDF file.

    Args:
        pdf_path: Path to the PDF file to test
        verbose: Enable verbose logging
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    # Check if PDF file exists
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return

    if not pdf_file.suffix.lower() == ".pdf":
        logger.error(f"File is not a PDF: {pdf_path}")
        return

    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        logger.info("Please set your OpenAI API key:")
        logger.info("export OPENAI_API_KEY='your-api-key-here'")
        return

    logger.info(f"Testing LLM extraction with: {pdf_file.name}")
    logger.info(f"File size: {pdf_file.stat().st_size / 1024:.1f} KB")

    try:
        # Initialize the extractor
        extractor = LLMExtractor(api_key=api_key, temperature=0.1)

        # Process the PDF
        logger.info("Starting PDF processing...")
        result = extractor.process_pdf_file(pdf_file)

        # Display results
        logger.info("✅ PDF processing completed successfully!")
        print("\n" + "=" * 60)
        print("EXTRACTION RESULTS")
        print("=" * 60)

        # Pretty print the JSON result
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # Save results to file
        output_file = Path("test_extraction_result.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {output_file}")

        # Summary
        if "records" in result:
            num_records = len(result["records"])
            logger.info(f"📊 Extracted {num_records} record(s)")

            if num_records > 0:
                # Show field summary
                sample_record = result["records"][0]
                logger.info("📋 Fields found:")
                for field, value in sample_record.items():
                    logger.info(f"  - {field}: {type(value).__name__}")

        print("\n" + "=" * 60)

    except Exception as e:
        logger.error(f"❌ Error during PDF processing: {e}")
        if verbose:
            import traceback

            logger.error(traceback.format_exc())


def main():
    """Main function with command line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test LLM-based PDF extraction with a single file"
    )
    parser.add_argument("pdf_path", help="Path to the PDF file to process")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    test_single_pdf(args.pdf_path, args.verbose)


if __name__ == "__main__":
    main()
