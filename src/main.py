#!/usr/bin/env python3
"""
Main entry point for the cholera PDF scraper project.

This module provides a command-line interface for running different
components of the cholera PDF extraction pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path

from config import Config
from llm_extract import LLMExtractor
from parse_output import OutputParser
from compare import DataComparator


def setup_logging(level: str = "INFO"):
    """Set up logging configuration."""
    log_level = getattr(logging, level.upper())
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(Config.LOGS_DIR / "cholera_scraper.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def extract_from_pdf(pdf_path: str, output_path: str = None) -> None:
    """
    Extract data from a single PDF file.

    Args:
        pdf_path: Path to the PDF file
        output_path: Optional path for output CSV
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Extracting data from PDF: {pdf_path}")

    # Validate API key
    if not Config.OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not configured")
        sys.exit(1)

    try:
        # Initialize components
        extractor = LLMExtractor(Config.OPENAI_API_KEY)
        parser = OutputParser()

        # Process PDF
        extraction_result = extractor.process_pdf_file(Path(pdf_path))
        df = parser.parse_single_extraction(extraction_result)

        # Save output
        if output_path:
            output_file = Path(output_path)
        else:
            output_file = Config.OUTPUTS_DIR / f"extracted_{Path(pdf_path).stem}.csv"

        df.to_csv(output_file, index=False)
        logger.info(f"Extraction completed. Output saved to {output_file}")

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        sys.exit(1)


def compare_extractions(
    llm_path: str, baseline_path: str, report_path: str = None
) -> None:
    """
    Compare LLM extractions against baseline data.

    Args:
        llm_path: Path to LLM extraction results
        baseline_path: Path to baseline data
        report_path: Optional path for comparison report
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting extraction comparison")

    try:
        comparator = DataComparator(tolerance=Config.NUMERICAL_TOLERANCE)

        if report_path:
            output_file = Path(report_path)
        else:
            output_file = Config.OUTPUTS_DIR / "comparison_report.json"

        report = comparator.generate_comparison_report(
            llm_path, baseline_path, str(output_file)
        )

        logger.info(f"Comparison completed. Report saved to {output_file}")

        # Print summary
        summary = report.get("summary", {})
        print(f"\nComparison Summary:")
        print(f"LLM Records: {summary.get('llm_total_records', 'N/A')}")
        print(f"Baseline Records: {summary.get('baseline_total_records', 'N/A')}")
        print(f"Aligned Records: {summary.get('aligned_records', 'N/A')}")

    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        sys.exit(1)


def validate_config() -> None:
    """Validate project configuration."""
    validation = Config.validate_config()

    if validation["valid"]:
        print("✓ Configuration is valid")
    else:
        print("✗ Configuration issues found:")
        for issue in validation["issues"]:
            print(f"  - {issue}")
        sys.exit(1)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Cholera PDF Scraper - LLM-based data extraction pipeline"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract data from PDF")
    extract_parser.add_argument("pdf_path", help="Path to PDF file")
    extract_parser.add_argument("--output", help="Output CSV file path")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare extractions")
    compare_parser.add_argument("llm_path", help="Path to LLM extraction results")
    compare_parser.add_argument("baseline_path", help="Path to baseline data")
    compare_parser.add_argument("--report", help="Output report file path")

    # Config command
    subparsers.add_parser("validate-config", help="Validate configuration")

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.log_level)

    if args.command == "extract":
        extract_from_pdf(args.pdf_path, args.output)
    elif args.command == "compare":
        compare_extractions(args.llm_path, args.baseline_path, args.report)
    elif args.command == "validate-config":
        validate_config()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
