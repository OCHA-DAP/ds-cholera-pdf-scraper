#!/usr/bin/env python3
"""
Simple interactive test for LLM PDF extraction.

Run this script and it will prompt you for a PDF file path.
"""

import json
import logging
import os
import sys
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from src.llm_extract import LLMExtractor


def main():
    """Interactive test for PDF extraction."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    print("🔍 LLM PDF Extraction Test")
    print("=" * 40)

    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return

    # Get PDF file path
    while True:
        pdf_path = input("\n📁 Enter the path to your PDF file: ").strip()

        if not pdf_path:
            print("❌ Please enter a file path")
            continue

        pdf_file = Path(pdf_path)

        if not pdf_file.exists():
            print(f"❌ File not found: {pdf_path}")
            continue

        if not pdf_file.suffix.lower() == ".pdf":
            print(f"❌ File is not a PDF: {pdf_path}")
            continue

        break

    print(f"\n✅ Found PDF: {pdf_file.name}")
    print(f"📊 File size: {pdf_file.stat().st_size / 1024:.1f} KB")

    # Confirm processing
    save_csv = input("\n💾 Save results to CSV? (y/n): ").strip().lower() == "y"
    confirm = input("🚀 Process this PDF? (y/n): ").strip().lower()
    if confirm != "y":
        print("👋 Cancelled")
        return

    try:
        # Initialize extractor
        print("\n🤖 Initializing LLM extractor...")
        extractor = LLMExtractor(api_key=api_key, temperature=0.1)

        # Process PDF
        print("📤 Uploading PDF to OpenAI...")
        print("🧠 Extracting data with LLM...")
        result = extractor.process_pdf_file(pdf_file, save_csv=save_csv)

        # Display results
        print("\n" + "=" * 60)
        print("✅ EXTRACTION COMPLETED!")
        print("=" * 60)

        # Pretty print results
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # Save to JSON file
        output_file = Path("extraction_result.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\n💾 JSON results saved to: {output_file}")

        if save_csv:
            csv_file = Path(f"{pdf_file.stem}_extracted.csv")
            print(f"📊 CSV results saved to: {csv_file}")

        # Summary
        if "records" in result:
            num_records = len(result["records"])
            print(f"\n📊 Summary: {num_records} record(s) extracted")

            if num_records > 0:
                sample = result["records"][0]
                print("📋 Sample record fields:")
                for field in sample.keys():
                    print(f"  • {field}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"Extraction failed: {e}")


if __name__ == "__main__":
    main()
