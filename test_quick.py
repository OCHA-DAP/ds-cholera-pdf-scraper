#!/usr/bin/env python3
"""
Quick test script for LLM PDF extraction.
Usage: python test_quick.py /path/to/your/file.pdf
"""

import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from llm_extract import LLMExtractor


def main():
    if len(sys.argv) != 2:
        print("Usage: python test_quick.py /path/to/your/file.pdf")
        return

    pdf_path = Path(sys.argv[1])

    if not pdf_path.exists():
        print(f"❌ File not found: {pdf_path}")
        return

    if not pdf_path.suffix.lower() == ".pdf":
        print(f"❌ Not a PDF file: {pdf_path}")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Set OPENAI_API_KEY environment variable")
        return

    print(f"🔍 Processing: {pdf_path.name}")

    try:
        extractor = LLMExtractor(api_key)
        result = extractor.process_pdf_file(pdf_path)

        print("✅ Success!")
        print(json.dumps(result, indent=2))

        # Save result
        with open("result.json", "w") as f:
            json.dump(result, f, indent=2)
        print("💾 Saved to result.json")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
