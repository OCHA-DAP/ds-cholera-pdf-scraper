#!/usr/bin/env python3
"""
CLI for running cholera PDF extraction with different models.
Convenient wrapper around the LLM text extraction system.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.llm_client import POPULAR_MODELS


def main():
    parser = argparse.ArgumentParser(
        description="Run cholera PDF extraction with different models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract with Claude 3.5 Sonnet
  python scripts/run_extraction.py --model claude-3.5-sonnet
  
  # Extract with pdfplumber preprocessing + GPT-4o
  python scripts/run_extraction.py --model gpt-4o --preprocessor pdfplumber
  
  # Extract with blank-treatment preprocessing + GPT-4o
  python scripts/run_extraction.py --model gpt-4o --preprocessor blank-treatment
  
  # Extract from custom PDF with blank-treatment
  python scripts/run_extraction.py --model gemini-pro --pdf path/to/your.pdf --preprocessor blank-treatment
  
  # Apply JSON corrections to existing extracted data (sample mode)
  python scripts/run_extraction.py --model gpt-4o --preprocessor json-correction
  
  # Process full dataset in batches
  python scripts/run_extraction.py --model gpt-4o --preprocessor json-correction --run-mode full
  
  # List available models
  python scripts/run_extraction.py --list-models

Available model shortcuts:
"""
        + "\n".join(f"  {k:20} → {v}" for k, v in list(POPULAR_MODELS.items())[:10]),
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Model to test (use shortcuts like 'claude-3.5-sonnet' or full IDs)",
    )
    parser.add_argument(
        "--pdf", type=str, help="Path to PDF file (optional, uses default test PDF)"
    )
    parser.add_argument(
        "--prompt-version",
        "-p",
        type=str,
        help="Prompt version to use (e.g., v1.1.2). Defaults to current version if not specified.",
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List all available model shortcuts"
    )
    parser.add_argument(
        "--preprocessor",
        type=str,
        choices=[
            "pdfplumber",
            "blank-treatment",
            "table-focused",
            "none-pdf-upload",
            "self-code",
            "json-correction",
        ],
        help="Use preprocessing before LLM (pdfplumber: table extraction, blank-treatment: standardize blank fields, table-focused: WHO surveillance table extraction + LLM correction, none-pdf-upload: direct PDF upload without text extraction, self-code: let LLM write its own preprocessing code, json-correction: apply LLM corrections to existing JSON data)",
    )
    parser.add_argument(
        "--json-path",
        type=str,
        help="Path to JSON file for correction (used with json-correction preprocessor)",
        default="outputs/enhanced_extraction/master_surveillance_data.json"
    )
    parser.add_argument(
        "--run-mode",
        type=str,
        choices=["sample", "full"],
        default="sample",
        help="Run mode for json-correction: 'sample' (20 random PDFs) or 'full' (process all data in batches)"
    )

    args = parser.parse_args()

    if args.list_models:
        print("🤖 Available Model Shortcuts:")
        for nickname, full_id in POPULAR_MODELS.items():
            print(f"  {nickname:25} → {full_id}")
        print(f"\nTotal: {len(POPULAR_MODELS)} predefined models")
        print(
            "\nUse any of these with: python scripts/run_extraction.py --model <shortcut>"
        )
        return

    if not args.model:
        print("❌ Please specify a model with --model")
        print("💡 Use --list-models to see available options")
        return

    # Build command (run from project root)
    cmd_parts = ["python", "src/llm_text_extract.py", "--model", args.model]

    if args.pdf:
        cmd_parts.extend(["--pdf-path", args.pdf])

    if args.prompt_version:
        cmd_parts.extend(["--prompt-version", args.prompt_version])

    if args.preprocessor:
        cmd_parts.extend(["--preprocessor", args.preprocessor])

    if args.json_path and args.preprocessor == "json-correction":
        cmd_parts.extend(["--json-path", args.json_path])

    if args.run_mode and args.preprocessor == "json-correction":
        cmd_parts.extend(["--run-mode", args.run_mode])

    # Show what we're running
    print(f"🚀 Running extraction with model: {args.model}")
    if args.pdf:
        print(f"📄 PDF: {args.pdf}")
    else:
        print("📄 Using default test PDF")
    if args.prompt_version:
        print(f"📝 Prompt version: {args.prompt_version}")
    else:
        print("📝 Using current prompt version")
    if args.preprocessor:
        print(f"🔧 Preprocessor: {args.preprocessor}")
        if args.preprocessor == "json-correction":
            if args.json_path:
                print(f"📄 JSON file: {args.json_path}")
            print(f"🎯 Run mode: {args.run_mode}")

    print(f"🔧 Command: {' '.join(cmd_parts)}")
    print("-" * 60)

    # Execute from project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    os.system(" ".join(cmd_parts))


if __name__ == "__main__":
    main()
