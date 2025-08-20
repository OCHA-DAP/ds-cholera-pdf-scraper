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
  
  # Extract with GPT-4o using specific prompt version
  python scripts/run_extraction.py --model gpt-4o --prompt-version v1.1.2
  
  # Extract from custom PDF
  python scripts/run_extraction.py --model gemini-pro --pdf path/to/your.pdf
  
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

    print(f"🔧 Command: {' '.join(cmd_parts)}")
    print("-" * 60)

    # Execute from project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    os.system(" ".join(cmd_parts))


if __name__ == "__main__":
    main()
