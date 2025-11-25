#!/usr/bin/env python3
"""
Debug script for realtime comparison.

Usage:
    uv run python debug_realtime_comparison.py

This will:
1. Load the module
2. Set up a test case
3. Drop you into the Python debugger (pdb)

Debugger commands:
    n (next)        - Execute next line
    s (step)        - Step into function
    c (continue)    - Continue execution
    p <var>         - Print variable value
    pp <var>        - Pretty print variable
    l (list)        - Show current code
    w (where)       - Show stack trace
    u (up)          - Move up stack frame
    d (down)        - Move down stack frame
    b <line>        - Set breakpoint at line
    cl <num>        - Clear breakpoint
    q (quit)        - Quit debugger

Example workflow:
    >>> n                           # Step to next line
    >>> p loader                    # Print loader object
    >>> p loader.stage              # Print specific attribute
    >>> s                           # Step into function
    >>> pp rb_list.head()          # Pretty print dataframe head
    >>> c                           # Continue to next breakpoint
"""

import sys
import pdb
from pathlib import Path

# Add repo to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

from book_cholera_scraping.realtime_comparison_04 import (
    BlobExtractionLoader,
    load_and_compare
)

def debug_blob_loader():
    """Debug the BlobExtractionLoader step by step."""

    print("=== Debugging BlobExtractionLoader ===\n")

    # Set breakpoint here - execution will pause
    pdb.set_trace()

    # Initialize loader
    loader = BlobExtractionLoader(stage='dev')

    # List rule-based extractions
    rb_list = loader.list_available_rule_based_extractions()

    # List LLM extractions
    llm_list = loader.list_available_llm_extractions()

    # Pick a test week
    if len(rb_list) > 0:
        test_week = rb_list.iloc[0]['week']
        test_year = rb_list.iloc[0]['year']

        # Load rule-based data
        rb_df = loader.load_rule_based_extraction(week=test_week, year=test_year)

        # Load LLM data
        llm_df = loader.load_llm_extraction(week=test_week, year=test_year, model='gpt-5')

    print("\n✅ Debugging complete!")
    return loader, rb_list, llm_list


def debug_full_comparison():
    """Debug the full comparison workflow."""

    print("=== Debugging Full Comparison ===\n")

    # Set breakpoint
    pdb.set_trace()

    # Run comparison
    results = load_and_compare(
        stage='dev',
        weeks=[37],  # Adjust week as needed
        year=2025,
        model='gpt-5',
        verbose=True
    )

    print("\n✅ Comparison complete!")
    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Debug realtime comparison')
    parser.add_argument(
        '--mode',
        choices=['loader', 'full'],
        default='loader',
        help='What to debug: loader (blob loading) or full (complete comparison)'
    )

    args = parser.parse_args()

    print("Python Debugger (pdb) Quick Reference:")
    print("  n (next)    - Next line")
    print("  s (step)    - Step into function")
    print("  c (continue) - Continue execution")
    print("  p <var>     - Print variable")
    print("  l (list)    - Show code")
    print("  q (quit)    - Quit\n")

    if args.mode == 'loader':
        debug_blob_loader()
    else:
        debug_full_comparison()
