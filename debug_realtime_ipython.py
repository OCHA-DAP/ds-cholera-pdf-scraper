#!/usr/bin/env python3
"""
IPython-based debugger for realtime comparison.

Usage:
    uv run python debug_realtime_ipython.py

This uses IPython's debugger which provides:
- Tab completion
- Syntax highlighting
- Better history
- More intuitive interface

Commands (same as pdb plus):
    <Tab>           - Auto-complete
    ?               - Help
    ??              - Detailed help
    %debug          - Post-mortem debugging
    h               - Help on commands
"""

import sys
import importlib.util
from pathlib import Path

# Add repo to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

# Try to import IPython debugger, fall back to pdb if not available
try:
    from IPython.core.debugger import set_trace
    debugger = 'ipdb'
except ImportError:
    from pdb import set_trace
    debugger = 'pdb'
    print("⚠️  IPython not found, using standard pdb")
    print("   For better experience, run: uv add --dev ipython\n")

# Import directly from the file since it starts with a number
spec = importlib.util.spec_from_file_location(
    "realtime_comparison",
    repo_root / "book_cholera_scraping" / "04_realtime_comparison.py"
)
rtc = importlib.util.module_from_spec(spec)
sys.modules["realtime_comparison"] = rtc
spec.loader.exec_module(rtc)

# Now we can use the module
BlobExtractionLoader = rtc.BlobExtractionLoader
load_and_compare = rtc.load_and_compare


def main():
    """
    Interactive debugging session.

    The script will pause at the set_trace() call, allowing you to:
    1. Inspect variables
    2. Step through code
    3. Test expressions
    """

    print(f"=== Starting {debugger} debugger ===\n")
    print("Quick tips:")
    print("  n           - Next line")
    print("  s           - Step into function")
    print("  c           - Continue")
    print("  p <expr>    - Print/evaluate expression")
    print("  ll          - List full function code")
    print("  w           - Where am I? (stack trace)")
    print()

    # Initialize - will pause here
    print("Initializing BlobExtractionLoader...")
    set_trace()  # <-- Debugger starts here

    loader = BlobExtractionLoader(stage='dev')

    # You can inspect loader here
    # Try: p loader.stage, p loader.container, etc.

    # List available extractions
    rb_list = loader.list_available_rule_based_extractions()
    llm_list = loader.list_available_llm_extractions()

    # Pick a test week
    if len(rb_list) > 0:
        test_week = rb_list.iloc[0]['week']
        test_year = rb_list.iloc[0]['year']

        print(f"\nTesting with Week {test_week}, {test_year}")

        # Load data
        rb_df = loader.load_rule_based_extraction(week=test_week, year=test_year)
        llm_df = loader.load_llm_extraction(week=test_week, year=test_year, model='gpt-5')

        # Now you can inspect the dataframes
        # Try: p rb_df.shape, p rb_df.columns, p rb_df.head()

    print("\n✅ Debug session complete!")


if __name__ == '__main__':
    main()
