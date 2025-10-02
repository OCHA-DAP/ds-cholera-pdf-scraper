"""
Batch Run Analysis Module

This module provides functions for analyzing batch extraction results
against baseline rule-based scraper data.

Main Functions:
    - load_batch_data: Load extracted batch run data
    - load_baseline_data: Load baseline comparison data
    - analyze_batch_vs_baseline: Compare batch results with baseline
    - categorize_discrepancies: Categorize types of discrepancies
"""

from .loader import (
    load_batch_data,
    load_baseline_data,
    parse_week_year_from_filename,
)

from .analyzer import (
    analyze_batch_vs_baseline,
    categorize_discrepancies,
    analyze_single_week,
    create_summary_statistics,
)

__all__ = [
    "load_batch_data",
    "load_baseline_data",
    "parse_week_year_from_filename",
    "analyze_batch_vs_baseline",
    "categorize_discrepancies",
    "analyze_single_week",
    "create_summary_statistics",
]
