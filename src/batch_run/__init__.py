"""
LLM vs Rule-Based Comparison Module

This module provides functions for analyzing LLM extraction results
against rule-based scraper data.

Main Functions:
    - load_llm_data: Load extracted LLM run data
    - load_rule_based_data: Load rule-based comparison data
    - analyze_llm_vs_rule_based: Compare LLM results with rule-based
    - categorize_discrepancies: Categorize types of discrepancies
"""

from .loader import (
    load_llm_data,
    load_rule_based_data,
    parse_week_year_from_filename,
)

from .analyzer import (
    analyze_llm_vs_rule_based,
    categorize_discrepancies,
    analyze_single_week,
    create_summary_statistics,
)

__all__ = [
    "load_llm_data",
    "load_rule_based_data",
    "parse_week_year_from_filename",
    "analyze_llm_vs_rule_based",
    "categorize_discrepancies",
    "analyze_single_week",
    "create_summary_statistics",
]
