#!/usr/bin/env python3
"""
Prompt comparison utilities for analyzing LLM extraction performance across versions.
Provides easy access to discrepancies, accuracy metrics, and multi-version comparison tools.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.backfill_accuracy_metrics import perform_discrepancy_analysis


def get_discrepancies_by_prompt_version(
    prompt_version: str, outputs_dir: str = "outputs", data_dir: str = "data"
) -> Optional[pd.DataFrame]:
    """
    Get discrepancies DataFrame for a specific prompt version.

    Args:
        prompt_version: Prompt version (e.g., "v1.1.1")
        outputs_dir: Directory containing extraction results
        data_dir: Directory containing baseline data

    Returns:
        DataFrame with discrepancies, or None if files not found
    """
    # Check if extraction file exists
    extraction_file = f"text_extracted_data_prompt_{prompt_version}.csv"
    extraction_path = Path(outputs_dir) / extraction_file

    if not extraction_path.exists():
        print(f"❌ Extraction file not found: {extraction_path}")
        return None

    # Check if baseline data exists
    baseline_path = Path(data_dir) / "final_data_for_powerbi_with_kpi.csv"
    if not baseline_path.exists():
        print(f"❌ Baseline data not found: {baseline_path}")
        return None

    print(f"🔍 Loading discrepancies for prompt {prompt_version}...")

    try:
        # Load data
        llm_data = pd.read_csv(extraction_path)
        baseline_df = pd.read_csv(baseline_path)

        # Filter baseline to Week 28, 2025 (same as QMD)
        baseline_week28 = baseline_df[
            (baseline_df["Year"] == 2025) & (baseline_df["WeekNumber"] == 28)
        ].copy()

        print(f"📊 LLM data: {len(llm_data)} records")
        print(f"📊 Baseline data: {len(baseline_week28)} records")

        # Perform discrepancy analysis
        discrepancies_df, llm_common, llm_only_df, baseline_only_df = (
            perform_discrepancy_analysis(llm_data, baseline_week28)
        )

        print(f"✅ Found {len(discrepancies_df)} discrepant records")
        print(f"   📊 Records compared: {len(llm_common)}")
        print(f"   📊 LLM-only records: {len(llm_only_df)}")
        print(f"   📊 Baseline-only records: {len(baseline_only_df)}")

        return discrepancies_df

    except Exception as e:
        print(f"❌ Error analyzing prompt {prompt_version}: {e}")
        return None


def get_analysis_summary_by_prompt_version(
    prompt_version: str, outputs_dir: str = "outputs", data_dir: str = "data"
) -> Optional[Dict[str, Any]]:
    """
    Get complete analysis summary for a prompt version.

    Returns:
        Dictionary with discrepancies_df, llm_common, llm_only_df, baseline_only_df
    """
    # Check if extraction file exists
    extraction_file = f"text_extracted_data_prompt_{prompt_version}.csv"
    extraction_path = Path(outputs_dir) / extraction_file

    if not extraction_path.exists():
        print(f"❌ Extraction file not found: {extraction_path}")
        return None

    try:
        # Load data
        llm_data = pd.read_csv(extraction_path)
        baseline_df = pd.read_csv(
            Path(data_dir) / "final_data_for_powerbi_with_kpi.csv"
        )

        # Filter baseline to Week 28, 2025
        baseline_week28 = baseline_df[
            (baseline_df["Year"] == 2025) & (baseline_df["WeekNumber"] == 28)
        ].copy()

        # Perform discrepancy analysis
        discrepancies_df, llm_common, llm_only_df, baseline_only_df = (
            perform_discrepancy_analysis(llm_data, baseline_week28)
        )

        return {
            "discrepancies_df": discrepancies_df,
            "llm_common": llm_common,
            "llm_only_df": llm_only_df,
            "baseline_only_df": baseline_only_df,
            "prompt_version": prompt_version,
            "extraction_file": str(extraction_path),
        }

    except Exception as e:
        print(f"❌ Error analyzing prompt {prompt_version}: {e}")
        return None


def list_available_prompt_versions(outputs_dir: str = "outputs") -> list:
    """
    List all available prompt versions based on extraction files.

    Returns:
        List of prompt versions found
    """
    outputs_path = Path(outputs_dir)
    if not outputs_path.exists():
        print(f"❌ Outputs directory not found: {outputs_path}")
        return []

    # Find all prompt-versioned files
    pattern = "text_extracted_data_prompt_*.csv"
    files = list(outputs_path.glob(pattern))

    versions = []
    for file in files:
        # Extract version from filename
        name = file.stem  # Remove .csv
        if name.startswith("text_extracted_data_prompt_"):
            version = name.replace("text_extracted_data_prompt_", "")
            versions.append(version)

    versions.sort()
    print(f"📁 Found {len(versions)} prompt versions: {versions}")
    return versions


def quick_discrepancy_check(prompt_version: str):
    """
    Quick check of discrepancies for a prompt version.
    Prints summary statistics.
    """
    discrepancies_df = get_discrepancies_by_prompt_version(prompt_version)

    if discrepancies_df is None:
        return

    if len(discrepancies_df) == 0:
        print(f"🎉 No discrepancies found for prompt {prompt_version}!")
        return

    print(f"\n📊 Discrepancy Summary for {prompt_version}:")
    print(f"   Total discrepant records: {len(discrepancies_df)}")

    # Count discrepancies by field
    fields = ["TotalCases", "CasesConfirmed", "Deaths", "CFR", "Grade"]
    for field in fields:
        discrepancy_col = f"{field}_discrepancy"
        if discrepancy_col in discrepancies_df.columns:
            count = discrepancies_df[discrepancy_col].sum()
            if count > 0:
                print(f"   {field}: {count} discrepancies")

    # Show most common countries with discrepancies
    if "Country" in discrepancies_df.columns:
        country_counts = discrepancies_df["Country"].value_counts().head(3)
        print(f"   Top countries with discrepancies: {dict(country_counts)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Get prompt comparison analysis results by version"
    )
    parser.add_argument("prompt_version", help="Prompt version (e.g., v1.1.1)")
    parser.add_argument("--outputs", default="outputs", help="Outputs directory")
    parser.add_argument("--data", default="data", help="Data directory")
    parser.add_argument("--summary", action="store_true", help="Show quick summary")

    args = parser.parse_args()

    if args.summary:
        quick_discrepancy_check(args.prompt_version)
    else:
        discrepancies_df = get_discrepancies_by_prompt_version(
            args.prompt_version, args.outputs, args.data
        )
        if discrepancies_df is not None:
            print(f"\nDiscrepancies DataFrame shape: {discrepancies_df.shape}")
            print(f"Columns: {list(discrepancies_df.columns)}")
