"""
Data loading utilities for batch run analysis.
"""

import re
from pathlib import Path
import pandas as pd


def parse_week_year_from_filename(filename):
    """
    Extract week number and year from filenames like:
    - Week_21__19_-_25_May_2025.csv -> (21, 2025)
    - Week_33__14_-_20_August_2023.csv -> (33, 2023)

    Args:
        filename: String filename to parse

    Returns:
        Tuple of (week_number, year) or (None, None) if parsing fails
    """
    # Try to extract year from filename
    year_match = re.search(r'(\d{4})', filename)
    year = int(year_match.group(1)) if year_match else None

    # Extract week number
    week_match = re.search(r'Week_(\d+)', filename)
    week = int(week_match.group(1)) if week_match else None

    return week, year


def load_batch_data(batch_dir="outputs/batch_run", add_metadata=True):
    """
    Load all batch run CSV files and optionally add metadata columns.

    Args:
        batch_dir: Directory containing batch run CSV files
        add_metadata: If True, add WeekNumber, Year, and SourceFile columns

    Returns:
        Tuple of (combined_df, list_of_dfs) where:
        - combined_df: All batch data concatenated
        - list_of_dfs: List of individual DataFrames (one per file)
    """
    batch_dir = Path(batch_dir)
    batch_files = sorted(batch_dir.glob("*.csv"))

    if not batch_files:
        raise FileNotFoundError(f"No CSV files found in {batch_dir}")

    batch_data = []
    for filepath in batch_files:
        df = pd.read_csv(filepath)

        if add_metadata:
            week, year = parse_week_year_from_filename(filepath.name)
            df['WeekNumber'] = week
            df['Year'] = year
            df['SourceFile'] = filepath.name

        batch_data.append(df)

    # Combine all batch runs
    all_batches = pd.concat(batch_data, ignore_index=True)

    print(f"✓ Loaded {len(all_batches)} records from {len(batch_files)} batch files")
    if add_metadata:
        weeks = all_batches['WeekNumber'].dropna().unique()
        years = all_batches['Year'].dropna().unique()
        print(f"  Weeks: {sorted([int(w) for w in weeks])}")
        print(f"  Years: {sorted([int(y) for y in years])}")

    return all_batches, batch_data


def load_baseline_data(baseline_path="data/final_data_for_powerbi_with_kpi.csv", standardize=True):
    """
    Load baseline rule-based scraper data.

    Args:
        baseline_path: Path to baseline CSV file
        standardize: If True, standardize column names to match batch data

    Returns:
        DataFrame with baseline data
    """
    baseline_df = pd.read_csv(baseline_path, low_memory=False)

    if standardize:
        baseline_df = standardize_column_names(baseline_df, is_baseline=True)

    print(f"✓ Loaded {len(baseline_df)} baseline records")
    if 'Year' in baseline_df.columns and 'WeekNumber' in baseline_df.columns:
        years = baseline_df['Year'].dropna().unique()
        print(f"  Coverage: {baseline_df['Year'].min():.0f}-{baseline_df['Year'].max():.0f}")

    return baseline_df


def standardize_column_names(df, is_baseline=False):
    """
    Standardize column names between batch runs and baseline.

    Args:
        df: DataFrame to standardize
        is_baseline: If True, applies baseline-specific name mappings

    Returns:
        DataFrame with standardized column names
    """
    if is_baseline:
        # Baseline has spaces in column names
        rename_map = {
            'Date notified to WCO': 'DateNotified',
            'Start of reporting period': 'StartReportingPeriod',
            'End of reporting period': 'EndReportingPeriod',
            'Total cases': 'TotalCases',
            'Cases Confirmed': 'CasesConfirmed'
        }
    else:
        # Batch runs might have different naming conventions
        rename_map = {
            'Total cases': 'TotalCases',
            'Cases Confirmed': 'CasesConfirmed',
            'Date notified to WCO': 'DateNotified',
            'Start of reporting period': 'StartReportingPeriod',
            'End of reporting period': 'EndReportingPeriod',
        }

    return df.rename(columns=rename_map)
