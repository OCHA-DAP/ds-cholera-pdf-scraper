"""
Data loading utilities for LLM vs rule-based comparison analysis.
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


def load_llm_data(llm_dir="outputs/batch_run", add_metadata=True):
    """
    Load all LLM extraction CSV files and optionally add metadata columns.

    Args:
        llm_dir: Directory containing LLM extraction CSV files
        add_metadata: If True, add WeekNumber, Year, and SourceFile columns

    Returns:
        Tuple of (combined_df, list_of_dfs) where:
        - combined_df: All LLM data concatenated
        - list_of_dfs: List of individual DataFrames (one per file)
    """
    llm_dir = Path(llm_dir)
    llm_files = sorted(llm_dir.glob("*.csv"))

    if not llm_files:
        raise FileNotFoundError(f"No CSV files found in {llm_dir}")

    llm_data = []
    for filepath in llm_files:
        df = pd.read_csv(filepath)

        if add_metadata:
            week, year = parse_week_year_from_filename(filepath.name)
            df['WeekNumber'] = week
            df['Year'] = year
            df['SourceFile'] = filepath.name

        llm_data.append(df)

    # Combine all LLM extraction runs
    all_llm_data = pd.concat(llm_data, ignore_index=True)

    print(f"✓ Loaded {len(all_llm_data)} records from {len(llm_files)} LLM extraction files")
    if add_metadata:
        weeks = all_llm_data['WeekNumber'].dropna().unique()
        years = all_llm_data['Year'].dropna().unique()
        print(f"  Weeks: {sorted([int(w) for w in weeks])}")
        print(f"  Years: {sorted([int(y) for y in years])}")

    return all_llm_data, llm_data


def load_rule_based_data(rule_based_path="data/final_data_for_powerbi_with_kpi.csv", standardize=True):
    """
    Load rule-based scraper data.

    Args:
        rule_based_path: Path to rule-based CSV file
        standardize: If True, standardize column names to match LLM data

    Returns:
        DataFrame with rule-based data
    """
    rule_based_df = pd.read_csv(rule_based_path, low_memory=False)

    if standardize:
        rule_based_df = standardize_column_names(rule_based_df, is_rule_based=True)

    print(f"✓ Loaded {len(rule_based_df)} rule-based records")
    if 'Year' in rule_based_df.columns and 'WeekNumber' in rule_based_df.columns:
        print(f"  Coverage: {rule_based_df['Year'].min():.0f}-{rule_based_df['Year'].max():.0f}")

    return rule_based_df


def standardize_column_names(df, is_rule_based=False):
    """
    Standardize column names between LLM extractions and rule-based data.

    Args:
        df: DataFrame to standardize
        is_rule_based: If True, applies rule-based-specific name mappings

    Returns:
        DataFrame with standardized column names
    """
    if is_rule_based:
        # Rule-based data has spaces in column names
        rename_map = {
            'Date notified to WCO': 'DateNotified',
            'Start of reporting period': 'StartReportingPeriod',
            'End of reporting period': 'EndReportingPeriod',
            'Total cases': 'TotalCases',
            'Cases Confirmed': 'CasesConfirmed'
        }
    else:
        # LLM extractions might have different naming conventions
        rename_map = {
            'Total cases': 'TotalCases',
            'Cases Confirmed': 'CasesConfirmed',
            'Date notified to WCO': 'DateNotified',
            'Start of reporting period': 'StartReportingPeriod',
            'End of reporting period': 'EndReportingPeriod',
        }

    return df.rename(columns=rename_map)
