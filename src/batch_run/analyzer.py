"""
Analysis functions for comparing batch extraction results with baseline data.
"""

import pandas as pd
import numpy as np
from ..compare import perform_discrepancy_analysis
from ..post_processing import apply_post_processing_pipeline


def analyze_single_week(batch_df, baseline_df, week, year, correct_gap_fill_errors=False):
    """
    Perform discrepancy analysis for a single week/year combination.

    Args:
        batch_df: DataFrame with batch extraction data
        baseline_df: DataFrame with baseline data
        week: Week number to analyze
        year: Year to analyze
        correct_gap_fill_errors: If True, apply experimental gap-filling corrections

    Returns:
        Dictionary with analysis results including:
        - week, year: Identifiers
        - batch_count, baseline_count: Record counts
        - llm_only_count, baseline_only_count: Unique record counts
        - discrepancies: DataFrame with value discrepancies
        - discrepancies_long: Long-format discrepancy data
        - llm_only, baseline_only, llm_common: Detailed comparison DataFrames
    """
    # Filter to specific week and year
    baseline_week = baseline_df[
        (baseline_df["Year"] == year) & (baseline_df["WeekNumber"] == week)
    ].copy()

    batch_week = batch_df[
        (batch_df["Year"] == year) & (batch_df["WeekNumber"] == week)
    ].copy()

    if len(batch_week) == 0:
        print(f"  ⚠️  No batch data for Week {week}, Year {year}")
        return None

    if len(baseline_week) == 0:
        print(f"  ⚠️  No baseline data for Week {week}, Year {year}")
        return None

    # Apply post-processing to batch data if requested
    if correct_gap_fill_errors:
        print(f"  🔧 Applying gap-filling corrections to Week {week}, {year}")
        batch_week = apply_post_processing_pipeline(
            batch_week,
            source="llm",
            correct_gap_fill_errors=True
        )

    # Perform discrepancy analysis
    discrepancies, llm_common, llm_only, baseline_only = perform_discrepancy_analysis(
        batch_week, baseline_week
    )

    # Pivot discrepancies to long format
    discrepancies_long = pivot_discrepancies_long(discrepancies)

    # Add metadata
    if len(discrepancies_long) > 0:
        discrepancies_long['Week'] = week
        discrepancies_long['Year'] = year
        source_file = batch_week['SourceFile'].iloc[0] if 'SourceFile' in batch_week.columns else 'Unknown'
        discrepancies_long['SourceFile'] = source_file

    return {
        'week': week,
        'year': year,
        'source_file': batch_week['SourceFile'].iloc[0] if 'SourceFile' in batch_week.columns else 'Unknown',
        'batch_count': len(batch_week),
        'baseline_count': len(baseline_week),
        'llm_only_count': len(llm_only) if llm_only is not None else 0,
        'baseline_only_count': len(baseline_only) if baseline_only is not None else 0,
        'discrepancies': discrepancies,
        'discrepancies_long': discrepancies_long,
        'llm_only': llm_only,
        'baseline_only': baseline_only,
        'llm_common': llm_common
    }


def analyze_batch_vs_baseline(batch_df, baseline_df, correct_gap_fill_errors=False, verbose=True):
    """
    Analyze all batch runs against baseline data.

    Args:
        batch_df: DataFrame with all batch extraction data (must have WeekNumber and Year)
        baseline_df: DataFrame with baseline data
        correct_gap_fill_errors: If True, apply experimental gap-filling corrections
        verbose: If True, print progress information

    Returns:
        Tuple of (results_list, combined_discrepancies_df) where:
        - results_list: List of analysis results for each week
        - combined_discrepancies_df: All discrepancies combined into one DataFrame
    """
    # Get unique week/year combinations from batch runs
    week_year_combos = batch_df[['WeekNumber', 'Year']].drop_duplicates().sort_values(['Year', 'WeekNumber'])

    if verbose:
        print(f"\n{'='*80}")
        print(f"BATCH RUN ANALYSIS: {len(week_year_combos)} weeks")
        if correct_gap_fill_errors:
            print("⚠️  Gap-filling corrections: ENABLED (experimental)")
        else:
            print("ℹ️  Gap-filling corrections: DISABLED (default)")
        print(f"{'='*80}\n")

    # Analyze each week
    results = []
    all_discrepancies_long = []

    for _, row in week_year_combos.iterrows():
        week = row['WeekNumber']
        year = row['Year']

        if verbose:
            print(f"Week {week}, {year}:")

        result = analyze_single_week(
            batch_df,
            baseline_df,
            week,
            year,
            correct_gap_fill_errors=correct_gap_fill_errors
        )

        if result:
            results.append(result)

            # Print summary
            if verbose:
                print(f"  Batch: {result['batch_count']} | Baseline: {result['baseline_count']}")
                print(f"  LLM-only: {result['llm_only_count']} | Baseline-only: {result['baseline_only_count']}")

            if result['discrepancies'] is not None and len(result['discrepancies']) > 0:
                discrepancies_only = result['discrepancies_long'][
                    result['discrepancies_long']['Discrepancy'] == True
                ]
                if verbose:
                    print(f"  Value discrepancies: {len(discrepancies_only)}")

                if len(discrepancies_only) > 0:
                    all_discrepancies_long.append(discrepancies_only)
            else:
                if verbose:
                    print(f"  Value discrepancies: 0")

        if verbose:
            print()

    # Combine all discrepancies
    combined_discrepancies = None
    if all_discrepancies_long:
        combined_discrepancies = pd.concat(all_discrepancies_long, ignore_index=True)

        if verbose:
            print(f"\n{'='*80}")
            print(f"SUMMARY: {len(combined_discrepancies)} total discrepancies across all weeks")
            print(f"{'='*80}")

    return results, combined_discrepancies


def categorize_discrepancies(discrepancies_df):
    """
    Categorize discrepancies by type (zero vs non-zero, comma issues, magnitude).

    Args:
        discrepancies_df: DataFrame with discrepancies (must have LLM and Baseline columns)

    Returns:
        DataFrame with added 'Category' column and categorization details
    """
    if discrepancies_df is None or len(discrepancies_df) == 0:
        return pd.DataFrame()

    df = discrepancies_df.copy()

    # Convert to numeric if needed
    for col in ['LLM', 'Baseline']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    def categorize_row(row):
        llm_val = row.get('LLM', 0)
        baseline_val = row.get('Baseline', 0)

        # Handle NaN
        llm_val = 0 if pd.isna(llm_val) else llm_val
        baseline_val = 0 if pd.isna(baseline_val) else baseline_val

        # Zero vs Non-Zero
        if (llm_val == 0 and baseline_val != 0) or (llm_val != 0 and baseline_val == 0):
            return "Zero vs Non-Zero"

        # Both zero
        if llm_val == 0 and baseline_val == 0:
            return "Both Zero"

        # Calculate difference
        diff = abs(llm_val - baseline_val)

        # Comma/Thousands Issue (baseline appears to be truncated)
        # e.g., 26834 vs 26, or 1234 vs 1
        if baseline_val > 0 and llm_val > baseline_val:
            ratio = llm_val / baseline_val
            # Check if LLM value is roughly 1000x or 10x or 100x the baseline
            if 900 < ratio < 1100 or 9 < ratio < 11 or 90 < ratio < 110:
                return "Comma/Thousands Issue"

        # Magnitude-based categories
        if diff > 100:
            return "Large Difference (>100)"
        elif diff > 20:
            return "Medium Difference (21-100)"
        else:
            return "Small Difference (≤20)"

    df['Category'] = df.apply(categorize_row, axis=1)

    # Add sub-category for zero vs non-zero (which one is zero?)
    def sub_categorize_zero(row):
        if row['Category'] == 'Zero vs Non-Zero':
            llm_val = row.get('LLM', 0)
            baseline_val = row.get('Baseline', 0)
            if pd.isna(llm_val) or llm_val == 0:
                return "LLM has 0"
            else:
                return "Baseline has 0"
        return ""

    df['SubCategory'] = df.apply(sub_categorize_zero, axis=1)

    return df


def create_summary_statistics(results_list):
    """
    Create summary statistics table from analysis results.

    Args:
        results_list: List of analysis result dictionaries

    Returns:
        DataFrame with summary statistics by week
    """
    summary_data = []

    for result in results_list:
        disc_count = 0
        if result.get('discrepancies_long') is not None and len(result['discrepancies_long']) > 0:
            disc_count = len(result['discrepancies_long'][
                result['discrepancies_long']['Discrepancy'] == True
            ])

        summary_data.append({
            'Week': result['week'],
            'Year': result['year'],
            'SourceFile': result.get('source_file', 'Unknown'),
            'BatchRecords': result['batch_count'],
            'BaselineRecords': result['baseline_count'],
            'RecordDifference': result['batch_count'] - result['baseline_count'],
            'LLMOnly': result['llm_only_count'],
            'BaselineOnly': result['baseline_only_count'],
            'ValueDiscrepancies': disc_count
        })

    return pd.DataFrame(summary_data)


def pivot_discrepancies_long(discrepancies_df):
    """
    Pivot discrepancies dataframe to long format.

    Args:
        discrepancies_df: Wide-format discrepancy DataFrame

    Returns:
        Long-format DataFrame with columns:
        Country, Event, Parameter, LLM, Baseline, Discrepancy
    """
    if discrepancies_df is None or len(discrepancies_df) == 0:
        return pd.DataFrame(
            columns=["Country", "Event", "Parameter", "LLM", "Baseline", "Discrepancy"]
        )

    # Identify the parameters by finding discrepancy columns
    discrepancy_cols = [
        col for col in discrepancies_df.columns if col.endswith("_discrepancy")
    ]
    parameters = [col.replace("_discrepancy", "") for col in discrepancy_cols]

    # Create the long format dataframe
    long_data = []

    for _, row in discrepancies_df.iterrows():
        country = row["Country"]
        event = row["Event"]

        for param in parameters:
            # Get the values for this parameter
            discrepancy_col = f"{param}_discrepancy"
            llm_col = f"llm_{param}"
            baseline_col = f"baseline_{param}"

            if all(
                col in discrepancies_df.columns
                for col in [discrepancy_col, llm_col, baseline_col]
            ):
                long_data.append(
                    {
                        "Country": country,
                        "Event": event,
                        "Parameter": param,
                        "LLM": row[llm_col],
                        "Baseline": row[baseline_col],
                        "Discrepancy": row[discrepancy_col],
                    }
                )

    return pd.DataFrame(long_data)
