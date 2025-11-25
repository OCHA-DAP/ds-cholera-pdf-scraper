"""
Analysis functions for comparing LLM extraction results with rule-based data.
"""

import pandas as pd
import numpy as np
from ..compare import perform_discrepancy_analysis
from ..post_processing import apply_post_processing_pipeline


def analyze_single_week(llm_df, rule_based_df, week, year, correct_gap_fill_errors=False):
    """
    Perform discrepancy analysis for a single week/year combination.

    Args:
        llm_df: DataFrame with LLM extraction data
        rule_based_df: DataFrame with rule-based data
        week: Week number to analyze
        year: Year to analyze
        correct_gap_fill_errors: If True, apply experimental gap-filling corrections

    Returns:
        Dictionary with analysis results including:
        - week, year: Identifiers
        - llm_count, rule_based_count: Record counts
        - llm_only_count, rule_based_only_count: Unique record counts
        - discrepancies: DataFrame with value discrepancies
        - discrepancies_long: Long-format discrepancy data
        - llm_only, rule_based_only, llm_common: Detailed comparison DataFrames
    """
    # Filter to specific week and year
    rule_based_week = rule_based_df[
        (rule_based_df["Year"] == year) & (rule_based_df["WeekNumber"] == week)
    ].copy()

    llm_week = llm_df[
        (llm_df["Year"] == year) & (llm_df["WeekNumber"] == week)
    ].copy()

    if len(llm_week) == 0:
        print(f"  ⚠️  No LLM data for Week {week}, Year {year}")
        return None

    if len(rule_based_week) == 0:
        print(f"  ⚠️  No rule-based data for Week {week}, Year {year}")
        return None

    # Apply post-processing to LLM data if requested
    if correct_gap_fill_errors:
        print(f"  🔧 Applying gap-filling corrections to Week {week}, {year}")
        llm_week = apply_post_processing_pipeline(
            llm_week,
            source="llm",
            correct_gap_fill_errors=True
        )

    # Perform discrepancy analysis
    discrepancies, llm_common, llm_only, rule_based_only = perform_discrepancy_analysis(
        llm_week, rule_based_week
    )

    # Pivot discrepancies to long format
    discrepancies_long = pivot_discrepancies_long(discrepancies)

    # Add metadata
    if len(discrepancies_long) > 0:
        discrepancies_long['Week'] = week
        discrepancies_long['Year'] = year
        source_file = llm_week['SourceFile'].iloc[0] if 'SourceFile' in llm_week.columns else 'Unknown'
        discrepancies_long['SourceFile'] = source_file

    return {
        'week': week,
        'year': year,
        'source_file': llm_week['SourceFile'].iloc[0] if 'SourceFile' in llm_week.columns else 'Unknown',
        'llm_count': len(llm_week),
        'rule_based_count': len(rule_based_week),
        'llm_only_count': len(llm_only) if llm_only is not None else 0,
        'rule_based_only_count': len(rule_based_only) if rule_based_only is not None else 0,
        'discrepancies': discrepancies,
        'discrepancies_long': discrepancies_long,
        'llm_only': llm_only,
        'rule_based_only': rule_based_only,
        'llm_common': llm_common
    }


def analyze_llm_vs_rule_based(llm_df, rule_based_df, correct_gap_fill_errors=False, verbose=True):
    """
    Analyze all LLM extractions against rule-based data.

    Args:
        llm_df: DataFrame with all LLM extraction data (must have WeekNumber and Year)
        rule_based_df: DataFrame with rule-based data
        correct_gap_fill_errors: If True, apply experimental gap-filling corrections
        verbose: If True, print progress information

    Returns:
        Tuple of (results_list, combined_discrepancies_df) where:
        - results_list: List of analysis results for each week
        - combined_discrepancies_df: All discrepancies combined into one DataFrame
    """
    # Get unique week/year combinations from LLM extractions
    week_year_combos = llm_df[['WeekNumber', 'Year']].drop_duplicates().sort_values(['Year', 'WeekNumber'])

    if verbose:
        print(f"\n{'='*80}")
        print(f"LLM vs RULE-BASED ANALYSIS: {len(week_year_combos)} weeks")
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
            llm_df,
            rule_based_df,
            week,
            year,
            correct_gap_fill_errors=correct_gap_fill_errors
        )

        if result:
            results.append(result)

            # Print summary
            if verbose:
                print(f"  LLM: {result['llm_count']} | Rule-Based: {result['rule_based_count']}")
                print(f"  LLM-only: {result['llm_only_count']} | Rule-based-only: {result['rule_based_only_count']}")

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
        discrepancies_df: DataFrame with discrepancies (must have LLM and RuleBased columns)

    Returns:
        DataFrame with added 'Category' column and categorization details
    """
    if discrepancies_df is None or len(discrepancies_df) == 0:
        return pd.DataFrame()

    df = discrepancies_df.copy()

    # Convert to numeric if needed and create numeric columns
    for col in ['LLM', 'RuleBased']:
        if col in df.columns:
            df[f'{col}_numeric'] = pd.to_numeric(df[col], errors='coerce')
            # Also update the original column
            df[col] = df[f'{col}_numeric']

    # Calculate ratio for analysis (LLM / RuleBased)
    df['Ratio'] = np.nan
    mask = (df['RuleBased_numeric'] != 0) & (df['RuleBased_numeric'].notna())
    df.loc[mask, 'Ratio'] = df.loc[mask, 'LLM_numeric'] / df.loc[mask, 'RuleBased_numeric']

    def categorize_row(row):
        llm_val = row.get('LLM', 0)
        rule_based_val = row.get('RuleBased', 0)

        # Handle NaN
        llm_val = 0 if pd.isna(llm_val) else llm_val
        rule_based_val = 0 if pd.isna(rule_based_val) else rule_based_val

        # Zero vs Non-Zero
        if (llm_val == 0 and rule_based_val != 0) or (llm_val != 0 and rule_based_val == 0):
            return "Zero vs Non-Zero"

        # Both zero
        if llm_val == 0 and rule_based_val == 0:
            return "Both Zero"

        # Calculate difference
        diff = abs(llm_val - rule_based_val)

        # Comma/Thousands Issue (rule-based appears to be truncated)
        # e.g., 26834 vs 26, or 1234 vs 1
        if rule_based_val > 0 and llm_val > rule_based_val:
            ratio = llm_val / rule_based_val
            # Check if LLM value is roughly 1000x or 10x or 100x the rule-based
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
            rule_based_val = row.get('RuleBased', 0)
            if pd.isna(llm_val) or llm_val == 0:
                return "LLM has 0"
            elif pd.isna(rule_based_val) or rule_based_val == 0:
                return "RuleBased has 0"
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
            'LLMRecords': result['llm_count'],
            'RuleBasedRecords': result['rule_based_count'],
            'RecordDifference': result['llm_count'] - result['rule_based_count'],
            'LLMOnly': result['llm_only_count'],
            'RuleBasedOnly': result['rule_based_only_count'],
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
        Country, Event, Parameter, LLM, RuleBased, Discrepancy
    """
    if discrepancies_df is None or len(discrepancies_df) == 0:
        return pd.DataFrame(
            columns=["Country", "Event", "Parameter", "LLM", "RuleBased", "Discrepancy"]
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
            rule_based_col = f"rule_based_{param}"

            if all(
                col in discrepancies_df.columns
                for col in [discrepancy_col, llm_col, rule_based_col]
            ):
                long_data.append(
                    {
                        "Country": country,
                        "Event": event,
                        "Parameter": param,
                        "LLM": row[llm_col],
                        "RuleBased": row[rule_based_col],
                        "Discrepancy": row[discrepancy_col],
                    }
                )

    return pd.DataFrame(long_data)
