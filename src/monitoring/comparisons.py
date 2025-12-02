"""
Modular comparison system for cholera data extraction pipelines.

This module provides functions to compare LLM and rule-based extractions,
generate comparison reports, and upload results to blob storage.

Key Features:
- Flexible input options (DataFrames, blob paths, specific files)
- Multiple output modes (blob, local, return)
- Reuses existing comparison logic from src.batch_run
- CFR consistency checking
- CLI interface for standalone use

Usage:
    # As a module (from scripts or notebooks)
    from src.monitoring.comparisons import generate_comparison_reports

    results = generate_comparison_reports(
        week=42,
        year=2025,
        stage='dev',
        output_mode='blob'
    )

    # From command line
    python -m src.monitoring.comparisons --week 42 --year 2025 --stage dev
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import numpy as np
import ocha_stratus as stratus

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from src.config import Config
from src.batch_run.analyzer import (
    analyze_llm_vs_rule_based,
    categorize_discrepancies,
    create_summary_statistics,
)
from src.batch_run.loader import standardize_column_names
from src.compare import perform_discrepancy_analysis

# Optional visualization imports
try:
    from src.batch_run.visualization import check_cfr_consistency
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    check_cfr_consistency = None


# =============================================================================
# BLOB DATA LOADER (adapted from realtime-comparisons branch)
# =============================================================================


class BlobExtractionLoader:
    """
    Load extraction results from Azure Blob Storage.

    Supports loading both rule-based and LLM-based extractions from the
    processed monitoring folders.
    """

    def __init__(self, stage: str = None):
        """
        Initialize blob loader.

        Args:
            stage: Azure stage (dev/prod), defaults to Config.STAGE or 'dev'
        """
        self.stage = stage or os.getenv('STAGE', 'dev')
        self.container = Config.BLOB_CONTAINER
        self.proj_dir = Config.BLOB_PROJ_DIR
        self.account_name = f"imb0chd0{self.stage}"

        # Get SAS token
        sas_token_key = f"DSCI_AZ_BLOB_{self.stage.upper()}_SAS_WRITE"
        self.sas_token = os.getenv(sas_token_key)

        if not self.sas_token:
            # Try read-only token
            sas_token_key = f"DSCI_AZ_BLOB_{self.stage.upper()}_SAS"
            self.sas_token = os.getenv(sas_token_key)

        if not self.sas_token:
            raise ValueError(
                f"No Azure SAS token found. Set {sas_token_key} environment variable."
            )

    def list_available_extractions(self, source: str) -> pd.DataFrame:
        """
        List all available extractions in blob storage.

        Args:
            source: "llm" or "rule-based"

        Returns:
            DataFrame with columns: filename, week, year, blob_name (and model for LLM)
        """
        # Get blob prefix from config
        blob_paths = Config.get_blob_paths()
        if source == "llm":
            blob_prefix = blob_paths["processed_llm_extractions"]
        elif source == "rule-based":
            blob_prefix = blob_paths["processed_rule_based_extractions"]
        else:
            raise ValueError(f"Invalid source: {source}")

        # List blobs using stratus
        blob_names = stratus.list_container_blobs(
            name_starts_with=blob_prefix,
            stage=self.stage,
            container_name=self.container
        )

        # Parse blob names to extract metadata
        extraction_info = []
        for blob_name in blob_names:
            if blob_name.endswith('.csv') and '/archive/' not in blob_name:
                filename = Path(blob_name).name

                # Parse filename like: OEW42-2025_gpt-4o_1234567890.csv
                # or: OEW42-2025_rule-based_1234567890_processed.csv
                parts = filename.replace('.csv', '').replace('_processed', '').split('_')

                if parts[0].startswith('OEW'):
                    week_year = parts[0].replace('OEW', '')
                    if '-' in week_year:
                        week, year = week_year.split('-')

                        info = {
                            'blob_name': blob_name,
                            'filename': filename,
                            'week': int(week),
                            'year': int(year),
                        }

                        # Add model for LLM extractions
                        if source == "llm" and len(parts) > 1:
                            info['model'] = parts[1]

                        # Extract run_id/timestamp (last numeric part)
                        if len(parts) >= 3:
                            try:
                                info['run_id'] = int(parts[-1])
                            except ValueError:
                                info['run_id'] = 0

                        extraction_info.append(info)

        if not extraction_info:
            return pd.DataFrame()

        df = pd.DataFrame(extraction_info)
        sort_cols = ['year', 'week', 'run_id'] if 'run_id' in df.columns else ['year', 'week']
        return df.sort_values(sort_cols, ascending=False)

    def load_extraction(
        self,
        source: str,
        week: Optional[int] = None,
        year: Optional[int] = None,
        model: Optional[str] = None,
        blob_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load a specific extraction from blob storage.

        Args:
            source: "llm" or "rule-based"
            week: Week number (optional if blob_path provided)
            year: Year (optional if blob_path provided)
            model: Model name for LLM extractions (optional, uses first match if None)
            blob_path: Specific blob path (optional)

        Returns:
            DataFrame with extraction data
        """
        if blob_path:
            # Load from specific path
            blob_name = blob_path
        elif week and year:
            # Find extraction for this week/year
            available = self.list_available_extractions(source)

            if source == "llm":
                matches = available[
                    (available['week'] == week) &
                    (available['year'] == year)
                ]
                # Filter by model if specified
                if model and 'model' in matches.columns:
                    matches = matches[matches['model'] == model]
            else:
                matches = available[
                    (available['week'] == week) &
                    (available['year'] == year)
                ]

            if len(matches) == 0:
                model_msg = f" (model: {model})" if source == "llm" and model else ""
                raise FileNotFoundError(
                    f"No {source} extraction found for Week {week}, Year {year}{model_msg}"
                )

            # Take most recent (first row, already sorted)
            blob_name = matches.iloc[0]['blob_name']
        else:
            raise ValueError("Must provide either (week + year) or blob_path")

        # Load from blob using stratus
        print(f"  📥 Loading {source} extraction: {Path(blob_name).name}")
        df = stratus.load_csv_from_blob(
            blob_name=blob_name,
            stage=self.stage,
            container_name=self.container
        )

        # Standardize column names
        df = standardize_column_names(df, is_rule_based=(source == "rule-based"))

        # Add metadata if not present
        if week and 'WeekNumber' not in df.columns:
            df['WeekNumber'] = week
        if year and 'Year' not in df.columns:
            df['Year'] = year

        print(f"  ✅ Loaded {len(df)} records")
        return df

    def load_bulk(
        self,
        source: str,
        weeks: Optional[List[int]] = None,
        year: Optional[int] = None,
        model: str = 'gpt-4o'
    ) -> pd.DataFrame:
        """
        Load multiple extractions in bulk.

        Args:
            source: "llm" or "rule-based"
            weeks: List of week numbers (optional)
            year: Year to filter (required if weeks provided)
            model: Model name for LLM extractions (default: 'gpt-4o')

        Returns:
            Combined DataFrame
        """
        available = self.list_available_extractions(source)

        if len(available) == 0:
            raise FileNotFoundError(f"No {source} extractions found")

        # Filter by criteria
        if weeks and year:
            available = available[
                (available['week'].isin(weeks)) & (available['year'] == year)
            ]

        if source == "llm":
            available = available[available['model'] == model]

        if len(available) == 0:
            raise FileNotFoundError(f"No {source} extractions found matching criteria")

        # Get unique week/year combinations and take most recent
        group_cols = ['week', 'year', 'model'] if source == "llm" else ['week', 'year']
        unique_weeks = available.groupby(group_cols).first().reset_index()

        # Load each extraction
        dfs = []
        for _, row in unique_weeks.iterrows():
            try:
                df = self.load_extraction(
                    source=source,
                    week=row['week'],
                    year=row['year'],
                    model=row.get('model', model)
                )
                df['SourceFile'] = row['filename']
                dfs.append(df)
            except Exception as e:
                print(f"  ⚠️  Failed to load {row['filename']}: {e}")

        if not dfs:
            raise ValueError("Failed to load any extractions")

        combined = pd.concat(dfs, ignore_index=True)
        print(f"  ✅ Combined {len(dfs)} files ({len(combined)} total records)")
        return combined


# =============================================================================
# CFR CONSISTENCY CHECKING
# =============================================================================


def compare_cfr_consistency(
    disc_cat: pd.DataFrame,
    llm_df: pd.DataFrame,
    rule_based_df: pd.DataFrame,
    parameter: str = 'TotalCases',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare CFR consistency between LLM and rule-based for discrepancies.

    Uses the check_cfr_consistency function from visualization module to
    validate which extraction has values more consistent with reported CFR.

    Args:
        disc_cat: Categorized discrepancies DataFrame
        llm_df: LLM extraction data
        rule_based_df: Rule-based extraction data
        parameter: Parameter to analyze (default: 'TotalCases')
        verbose: Print results

    Returns:
        DataFrame with CFR comparison results
    """
    if not VISUALIZATION_AVAILABLE or check_cfr_consistency is None:
        if verbose:
            print("  ⚠️  CFR consistency check requires visualization module")
        return pd.DataFrame()

    # Filter for specific parameter
    param_disc = disc_cat[disc_cat['Parameter'] == parameter].copy()

    if len(param_disc) == 0:
        if verbose:
            print(f"  ℹ️  No {parameter} discrepancies found for CFR analysis")
        return pd.DataFrame()

    cfr_results = []
    for _, disc_row in param_disc.iterrows():
        country = disc_row['Country']
        event = disc_row['Event']
        year = disc_row['Year']
        week = disc_row['Week']

        # Get matching records
        llm_rec = llm_df[
            (llm_df['Country'] == country) &
            (llm_df['Event'] == event) &
            (llm_df['Year'] == year) &
            (llm_df['WeekNumber'] == week)
        ]

        rb_rec = rule_based_df[
            (rule_based_df['Country'] == country) &
            (rule_based_df['Event'] == event) &
            (rule_based_df['Year'] == year) &
            (rule_based_df['WeekNumber'] == week)
        ]

        if len(llm_rec) > 0 and len(rb_rec) > 0:
            # Check CFR consistency
            llm_rec = check_cfr_consistency(llm_rec)
            rb_rec = check_cfr_consistency(rb_rec)

            llm_err = llm_rec['cfr_error'].iloc[0] if not pd.isna(llm_rec['cfr_error'].iloc[0]) else 999
            rb_err = rb_rec['cfr_error'].iloc[0] if not pd.isna(rb_rec['cfr_error'].iloc[0]) else 999

            if llm_err < 999 or rb_err < 999:
                cfr_results.append({
                    'Country': country,
                    'Event': event,
                    'Year': year,
                    'Week': week,
                    'Category': disc_row['Category'],
                    'LLM_CFR_Error': llm_err,
                    'RuleBased_CFR_Error': rb_err,
                    'LLM_Better': llm_err < rb_err
                })

    cfr_df = pd.DataFrame(cfr_results)

    if verbose and len(cfr_df) > 0:
        total = len(cfr_df)
        llm_wins = cfr_df['LLM_Better'].sum()
        rb_wins = (~cfr_df['LLM_Better']).sum()

        print(f"\n  📊 CFR Consistency Analysis ({parameter}):")
        print(f"     Total discrepancies with CFR data: {total}")
        print(f"     LLM has better CFR consistency: {llm_wins} ({llm_wins/total*100:.1f}%)")
        print(f"     Rule-based has better CFR consistency: {rb_wins} ({rb_wins/total*100:.1f}%)")

    return cfr_df


# =============================================================================
# MAIN COMPARISON FUNCTION
# =============================================================================


def generate_comparison_reports(
    # Input options (flexible)
    llm_df: Optional[pd.DataFrame] = None,
    rule_based_df: Optional[pd.DataFrame] = None,
    week: Optional[int] = None,
    year: Optional[int] = None,
    weeks: Optional[List[int]] = None,
    llm_blob_path: Optional[str] = None,
    rule_based_blob_path: Optional[str] = None,
    # Processing options
    stage: str = 'dev',
    model: Optional[str] = None,
    correct_gap_fill_errors: bool = False,
    # Output options
    output_mode: str = 'blob',  # 'blob', 'local', or 'return'
    output_dir: Optional[str] = None,
    # Other options
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Generate comparison reports between LLM and rule-based extractions.

    This is the main entry point for generating comparison reports. It supports
    multiple input methods and output modes.

    Input Options (prioritized in this order):
        1. Direct DataFrames: llm_df and rule_based_df
        2. Specific blob paths: llm_blob_path and rule_based_blob_path
        3. Week/year loading: week, year (or weeks, year)

    Output Modes:
        - 'blob': Upload results to Azure blob storage (default)
        - 'local': Save to local directory (requires output_dir)
        - 'return': Just return the results dict (no file output)

    Args:
        llm_df: LLM extraction DataFrame (optional)
        rule_based_df: Rule-based extraction DataFrame (optional)
        week: Single week number to compare (optional)
        year: Year to compare (optional)
        weeks: List of week numbers to compare (optional)
        llm_blob_path: Specific LLM blob path (optional)
        rule_based_blob_path: Specific rule-based blob path (optional)
        stage: Azure stage (dev/prod)
        model: LLM model name (optional, uses first match if None)
        correct_gap_fill_errors: Apply experimental gap-fill corrections
        output_mode: Output mode ('blob', 'local', or 'return')
        output_dir: Local output directory (required if output_mode='local')
        verbose: Print progress information

    Returns:
        Dictionary with comparison results:
            - llm_df: LLM extraction data
            - rule_based_df: Rule-based extraction data
            - analysis_results: Detailed analysis by week
            - combined_discrepancies: All discrepancies combined
            - discrepancy_categories: Categorized discrepancies
            - summary_by_week: Summary statistics
            - cfr_comparison: CFR consistency comparison
            - unique_records: Records only in one source
            - output_files: Paths to generated files (if applicable)

    Example:
        >>> # Compare specific week from blob storage
        >>> results = generate_comparison_reports(
        ...     week=42, year=2025, stage='dev', output_mode='blob'
        ... )

        >>> # Compare using direct DataFrames
        >>> results = generate_comparison_reports(
        ...     llm_df=my_llm_data,
        ...     rule_based_df=my_rb_data,
        ...     output_mode='local',
        ...     output_dir='./comparison_results'
        ... )
    """
    if verbose:
        print("=" * 80)
        print("COMPARISON REPORT GENERATION")
        print("=" * 80)
        print(f"Stage: {stage}")
        print(f"Model: {model}")
        print(f"Output mode: {output_mode}")
        if week:
            print(f"Week: {week}, Year: {year}")
        elif weeks:
            print(f"Weeks: {weeks}, Year: {year}")
        print("=" * 80)
        print()

    # Step 1: Load data based on input method
    loader = None

    if llm_df is not None and rule_based_df is not None:
        # Direct DataFrames provided
        if verbose:
            print("📊 Using provided DataFrames")
            print(f"  LLM records: {len(llm_df)}")
            print(f"  Rule-based records: {len(rule_based_df)}")
            print()

    else:
        # Need to load from blob storage
        if verbose:
            print("📦 Loading data from blob storage...")
            print()

        loader = BlobExtractionLoader(stage=stage)

        # Load LLM data
        if llm_blob_path:
            llm_df = loader.load_extraction('llm', blob_path=llm_blob_path)
        elif week and year:
            llm_df = loader.load_extraction('llm', week=week, year=year, model=model)
        elif weeks and year:
            llm_df = loader.load_bulk('llm', weeks=weeks, year=year, model=model)
        else:
            raise ValueError(
                "Must provide either: (llm_df + rule_based_df), "
                "(llm_blob_path + rule_based_blob_path), or (week/weeks + year)"
            )

        # Load rule-based data
        if rule_based_blob_path:
            rule_based_df = loader.load_extraction('rule-based', blob_path=rule_based_blob_path)
        elif week and year:
            rule_based_df = loader.load_extraction('rule-based', week=week, year=year)
        elif weeks and year:
            rule_based_df = loader.load_bulk('rule-based', weeks=weeks, year=year)

        print()

    # Step 2: Perform analysis
    if verbose:
        print("🔍 Analyzing discrepancies...")
        print()

    analysis_results, combined_discrepancies = analyze_llm_vs_rule_based(
        llm_df=llm_df,
        rule_based_df=rule_based_df,
        correct_gap_fill_errors=correct_gap_fill_errors,
        verbose=verbose
    )

    # Step 3: Create summary statistics
    summary_by_week = create_summary_statistics(analysis_results)

    # Step 4: Categorize discrepancies
    discrepancy_categories = None
    if combined_discrepancies is not None and len(combined_discrepancies) > 0:
        discrepancy_categories = categorize_discrepancies(combined_discrepancies)

        if verbose:
            print(f"\n📊 Discrepancy breakdown:")
            print(discrepancy_categories.groupby('Category').size())
            print()

    # Step 5: CFR consistency comparison
    cfr_comparison = None
    if discrepancy_categories is not None and len(discrepancy_categories) > 0:
        cfr_comparison = compare_cfr_consistency(
            discrepancy_categories,
            llm_df,
            rule_based_df,
            verbose=verbose
        )

    # Step 6: Collect unique records (LLM-only and Rule-based-only)
    unique_records = []
    for analysis in analysis_results:
        if analysis['llm_only'] is not None and len(analysis['llm_only']) > 0:
            llm_only = analysis['llm_only'].copy()
            llm_only['unique_source'] = 'llm_only'
            unique_records.append(llm_only)

        if analysis['rule_based_only'] is not None and len(analysis['rule_based_only']) > 0:
            rb_only = analysis['rule_based_only'].copy()
            rb_only['unique_source'] = 'rule_based_only'
            unique_records.append(rb_only)

    unique_records_df = pd.concat(unique_records, ignore_index=True) if unique_records else pd.DataFrame()

    # Step 7: Generate output files based on mode
    output_files = {}

    if output_mode in ['blob', 'local']:
        if verbose:
            print(f"\n💾 Generating output files...")
            print()

        # Determine filename prefix
        if week and year:
            prefix = f"OEW{week:02d}-{year}"
        elif weeks and year:
            week_range = f"{min(weeks):02d}-{max(weeks):02d}"
            prefix = f"OEW{week_range}-{year}"
        else:
            prefix = "comparison"

        # Generate files
        files_to_generate = [
            (f"{prefix}_comparison_summary.csv", summary_by_week),
            (f"{prefix}_discrepancies.csv", discrepancy_categories),
            (f"{prefix}_cfr_comparison.csv", cfr_comparison),
            (f"{prefix}_unique_records.csv", unique_records_df),
        ]

        if output_mode == 'blob':
            # Upload to blob storage
            blob_base_path = Config.get_blob_paths()["comparison_outputs"]

            for filename, df in files_to_generate:
                if df is not None and len(df) > 0:
                    blob_path = f"{blob_base_path}{filename}"
                    try:
                        stratus.upload_csv_to_blob(
                            df=df,
                            blob_name=blob_path,
                            stage=stage,
                            container_name=Config.BLOB_CONTAINER,
                        )
                        output_files[filename] = blob_path
                        if verbose:
                            print(f"  ✅ Uploaded {filename} to blob storage")
                    except Exception as e:
                        if verbose:
                            print(f"  ❌ Failed to upload {filename}: {e}")

        elif output_mode == 'local':
            # Save to local directory
            if not output_dir:
                raise ValueError("output_dir required when output_mode='local'")

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            for filename, df in files_to_generate:
                if df is not None and len(df) > 0:
                    file_path = output_path / filename
                    df.to_csv(file_path, index=False)
                    output_files[filename] = str(file_path)
                    if verbose:
                        print(f"  ✅ Saved {filename}")

        if verbose:
            print()

    # Step 8: Return results
    if verbose:
        print("=" * 80)
        print("✅ Comparison complete!")
        print("=" * 80)
        print()

    return {
        'llm_df': llm_df,
        'rule_based_df': rule_based_df,
        'analysis_results': analysis_results,
        'combined_discrepancies': combined_discrepancies,
        'discrepancy_categories': discrepancy_categories,
        'summary_by_week': summary_by_week,
        'cfr_comparison': cfr_comparison,
        'unique_records': unique_records_df,
        'output_files': output_files,
    }


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================


def main():
    """Command-line interface for comparison generation."""
    parser = argparse.ArgumentParser(
        description="Generate comparison reports between LLM and rule-based extractions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare specific week
  python -m src.monitoring.comparisons --week 42 --year 2025 --stage dev

  # Compare specific files
  python -m src.monitoring.comparisons \\
    --llm-file ds-cholera-pdf-scraper/processed/monitoring/llm_extractions/OEW42-2025_gpt-4o_123.csv \\
    --rule-based-file ds-cholera-pdf-scraper/processed/monitoring/rule_based_extractions/OEW42-2025_rb_123.csv \\
    --output ./results

  # Compare multiple weeks
  python -m src.monitoring.comparisons --weeks 42 43 44 --year 2025 --stage dev

  # Save to local directory instead of blob
  python -m src.monitoring.comparisons --week 42 --year 2025 --output ./comparison_results
        """
    )

    # Input arguments
    parser.add_argument('--week', type=int, help='Week number to compare')
    parser.add_argument('--weeks', type=int, nargs='+', help='Multiple week numbers to compare')
    parser.add_argument('--year', type=int, help='Year to compare')
    parser.add_argument('--llm-file', type=str, help='Specific LLM blob path')
    parser.add_argument('--rule-based-file', type=str, help='Specific rule-based blob path')

    # Processing arguments
    parser.add_argument('--stage', type=str, default='dev', choices=['dev', 'prod'],
                        help='Azure stage (default: dev)')
    parser.add_argument('--model', type=str, default=None,
                        help='LLM model name (optional, uses first match if not specified)')
    parser.add_argument('--correct-gap-fill', action='store_true',
                        help='Apply experimental gap-filling corrections')

    # Output arguments
    parser.add_argument('--output', type=str,
                        help='Local output directory (if not specified, uploads to blob)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args()

    # Validate arguments
    if not args.llm_file and not args.week and not args.weeks:
        parser.error("Must provide either --week/--weeks or --llm-file")

    if (args.week or args.weeks) and not args.year:
        parser.error("--year is required when using --week or --weeks")

    if args.llm_file and not args.rule_based_file:
        parser.error("Must provide both --llm-file and --rule-based-file")

    # Determine output mode
    output_mode = 'local' if args.output else 'blob'

    # Run comparison
    try:
        results = generate_comparison_reports(
            week=args.week,
            year=args.year,
            weeks=args.weeks,
            llm_blob_path=args.llm_file,
            rule_based_blob_path=args.rule_based_file,
            stage=args.stage,
            model=args.model,
            correct_gap_fill_errors=args.correct_gap_fill,
            output_mode=output_mode,
            output_dir=args.output,
            verbose=not args.quiet,
        )

        # Print summary
        if not args.quiet:
            if results['output_files']:
                print("Generated files:")
                for filename, path in results['output_files'].items():
                    print(f"  - {filename}")
                    if output_mode == 'blob':
                        print(f"    {path}")
                    else:
                        print(f"    {path}")

        sys.exit(0)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
