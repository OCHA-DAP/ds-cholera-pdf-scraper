"""
Real-time Comparison: Rule-Based vs LLM Extractions from Blob Storage

This module provides functions to load and compare rule-based and LLM-based
extractions directly from Azure Blob Storage without requiring local files.

Key Features:
- Load rule-based extractions from blob storage
- Load LLM extractions from blob storage
- Perform discrepancy analysis using existing src.batch_run modules
- CFR consistency checking to validate accuracy
- Timeline visualizations with CFR winner highlighting
- Support for filtering by week ranges, years, and date ranges

Usage:
    from book_cholera_scraping.realtime_comparison import load_and_compare

    # Compare all available data
    results = load_and_compare(stage='dev')

    # Compare specific weeks
    results = load_and_compare(
        stage='dev',
        weeks=[42, 43, 44],
        year=2025
    )

    # Compare date range
    results = load_and_compare(
        stage='dev',
        start_date='2025-01-01',
        end_date='2025-12-31'
    )
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd

# Setup path to import from src
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.config import Config
from src.cloud_logging import DuckDBCloudQuery
from src.batch_run import (
    analyze_llm_vs_rule_based,
    categorize_discrepancies,
    create_summary_statistics,
)
from src.batch_run.loader import standardize_column_names

# Optional visualization imports (requires plotly)
try:
    from src.batch_run.visualization import (
        check_cfr_consistency,
        create_individual_timeline_plots,
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    # Create dummy functions
    def check_cfr_consistency(*args, **kwargs):
        raise ImportError("Visualization requires plotly. Install with: pip install plotly")
    def create_individual_timeline_plots(*args, **kwargs):
        raise ImportError("Visualization requires plotly. Install with: pip install plotly")


# =============================================================================
# BLOB DATA LOADERS
# =============================================================================


class BlobExtractionLoader:
    """
    Load extraction results from Azure Blob Storage.

    Supports loading both rule-based and LLM-based extractions.
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

        # Initialize DuckDB for cloud querying
        self.duckdb_query = DuckDBCloudQuery(stage=self.stage)

    def list_available_rule_based_extractions(self) -> pd.DataFrame:
        """
        List all available rule-based extractions in blob storage.

        Returns:
            DataFrame with columns: filename, week, year, blob_name
        """
        import ocha_stratus as stratus

        # List rule-based extractions using stratus
        blob_prefix = f"{self.proj_dir}/processed/rule_based_extractions/"
        blob_names = stratus.list_container_blobs(
            name_starts_with=blob_prefix,
            stage=self.stage,
            container_name=self.container
        )

        # Parse blob names to extract week/year/run_id
        extraction_info = []
        for blob_name in blob_names:
            if blob_name.endswith('.csv'):
                filename = Path(blob_name).name
                # Parse filename like: OEW42-2025_rule-based_20251117_140523.csv
                # or: OEW42-2025_rule-based_1762895975.csv (unix timestamp)
                parts = filename.replace('.csv', '').split('_')
                if parts[0].startswith('OEW'):
                    week_year = parts[0].replace('OEW', '')
                    if '-' in week_year:
                        week, year = week_year.split('-')
                        # Extract run_id (last numeric part after split by _)
                        run_id = 0
                        if len(parts) >= 3:
                            # Try to parse the last part as integer (unix timestamp or run number)
                            try:
                                run_id = int(parts[-1])
                            except ValueError:
                                pass
                        extraction_info.append({
                            'blob_name': blob_name,
                            'filename': filename,
                            'week': int(week),
                            'year': int(year),
                            'run_id': run_id,
                        })

        return pd.DataFrame(extraction_info).sort_values(['year', 'week', 'run_id'], ascending=False)

    def list_available_llm_extractions(self) -> pd.DataFrame:
        """
        List all available LLM extractions in blob storage.

        Returns:
            DataFrame with columns: filename, week, year, model, blob_name
        """
        import ocha_stratus as stratus

        # List LLM extractions using stratus
        blob_prefix = f"{self.proj_dir}/processed/llm_extractions/"
        blob_names = stratus.list_container_blobs(
            name_starts_with=blob_prefix,
            stage=self.stage,
            container_name=self.container
        )

        # Parse blob names to extract week/year/model/run_id
        extraction_info = []
        for blob_name in blob_names:
            if blob_name.endswith('.csv'):
                filename = Path(blob_name).name
                # Parse filename like: OEW42-2025_gpt-5_210.csv or OEW42-2025_gpt-5_20251117_140523.csv
                parts = filename.replace('.csv', '').split('_')
                if parts[0].startswith('OEW'):
                    week_year = parts[0].replace('OEW', '')
                    if '-' in week_year:
                        week, year = week_year.split('-')
                        model = parts[1] if len(parts) > 1 else 'unknown'
                        # Extract run_id (last numeric part)
                        run_id = 0
                        if len(parts) >= 3:
                            try:
                                run_id = int(parts[-1])
                            except ValueError:
                                pass
                        extraction_info.append({
                            'blob_name': blob_name,
                            'filename': filename,
                            'week': int(week),
                            'year': int(year),
                            'model': model,
                            'run_id': run_id,
                        })

        return pd.DataFrame(extraction_info).sort_values(['year', 'week', 'run_id'], ascending=False)

    def load_rule_based_extraction(
        self,
        week: Optional[int] = None,
        year: Optional[int] = None,
        filename: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load a specific rule-based extraction from blob.

        Args:
            week: Week number (optional if filename provided)
            year: Year (optional if filename provided)
            filename: Specific blob filename (optional)

        Returns:
            DataFrame with extraction data
        """
        import ocha_stratus as stratus

        if filename:
            blob_name = f"{self.proj_dir}/processed/rule_based_extractions/{filename}"
        elif week and year:
            # Find extraction for this week/year
            available = self.list_available_rule_based_extractions()
            matches = available[
                (available['week'] == week) & (available['year'] == year)
            ]

            if len(matches) == 0:
                raise FileNotFoundError(
                    f"No rule-based extraction found for Week {week}, Year {year}"
                )

            # Take first match (already sorted by run_id descending)
            blob_name = matches.iloc[0]['blob_name']
        else:
            raise ValueError("Must provide either (week + year) or filename")

        # Load from blob using stratus
        print(f"📥 Loading rule-based extraction: {Path(blob_name).name}")
        df = stratus.load_csv_from_blob(
            blob_name=blob_name,
            stage=self.stage,
            container_name=self.container
        )

        # Standardize column names
        df = standardize_column_names(df, is_baseline=False)

        # Add metadata if not present
        if week and 'WeekNumber' not in df.columns:
            df['WeekNumber'] = week
        if year and 'Year' not in df.columns:
            df['Year'] = year

        print(f"✓ Loaded {len(df)} records")
        return df

    def load_llm_extraction(
        self,
        week: Optional[int] = None,
        year: Optional[int] = None,
        model: str = 'gpt-5',
        filename: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load a specific LLM extraction from blob.

        Args:
            week: Week number (optional if filename provided)
            year: Year (optional if filename provided)
            model: Model name (default: 'gpt-5')
            filename: Specific blob filename (optional)

        Returns:
            DataFrame with extraction data
        """
        import ocha_stratus as stratus

        if filename:
            blob_name = f"{self.proj_dir}/processed/llm_extractions/{filename}"
        elif week and year:
            # Find extraction for this week/year/model
            available = self.list_available_llm_extractions()
            matches = available[
                (available['week'] == week) &
                (available['year'] == year) &
                (available['model'] == model)
            ]

            if len(matches) == 0:
                raise FileNotFoundError(
                    f"No LLM extraction found for Week {week}, Year {year}, Model {model}"
                )

            # Take first match (already sorted by run_id descending)
            blob_name = matches.iloc[0]['blob_name']
        else:
            raise ValueError("Must provide either (week + year) or filename")

        # Load from blob using stratus
        print(f"📥 Loading LLM extraction: {Path(blob_name).name}")
        df = stratus.load_csv_from_blob(
            blob_name=blob_name,
            stage=self.stage,
            container_name=self.container
        )

        # Standardize column names
        df = standardize_column_names(df, is_baseline=False)

        # Add metadata if not present
        if week and 'WeekNumber' not in df.columns:
            df['WeekNumber'] = week
        if year and 'Year' not in df.columns:
            df['Year'] = year

        print(f"✓ Loaded {len(df)} records")
        return df

    def load_rule_based_bulk(
        self,
        weeks: Optional[List[int]] = None,
        year: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load multiple rule-based extractions in bulk.

        Args:
            weeks: List of week numbers (optional)
            year: Year to filter (required if weeks provided)
            start_date: Start date YYYY-MM-DD (optional)
            end_date: End date YYYY-MM-DD (optional)

        Returns:
            Combined DataFrame
        """
        available = self.list_available_rule_based_extractions()

        # Filter by criteria
        if weeks and year:
            available = available[
                (available['week'].isin(weeks)) & (available['year'] == year)
            ]
        # Note: start_date/end_date filtering not supported without last_modified column
        # Use weeks/year filtering instead

        if len(available) == 0:
            raise FileNotFoundError("No rule-based extractions found matching criteria")

        # Get unique week/year combinations and take most recent (highest run_id)
        # Already sorted by run_id descending, so take first for each group
        unique_weeks = available.groupby(['week', 'year']).first().reset_index()

        # Load each extraction (one per week/year)
        dfs = []
        for _, row in unique_weeks.iterrows():
            try:
                df = self.load_rule_based_extraction(
                    week=row['week'],
                    year=row['year']
                )
                df['SourceFile'] = row['filename']
                dfs.append(df)
            except Exception as e:
                print(f"⚠️  Failed to load {row['filename']}: {e}")

        if not dfs:
            raise ValueError("Failed to load any extractions")

        combined = pd.concat(dfs, ignore_index=True)
        print(f"\n✓ Loaded {len(dfs)} rule-based extractions ({len(combined)} total records)")
        return combined

    def load_llm_bulk(
        self,
        weeks: Optional[List[int]] = None,
        year: Optional[int] = None,
        model: str = 'gpt-5',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load multiple LLM extractions in bulk.

        Args:
            weeks: List of week numbers (optional)
            year: Year to filter (required if weeks provided)
            model: Model name (default: 'gpt-5')
            start_date: Start date YYYY-MM-DD (optional)
            end_date: End date YYYY-MM-DD (optional)

        Returns:
            Combined DataFrame
        """
        available = self.list_available_llm_extractions()

        # Filter by criteria
        available = available[available['model'] == model]

        if weeks and year:
            available = available[
                (available['week'].isin(weeks)) & (available['year'] == year)
            ]
        # Note: start_date/end_date filtering not supported without last_modified column
        # Use weeks/year filtering instead

        if len(available) == 0:
            raise FileNotFoundError(
                f"No LLM extractions found for model {model} matching criteria"
            )

        # Get unique week/year/model combinations and take most recent (highest run_id)
        # Already sorted by run_id descending, so take first for each group
        unique_weeks = available.groupby(['week', 'year', 'model']).first().reset_index()

        # Load each extraction (one per week/year/model)
        dfs = []
        for _, row in unique_weeks.iterrows():
            try:
                df = self.load_llm_extraction(
                    week=row['week'],
                    year=row['year'],
                    model=row['model']
                )
                df['SourceFile'] = row['filename']
                dfs.append(df)
            except Exception as e:
                print(f"⚠️  Failed to load {row['filename']}: {e}")

        if not dfs:
            raise ValueError("Failed to load any extractions")

        combined = pd.concat(dfs, ignore_index=True)
        print(f"\n✓ Loaded {len(dfs)} LLM extractions ({len(combined)} total records)")
        return combined


# =============================================================================
# COMPARISON FUNCTIONS
# =============================================================================


def load_and_compare(
    stage: str = 'dev',
    weeks: Optional[List[int]] = None,
    year: Optional[int] = None,
    model: str = 'gpt-5',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    correct_gap_fill_errors: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Load and compare rule-based vs LLM extractions from blob storage.

    Args:
        stage: Azure stage (dev/prod)
        weeks: List of week numbers to compare (optional)
        year: Year to filter (required if weeks provided)
        model: LLM model name (default: 'gpt-5')
        start_date: Start date YYYY-MM-DD (optional)
        end_date: End date YYYY-MM-DD (optional)
        correct_gap_fill_errors: Apply experimental gap-filling corrections
        verbose: Print progress information

    Returns:
        Dictionary with comparison results:
            - rule_based_df: Rule-based extraction data
            - llm_df: LLM extraction data
            - analysis_results: Detailed analysis by week
            - combined_discrepancies: All discrepancies combined
            - discrepancy_categories: Categorized discrepancies
            - summary_by_week: Summary statistics
            - cfr_comparison: CFR consistency comparison

    Example:
        >>> results = load_and_compare(
        ...     stage='dev',
        ...     weeks=[42, 43, 44],
        ...     year=2025,
        ...     model='gpt-5'
        ... )
        >>> print(results['summary_by_week'])
    """
    if verbose:
        print("=" * 80)
        print("REAL-TIME COMPARISON: Rule-Based vs LLM Extractions")
        print("=" * 80)
        print(f"Stage: {stage}")
        print(f"Model: {model}")
        if weeks:
            print(f"Weeks: {weeks} (Year {year})")
        elif start_date or end_date:
            print(f"Date range: {start_date or 'earliest'} to {end_date or 'latest'}")
        else:
            print("Loading all available data")
        print("=" * 80)
        print()

    # Initialize loader
    loader = BlobExtractionLoader(stage=stage)

    # Load data
    if verbose:
        print("📦 Loading extraction data from blob storage...")
        print()

    # Load rule-based data
    try:
        if weeks and year:
            rule_based_df = loader.load_rule_based_bulk(weeks=weeks, year=year)
        elif start_date or end_date:
            rule_based_df = loader.load_rule_based_bulk(
                start_date=start_date,
                end_date=end_date
            )
        else:
            # Load all available
            available_rb = loader.list_available_rule_based_extractions()
            if len(available_rb) == 0:
                raise FileNotFoundError("No rule-based extractions found")

            # Get unique week/year combinations
            week_years = available_rb[['week', 'year']].drop_duplicates()
            rule_based_df = loader.load_rule_based_bulk(
                weeks=week_years['week'].tolist(),
                year=week_years['year'].iloc[0] if len(week_years['year'].unique()) == 1 else None
            )
    except Exception as e:
        print(f"❌ Failed to load rule-based data: {e}")
        return None

    # Load LLM data
    try:
        if weeks and year:
            llm_df = loader.load_llm_bulk(weeks=weeks, year=year, model=model)
        elif start_date or end_date:
            llm_df = loader.load_llm_bulk(
                start_date=start_date,
                end_date=end_date,
                model=model
            )
        else:
            # Load all available
            available_llm = loader.list_available_llm_extractions()
            available_llm = available_llm[available_llm['model'] == model]

            if len(available_llm) == 0:
                raise FileNotFoundError(f"No LLM extractions found for model {model}")

            # Get unique week/year combinations
            week_years = available_llm[['week', 'year']].drop_duplicates()
            llm_df = loader.load_llm_bulk(
                weeks=week_years['week'].tolist(),
                year=week_years['year'].iloc[0] if len(week_years['year'].unique()) == 1 else None,
                model=model
            )
    except Exception as e:
        print(f"❌ Failed to load LLM data: {e}")
        return None

    # Perform analysis
    if verbose:
        print("\n🔍 Analyzing discrepancies...")
        print()

    results, combined_discrepancies = analyze_llm_vs_rule_based(
        llm_df=llm_df,
        rule_based_df=rule_based_df,
        correct_gap_fill_errors=correct_gap_fill_errors,
        verbose=verbose
    )

    # Create summary statistics
    summary_by_week = create_summary_statistics(results)

    # Categorize discrepancies
    disc_cat = None
    if combined_discrepancies is not None and len(combined_discrepancies) > 0:
        disc_cat = categorize_discrepancies(combined_discrepancies)

        if verbose:
            print(f"\n📊 Discrepancy breakdown:")
            print(disc_cat.groupby('Category').size())

    # CFR consistency comparison (requires visualization module)
    cfr_comparison = None
    if disc_cat is not None and len(disc_cat) > 0 and VISUALIZATION_AVAILABLE:
        try:
            cfr_comparison = compare_cfr_consistency(
                disc_cat, llm_df, rule_based_df, verbose=verbose
            )
        except ImportError as e:
            if verbose:
                print(f"\n⚠️  CFR comparison skipped: {e}")

    if verbose:
        print("\n" + "=" * 80)
        print("✅ Comparison complete!")
        print("=" * 80)

    return {
        'rule_based_df': rule_based_df,
        'llm_df': llm_df,
        'analysis_results': results,
        'combined_discrepancies': combined_discrepancies,
        'discrepancy_categories': disc_cat,
        'summary_by_week': summary_by_week,
        'cfr_comparison': cfr_comparison,
        'loader': loader
    }


def compare_cfr_consistency(
    disc_cat: pd.DataFrame,
    llm_df: pd.DataFrame,
    rule_based_df: pd.DataFrame,
    parameter: str = 'TotalCases',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare CFR consistency between LLM and rule-based for discrepancies.

    Args:
        disc_cat: Categorized discrepancies DataFrame
        llm_df: LLM extraction data
        rule_based_df: Rule-based extraction data
        parameter: Parameter to analyze (default: 'TotalCases')
        verbose: Print results

    Returns:
        DataFrame with CFR comparison results
    """
    # Filter for specific parameter
    param_disc = disc_cat[disc_cat['Parameter'] == parameter].copy()

    if len(param_disc) == 0:
        if verbose:
            print(f"No {parameter} discrepancies found for CFR analysis")
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

        print(f"\n📊 CFR Consistency Analysis ({parameter}):")
        print(f"   Total discrepancies with CFR data: {total}")
        print(f"   LLM has better CFR consistency: {llm_wins} ({llm_wins/total*100:.1f}%)")
        print(f"   Rule-based has better CFR consistency: {rb_wins} ({rb_wins/total*100:.1f}%)")

    return cfr_df


def create_comparison_visualizations(
    comparison_results: Dict[str, Any],
    parameter: str = 'TotalCases',
    n_top: Optional[int] = None,
    highlight_cfr_winner: bool = True,
    height: int = 400
) -> List:
    """
    Create timeline visualizations for comparison results.

    Args:
        comparison_results: Results from load_and_compare()
        parameter: Parameter to visualize (default: 'TotalCases')
        n_top: Number of top discrepancies (None = all)
        highlight_cfr_winner: Highlight CFR winner with lime markers
        height: Plot height in pixels

    Returns:
        List of plotly figures

    Note:
        Requires plotly to be installed
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError(
            "plotly is required for visualizations. Install with: pip install plotly"
        )

    disc_cat = comparison_results['discrepancy_categories']
    llm_df = comparison_results['llm_df']
    rule_based_df = comparison_results['rule_based_df']

    if disc_cat is None or len(disc_cat) == 0:
        print("No discrepancies to visualize")
        return []

    figures = create_individual_timeline_plots(
        disc_cat=disc_cat,
        llm_df=llm_df,
        rule_based_df=rule_based_df,
        parameter=parameter,
        n_top=n_top,
        highlight_cfr_winner=highlight_cfr_winner,
        height=height
    )

    return figures


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================


def main():
    """Command-line interface for real-time comparison."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare rule-based and LLM extractions from blob storage"
    )
    parser.add_argument(
        '--stage', type=str, default='dev',
        help='Azure stage (dev/prod)'
    )
    parser.add_argument(
        '--weeks', type=int, nargs='+',
        help='Week numbers to compare'
    )
    parser.add_argument(
        '--year', type=int,
        help='Year to filter'
    )
    parser.add_argument(
        '--model', type=str, default='gpt-5',
        help='LLM model name'
    )
    parser.add_argument(
        '--start-date', type=str,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date', type=str,
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--output', type=str,
        help='Output directory for results (optional)'
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='Create visualizations'
    )

    args = parser.parse_args()

    # Run comparison
    results = load_and_compare(
        stage=args.stage,
        weeks=args.weeks,
        year=args.year,
        model=args.model,
        start_date=args.start_date,
        end_date=args.end_date,
        verbose=True
    )

    if results is None:
        print("❌ Comparison failed")
        sys.exit(1)

    # Save results if output directory specified
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n💾 Saving results to {output_dir}")

        # Save summary
        if results['summary_by_week'] is not None:
            summary_path = output_dir / 'summary_by_week.csv'
            results['summary_by_week'].to_csv(summary_path, index=False)
            print(f"   ✓ Saved summary to {summary_path.name}")

        # Save discrepancies
        if results['discrepancy_categories'] is not None:
            disc_path = output_dir / 'discrepancies.csv'
            results['discrepancy_categories'].to_csv(disc_path, index=False)
            disc_count = len(results['discrepancy_categories'])
            print(f"   ✓ Saved {disc_count} discrepancies to {disc_path.name}")

        # Save CFR comparison
        if results['cfr_comparison'] is not None:
            cfr_path = output_dir / 'cfr_comparison.csv'
            results['cfr_comparison'].to_csv(cfr_path, index=False)
            print(f"   ✓ Saved CFR comparison to {cfr_path.name}")

        # Save unique records (LLM-only and Rule-based-only combined)
        if results['analysis_results'] and len(results['analysis_results']) > 0:
            unique_records = []

            for analysis in results['analysis_results']:
                # Add LLM-only records
                if analysis['llm_only'] is not None and len(analysis['llm_only']) > 0:
                    llm_only = analysis['llm_only'].copy()
                    llm_only['unique_source'] = 'llm_only'
                    unique_records.append(llm_only)

                # Add Rule-based-only records
                if analysis['baseline_only'] is not None and len(analysis['baseline_only']) > 0:
                    baseline_only = analysis['baseline_only'].copy()
                    baseline_only['unique_source'] = 'rule_based_only'
                    unique_records.append(baseline_only)

            if unique_records:
                combined_unique = pd.concat(unique_records, ignore_index=True)
                unique_path = output_dir / 'unique_records.csv'
                combined_unique.to_csv(unique_path, index=False)
                llm_only_count = len(combined_unique[combined_unique['unique_source'] == 'llm_only'])
                rule_based_only_count = len(combined_unique[combined_unique['unique_source'] == 'rule_based_only'])
                print(f"   ✓ Saved {len(combined_unique)} unique records to {unique_path.name}")
                print(f"      - llm_only: {llm_only_count}")
                print(f"      - rule_based_only: {rule_based_only_count}")

    # Create visualizations
    if args.visualize:
        print("\n📊 Creating visualizations...")
        try:
            figures = create_comparison_visualizations(results)
        except ImportError as e:
            print(f"❌ Visualization failed: {e}")
            sys.exit(1)

        if args.output and figures:
            try:
                import plotly.io as pio
            except ImportError:
                print("❌ plotly is required for saving visualizations")
                sys.exit(1)
            viz_dir = Path(args.output) / 'visualizations'
            viz_dir.mkdir(exist_ok=True)

            for i, fig in enumerate(figures, 1):
                html_path = viz_dir / f'timeline_{i:02d}.html'
                pio.write_html(fig, html_path)

            print(f"   ✓ Saved {len(figures)} visualizations to {viz_dir}")


if __name__ == '__main__':
    main()
