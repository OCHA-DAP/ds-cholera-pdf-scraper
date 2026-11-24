"""
Compare LLM extraction output against baseline data.

This module handles comparison between new LLM DataFrame and baseline,
reporting discrepancies and validation metrics.
"""

import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataComparator:
    """Compares LLM extraction results against baseline data."""

    def __init__(self, tolerance: float = 0.05):
        """
        Initialize the data comparator.

        Args:
            tolerance: Tolerance for numerical comparisons (5% default)
        """
        self.tolerance = tolerance
        self.comparison_results = {}

    def load_dataframes(
        self, llm_data_path: str, baseline_path: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both LLM and baseline DataFrames.

        Args:
            llm_data_path: Path to LLM extraction results
            baseline_path: Path to baseline data

        Returns:
            Tuple of (llm_df, baseline_df)
        """
        logger.info(f"Loading LLM data from {llm_data_path}")
        llm_df = pd.read_csv(llm_data_path)

        logger.info(f"Loading baseline data from {baseline_path}")
        baseline_df = pd.read_csv(baseline_path)

        return llm_df, baseline_df

    def align_dataframes(
        self,
        llm_df: pd.DataFrame,
        baseline_df: pd.DataFrame,
        key_columns: List[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align DataFrames for comparison using key columns.

        Args:
            llm_df: LLM extraction DataFrame
            baseline_df: Baseline DataFrame
            key_columns: Columns to use for alignment (e.g., date, country)

        Returns:
            Tuple of aligned DataFrames
        """
        if key_columns is None:
            key_columns = ["reporting_date", "country", "admin1"]

        logger.info(f"Aligning DataFrames on columns: {key_columns}")

        # Ensure key columns exist in both DataFrames
        available_keys = [
            col
            for col in key_columns
            if col in llm_df.columns and col in baseline_df.columns
        ]

        if not available_keys:
            logger.warning("No common key columns found for alignment")
            return llm_df, baseline_df

        # Merge DataFrames to find common records
        aligned_llm = llm_df.copy()
        aligned_baseline = baseline_df.copy()

        # Create comparison keys
        for df in [aligned_llm, aligned_baseline]:
            df["_comparison_key"] = df[available_keys].astype(str).agg("|".join, axis=1)

        # Find common keys
        common_keys = set(aligned_llm["_comparison_key"]) & set(
            aligned_baseline["_comparison_key"]
        )

        logger.info(f"Found {len(common_keys)} common records for comparison")

        # Filter to common records
        aligned_llm = aligned_llm[
            aligned_llm["_comparison_key"].isin(common_keys)
        ].copy()
        aligned_baseline = aligned_baseline[
            aligned_baseline["_comparison_key"].isin(common_keys)
        ].copy()

        # Sort by comparison key for aligned comparison
        aligned_llm = aligned_llm.sort_values("_comparison_key").reset_index(drop=True)
        aligned_baseline = aligned_baseline.sort_values("_comparison_key").reset_index(
            drop=True
        )

        return aligned_llm, aligned_baseline

    def compare_numerical_columns(
        self,
        llm_df: pd.DataFrame,
        baseline_df: pd.DataFrame,
        numerical_columns: List[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare numerical columns between DataFrames.

        Args:
            llm_df: LLM extraction DataFrame
            baseline_df: Baseline DataFrame
            numerical_columns: List of numerical columns to compare

        Returns:
            Dictionary with comparison results for each column
        """
        if numerical_columns is None:
            numerical_columns = [
                "suspected_cases",
                "confirmed_cases",
                "deaths",
                "case_fatality_rate",
                "population_at_risk",
            ]

        results = {}

        for col in numerical_columns:
            if col not in llm_df.columns or col not in baseline_df.columns:
                logger.warning(f"Column {col} not found in both DataFrames")
                continue

            logger.info(f"Comparing numerical column: {col}")

            llm_values = pd.to_numeric(llm_df[col], errors="coerce")
            baseline_values = pd.to_numeric(baseline_df[col], errors="coerce")

            # Calculate metrics
            mae = np.mean(np.abs(llm_values - baseline_values))
            rmse = np.sqrt(np.mean((llm_values - baseline_values) ** 2))

            # Percentage accuracy within tolerance
            relative_error = np.abs(
                (llm_values - baseline_values) / (baseline_values + 1e-10)
            )
            within_tolerance = np.mean(relative_error <= self.tolerance) * 100

            # Correlation
            correlation = llm_values.corr(baseline_values)

            results[col] = {
                "mae": mae,
                "rmse": rmse,
                "correlation": correlation,
                "within_tolerance_pct": within_tolerance,
                "total_records": len(llm_values),
                "llm_mean": llm_values.mean(),
                "baseline_mean": baseline_values.mean(),
            }

            logger.info(
                f"{col}: MAE={mae:.2f}, RMSE={rmse:.2f}, "
                f"Correlation={correlation:.3f}, "
                f"Within tolerance={within_tolerance:.1f}%"
            )

        return results

    def compare_categorical_columns(
        self,
        llm_df: pd.DataFrame,
        baseline_df: pd.DataFrame,
        categorical_columns: List[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare categorical columns between DataFrames.

        Args:
            llm_df: LLM extraction DataFrame
            baseline_df: Baseline DataFrame
            categorical_columns: List of categorical columns to compare

        Returns:
            Dictionary with comparison results for each column
        """
        if categorical_columns is None:
            categorical_columns = ["country", "admin1", "admin2"]

        results = {}

        for col in categorical_columns:
            if col not in llm_df.columns or col not in baseline_df.columns:
                logger.warning(f"Column {col} not found in both DataFrames")
                continue

            logger.info(f"Comparing categorical column: {col}")

            llm_values = llm_df[col].fillna("").astype(str)
            baseline_values = baseline_df[col].fillna("").astype(str)

            # Exact match accuracy
            exact_matches = (llm_values == baseline_values).sum()
            accuracy = exact_matches / len(llm_values) * 100

            # Unique values comparison
            llm_unique = set(llm_values.unique())
            baseline_unique = set(baseline_values.unique())

            results[col] = {
                "exact_match_pct": accuracy,
                "total_records": len(llm_values),
                "llm_unique_values": len(llm_unique),
                "baseline_unique_values": len(baseline_unique),
                "common_values": len(llm_unique & baseline_unique),
                "llm_only_values": list(llm_unique - baseline_unique),
                "baseline_only_values": list(baseline_unique - llm_unique),
            }

            logger.info(
                f"{col}: Exact match={accuracy:.1f}%, "
                f"LLM unique={len(llm_unique)}, "
                f"Baseline unique={len(baseline_unique)}"
            )

        return results

    def generate_comparison_report(
        self, llm_data_path: str, baseline_path: str, output_path: str = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive comparison report.

        Args:
            llm_data_path: Path to LLM extraction results
            baseline_path: Path to baseline data
            output_path: Optional path to save report

        Returns:
            Complete comparison report dictionary
        """
        logger.info("Generating comprehensive comparison report")

        # Load and align data
        llm_df, baseline_df = self.load_dataframes(llm_data_path, baseline_path)
        aligned_llm, aligned_baseline = self.align_dataframes(llm_df, baseline_df)

        # Compare numerical and categorical columns
        numerical_results = self.compare_numerical_columns(
            aligned_llm, aligned_baseline
        )
        categorical_results = self.compare_categorical_columns(
            aligned_llm, aligned_baseline
        )

        # Overall summary
        report = {
            "summary": {
                "llm_total_records": len(llm_df),
                "baseline_total_records": len(baseline_df),
                "aligned_records": len(aligned_llm),
                "comparison_timestamp": pd.Timestamp.now().isoformat(),
            },
            "numerical_comparison": numerical_results,
            "categorical_comparison": categorical_results,
        }

        # Save report if requested
        if output_path:
            self.save_report(report, output_path)

        logger.info("Comparison report generated successfully")
        return report

    def save_report(self, report: Dict[str, Any], output_path: str) -> None:
        """
        Save comparison report to file.

        Args:
            report: Comparison report dictionary
            output_path: Path to save the report
        """
        import json

        logger.info(f"Saving comparison report to {output_path}")

        try:
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Report saved successfully to {output_path}")

        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            raise


def main():
    """Main execution function for testing."""
    logging.basicConfig(level=logging.INFO)

    # Example usage
    comparator = DataComparator(tolerance=0.1)  # 10% tolerance

    # Placeholder paths - replace with actual data
    llm_results_path = "outputs/llm_extraction_results.csv"
    baseline_path = "data/baseline_extraction.csv"
    report_path = "outputs/comparison_report.json"

    if Path(llm_results_path).exists() and Path(baseline_path).exists():
        report = comparator.generate_comparison_report(
            llm_results_path, baseline_path, report_path
        )
        print(f"Comparison completed. Report saved to {report_path}")
    else:
        print("Sample data files not found. Please provide actual data paths.")


FIELDS_TO_COMPARE = ["TotalCases", "CasesConfirmed", "Deaths", "CFR", "Grade"]

# Extra columns used only to decide which rows are the same record when several
# rows share a comparison key. They are not reported as discrepancies.
PAIRING_ONLY_FIELDS = ["DateNotified", "StartReportingPeriod", "EndReportingPeriod"]


def _values_match(val1, val2, tolerance=0.01):
    """Check if two values match, handling NaN and numerical comparisons."""
    if pd.isna(val1) and pd.isna(val2):
        return True
    elif pd.isna(val1) or pd.isna(val2):
        return False
    else:
        try:
            # For numerical comparison
            num1 = float(val1)
            num2 = float(val2)
            return abs(num1 - num2) <= tolerance
        except:
            # For string comparison
            return str(val1).strip() == str(val2).strip()


def _normalise_key_part(value) -> str:
    """Reduce a country or event name to lower-case ASCII letters and digits.

    The two extractions spell the same record differently ("Côte d'Ivoire" vs
    "Cote d'Ivoire", "fever" vs "Fever", "(cVDPV2)" vs "(c VDPV2)"). Matching on
    this reduced form keeps those from being reported as records that exist in
    only one extraction.
    """
    ascii_text = (
        unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    )
    return re.sub(r"[^a-z0-9]", "", ascii_text.lower())


def _assign_pair_ids(llm_common: pd.DataFrame, rule_based_common: pd.DataFrame):
    """Pair LLM and rule-based rows one-to-one within each comparison key.

    A key is not always unique: a bulletin can list the same country and event
    twice (two separate humanitarian crises, say). Merging on the key alone then
    compares every LLM row with every rule-based row and reports the mismatched
    combinations as discrepancies.

    Within each key, pairs are taken in order of fewest differing fields, so a
    row is paired with its closest counterpart. A row left without a partner
    (one extraction has more rows for that key than the other) gets no pair id.

    Returns:
        Two Series of pair ids (nullable integers), indexed like the inputs.
    """
    fields = [
        field
        for field in FIELDS_TO_COMPARE + PAIRING_ONLY_FIELDS
        if field in llm_common and field in rule_based_common
    ]
    llm_pair_ids = pd.Series(pd.NA, index=llm_common.index, dtype="Int64")
    rule_based_pair_ids = pd.Series(pd.NA, index=rule_based_common.index, dtype="Int64")
    rule_based_groups = rule_based_common.groupby("comparison_key").groups

    for key, llm_index in llm_common.groupby("comparison_key").groups.items():
        # (number of differing fields, LLM row, rule-based row), closest first.
        # Row labels break ties, so the result does not depend on set ordering.
        candidates = sorted(
            (
                sum(
                    not _values_match(
                        llm_common.at[i, field], rule_based_common.at[j, field]
                    )
                    for field in fields
                ),
                i,
                j,
            )
            for i in llm_index
            for j in rule_based_groups[key]
        )
        paired_llm, paired_rule_based = set(), set()
        for _, i, j in candidates:
            if i in paired_llm or j in paired_rule_based:
                continue
            llm_pair_ids.at[i] = rule_based_pair_ids.at[j] = len(paired_llm)
            paired_llm.add(i)
            paired_rule_based.add(j)

    return llm_pair_ids, rule_based_pair_ids


def perform_discrepancy_analysis(llm_data: pd.DataFrame, rule_based_data: pd.DataFrame):
    """
    Perform comprehensive discrepancy analysis between LLM and rule-based data.

    This function applies post-processing to both datasets, creates comparison keys,
    and identifies discrepancies in key fields.

    Args:
        llm_data: DataFrame with LLM extraction results
        rule_based_data: DataFrame with rule-based/reference data

    Returns:
        Tuple containing:
        - discrepancies_df: DataFrame with detailed discrepancy analysis
        - llm_common: LLM records that have matching rule-based records
        - llm_only_df: LLM records not found in rule-based
        - rule_based_only_df: Rule-based records not found in LLM

    Records are matched on country and event, compared in a spelling-insensitive
    form. When several rows share a country and event they are paired
    one-to-one (see _assign_pair_ids); a row without a partner is returned in
    llm_only_df or rule_based_only_df.
    """
    # Import post-processing pipeline
    from .post_processing import apply_post_processing_pipeline

    # Apply post-processing
    llm_processed = apply_post_processing_pipeline(llm_data.copy(), source="llm")
    rule_based_processed = apply_post_processing_pipeline(
        rule_based_data.copy(), source="baseline"
    )

    # Create comparison keys
    for processed in (llm_processed, rule_based_processed):
        processed["comparison_key"] = (
            processed["Country"].map(_normalise_key_part)
            + "_"
            + processed["Event"].map(_normalise_key_part)
        )

    # Find common and unique records
    llm_keys = set(llm_processed["comparison_key"])
    rule_based_keys = set(rule_based_processed["comparison_key"])

    common_keys = llm_keys & rule_based_keys
    llm_only_keys = llm_keys - rule_based_keys
    rule_based_only_keys = rule_based_keys - llm_keys

    # Create comparison datasets
    llm_common = llm_processed[llm_processed["comparison_key"].isin(common_keys)].copy()
    rule_based_common = rule_based_processed[
        rule_based_processed["comparison_key"].isin(common_keys)
    ].copy()
    llm_only_df = llm_processed[
        llm_processed["comparison_key"].isin(llm_only_keys)
    ].copy()
    rule_based_only_df = rule_based_processed[
        rule_based_processed["comparison_key"].isin(rule_based_only_keys)
    ].copy()

    # Sort for alignment
    llm_common = llm_common.sort_values("comparison_key").reset_index(drop=True)
    rule_based_common = rule_based_common.sort_values("comparison_key").reset_index(
        drop=True
    )

    # Pair rows one-to-one within each key, then move rows without a partner
    # to the "only in one extraction" outputs
    llm_common["_pair_id"], rule_based_common["_pair_id"] = _assign_pair_ids(
        llm_common, rule_based_common
    )
    llm_only_df = pd.concat(
        [llm_only_df, llm_common[llm_common["_pair_id"].isna()]]
    ).drop(columns="_pair_id")
    rule_based_only_df = pd.concat(
        [rule_based_only_df, rule_based_common[rule_based_common["_pair_id"].isna()]]
    ).drop(columns="_pair_id")
    llm_common = llm_common[llm_common["_pair_id"].notna()]
    rule_based_common = rule_based_common[rule_based_common["_pair_id"].notna()]

    merged_data = llm_common.merge(
        rule_based_common,
        on=["comparison_key", "_pair_id"],
        suffixes=("_llm", "_rule_based"),
        how="inner",
        validate="one_to_one",
    )
    llm_common = llm_common.drop(columns="_pair_id")

    # Compare fields and create discrepancy records
    discrepant_records = []
    fields_to_compare = FIELDS_TO_COMPARE
    values_match = _values_match

    for i in range(len(merged_data)):
        row = merged_data.iloc[i]

        record_discrepancies = {}
        has_discrepancy = False

        # Compare each field
        for field in fields_to_compare:
            llm_val = row.get(f"{field}_llm")
            rule_based_val = row.get(f"{field}_rule_based")

            if not values_match(llm_val, rule_based_val):
                record_discrepancies[f"{field}_discrepancy"] = True
                record_discrepancies[f"llm_{field}"] = llm_val
                record_discrepancies[f"rule_based_{field}"] = rule_based_val
                has_discrepancy = True
            else:
                record_discrepancies[f"{field}_discrepancy"] = False
                record_discrepancies[f"llm_{field}"] = llm_val
                record_discrepancies[f"rule_based_{field}"] = rule_based_val

        if has_discrepancy:
            # Add record metadata
            record_discrepancies["comparison_key"] = row["comparison_key"]
            record_discrepancies["Country"] = row.get(
                "Country_llm", row.get("Country_rule_based")
            )
            record_discrepancies["Event"] = row.get(
                "Event_llm", row.get("Event_rule_based")
            )
            discrepant_records.append(record_discrepancies)

    discrepancies_df = pd.DataFrame(discrepant_records)

    return discrepancies_df, llm_common, llm_only_df, rule_based_only_df


if __name__ == "__main__":
    main()
