"""
Accuracy metrics system for evaluating LLM extraction performance against baseline.
Integrates with prompt logging to provide automated performance feedback.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))
from post_processing import apply_post_processing_pipeline


class AccuracyEvaluator:
    """
    Evaluates LLM extraction accuracy against baseline data.
    """

    def __init__(self, baseline_csv_path: str = None):
        """
        Initialize with baseline data path.

        Args:
            baseline_csv_path: Path to baseline CSV file
        """
        if baseline_csv_path is None:
            # Default baseline path
            project_root = Path(__file__).parent.parent
            baseline_csv_path = (
                project_root / "data" / "final_data_for_powerbi_with_kpi.csv"
            )

        self.baseline_csv_path = Path(baseline_csv_path)
        self.baseline_data = None
        self._load_baseline()

    def _load_baseline(self):
        """Load and filter baseline data."""
        if not self.baseline_csv_path.exists():
            raise FileNotFoundError(f"Baseline CSV not found: {self.baseline_csv_path}")

        # Load full baseline
        full_baseline = pd.read_csv(self.baseline_csv_path)

        # Filter to Week 28, 2025 (our test case)
        self.baseline_data = full_baseline[
            (full_baseline["Year"] == 2025) & (full_baseline["WeekNumber"] == 28)
        ].copy()

        print(
            f"📊 Loaded baseline data: {len(self.baseline_data)} records (Week 28, 2025)"
        )

    def evaluate_extraction(
        self, llm_raw_df: pd.DataFrame, call_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Evaluate LLM extraction against baseline.

        Args:
            llm_raw_df: Raw LLM extraction DataFrame
            call_metadata: Metadata from the LLM call (optional)

        Returns:
            Dict with comprehensive accuracy metrics
        """
        print("🔍 Evaluating LLM extraction accuracy...")

        # Apply post-processing to both datasets
        llm_processed = apply_post_processing_pipeline(llm_raw_df.copy(), source="llm")
        baseline_processed = apply_post_processing_pipeline(
            self.baseline_data.copy(), source="baseline"
        )

        # Create comparison keys
        llm_processed["comparison_key"] = (
            llm_processed["Country"] + "_" + llm_processed["Event"]
        )
        baseline_processed["comparison_key"] = (
            baseline_processed["Country"] + "_" + baseline_processed["Event"]
        )

        # Find overlaps
        llm_keys = set(llm_processed["comparison_key"])
        baseline_keys = set(baseline_processed["comparison_key"])

        common_keys = llm_keys & baseline_keys
        llm_only_keys = llm_keys - baseline_keys
        baseline_only_keys = baseline_keys - llm_keys

        # Coverage metrics
        coverage_rate = len(common_keys) / len(baseline_keys) if baseline_keys else 0

        # Precision metrics (how many LLM records are valid)
        precision_rate = len(common_keys) / len(llm_keys) if llm_keys else 0

        # Field-level accuracy analysis
        field_accuracy = self._analyze_field_accuracy(
            llm_processed[llm_processed["comparison_key"].isin(common_keys)],
            baseline_processed[baseline_processed["comparison_key"].isin(common_keys)],
        )

        # Overall accuracy (records with all fields matching)
        overall_accuracy = (
            field_accuracy["records_with_all_fields_correct"] / len(common_keys)
            if common_keys
            else 0
        )

        # Compile comprehensive metrics
        metrics = {
            # Basic counts
            "baseline_total_records": len(baseline_processed),
            "llm_total_records": len(llm_processed),
            "common_records": len(common_keys),
            "llm_only_records": len(llm_only_keys),
            "baseline_only_records": len(baseline_only_keys),
            # Performance rates
            "coverage_rate": round(coverage_rate * 100, 2),
            "precision_rate": round(precision_rate * 100, 2),
            "overall_accuracy": round(overall_accuracy * 100, 2),
            # Field-specific accuracy
            "field_accuracy": field_accuracy,
            # Call metadata (if provided)
            "call_metadata": call_metadata or {},
            # Summary score (composite metric)
            "composite_score": round(
                (coverage_rate * 0.3 + precision_rate * 0.3 + overall_accuracy * 0.4)
                * 100,
                2,
            ),
        }

        # Print summary
        print(f"📊 Accuracy Evaluation Results:")
        print(
            f"   Coverage Rate: {metrics['coverage_rate']}% ({len(common_keys)}/{len(baseline_keys)})"
        )
        print(
            f"   Precision Rate: {metrics['precision_rate']}% ({len(common_keys)}/{len(llm_keys)})"
        )
        print(f"   Overall Accuracy: {metrics['overall_accuracy']}%")
        print(f"   Composite Score: {metrics['composite_score']}%")

        return metrics

    def _analyze_field_accuracy(
        self, llm_common: pd.DataFrame, baseline_common: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze field-level accuracy between LLM and baseline data.

        Returns:
            Dict with field-level accuracy metrics
        """
        # Merge on comparison_key to ensure proper alignment
        merged = pd.merge(
            llm_common,
            baseline_common,
            on="comparison_key",
            suffixes=("_llm", "_baseline"),
            how="inner",
        )

        fields_to_compare = ["TotalCases", "CasesConfirmed", "Deaths", "CFR", "Grade"]
        field_accuracy = {}

        total_comparisons = len(merged)
        records_with_all_correct = 0

        for i in range(total_comparisons):
            all_fields_correct = True

            for field in fields_to_compare:
                if field not in field_accuracy:
                    field_accuracy[field] = {"correct": 0, "total": 0, "accuracy": 0}

                llm_val = merged.iloc[i].get(f"{field}_llm")
                baseline_val = merged.iloc[i].get(f"{field}_baseline")

                field_accuracy[field]["total"] += 1

                if self._values_match(llm_val, baseline_val):
                    field_accuracy[field]["correct"] += 1
                else:
                    all_fields_correct = False

            if all_fields_correct:
                records_with_all_correct += 1

        # Calculate accuracy percentages
        for field in field_accuracy:
            correct = field_accuracy[field]["correct"]
            total = field_accuracy[field]["total"]
            field_accuracy[field]["accuracy"] = (
                round((correct / total * 100), 2) if total > 0 else 0
            )

        field_accuracy["records_with_all_fields_correct"] = records_with_all_correct
        field_accuracy["total_records_compared"] = total_comparisons

        return field_accuracy

    def _values_match(self, val1, val2, tolerance=0.01):
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

    def save_evaluation_results(
        self,
        metrics: Dict[str, Any],
        output_path: str,
        llm_data: pd.DataFrame = None,
        save_discrepancies: bool = True,
    ):
        """
        Save evaluation results to files.

        Args:
            metrics: Evaluation metrics dictionary
            output_path: Base path for saving results
            llm_data: Original LLM data (for discrepancy analysis)
            save_discrepancies: Whether to save detailed discrepancy files
        """
        output_path = Path(output_path)
        output_dir = output_path.parent
        output_dir.mkdir(exist_ok=True)

        # Save metrics as JSON
        import json

        metrics_path = output_dir / f"{output_path.stem}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        print(f"💾 Saved metrics to: {metrics_path}")

        if save_discrepancies and llm_data is not None:
            # Generate and save discrepancy analysis
            discrepancy_data = self._generate_discrepancy_data(llm_data)

            if discrepancy_data["discrepancies"] is not None:
                discrepancies_path = (
                    output_dir / f"{output_path.stem}_discrepancies.csv"
                )
                discrepancy_data["discrepancies"].to_csv(
                    discrepancies_path, index=False
                )
                print(f"💾 Saved discrepancies to: {discrepancies_path}")

            if discrepancy_data["llm_only"] is not None:
                llm_only_path = output_dir / f"{output_path.stem}_llm_only.csv"
                discrepancy_data["llm_only"].to_csv(llm_only_path, index=False)
                print(f"💾 Saved LLM-only records to: {llm_only_path}")

    def _generate_discrepancy_data(
        self, llm_raw_df: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Generate detailed discrepancy analysis data."""
        # Apply post-processing
        llm_processed = apply_post_processing_pipeline(llm_raw_df.copy(), source="llm")
        baseline_processed = apply_post_processing_pipeline(
            self.baseline_data.copy(), source="baseline"
        )

        # Create comparison keys
        llm_processed["comparison_key"] = (
            llm_processed["Country"] + "_" + llm_processed["Event"]
        )
        baseline_processed["comparison_key"] = (
            baseline_processed["Country"] + "_" + baseline_processed["Event"]
        )

        # Find common and unique records
        llm_keys = set(llm_processed["comparison_key"])
        baseline_keys = set(baseline_processed["comparison_key"])
        common_keys = llm_keys & baseline_keys
        llm_only_keys = llm_keys - baseline_keys

        # Generate discrepancy records (similar to existing logic)
        fields_to_compare = ["TotalCases", "CasesConfirmed", "Deaths", "CFR", "Grade"]
        discrepant_records = []

        llm_common = (
            llm_processed[llm_processed["comparison_key"].isin(common_keys)]
            .sort_values("comparison_key")
            .reset_index(drop=True)
        )
        baseline_common = (
            baseline_processed[baseline_processed["comparison_key"].isin(common_keys)]
            .sort_values("comparison_key")
            .reset_index(drop=True)
        )

        for i in range(len(llm_common)):
            llm_row = llm_common.iloc[i]
            baseline_row = baseline_common.iloc[i]

            record_discrepancies = {}
            has_discrepancy = False

            for field in fields_to_compare:
                llm_val = llm_row.get(field)
                baseline_val = baseline_row.get(field)

                if not self._values_match(llm_val, baseline_val):
                    record_discrepancies[f"{field}_discrepancy"] = True
                    record_discrepancies[f"llm_{field}"] = llm_val
                    record_discrepancies[f"baseline_{field}"] = baseline_val
                    has_discrepancy = True
                else:
                    record_discrepancies[f"{field}_discrepancy"] = False
                    record_discrepancies[f"llm_{field}"] = llm_val
                    record_discrepancies[f"baseline_{field}"] = baseline_val

            if has_discrepancy:
                record_discrepancies["comparison_key"] = llm_row["comparison_key"]
                record_discrepancies["Country"] = llm_row["Country"]
                record_discrepancies["Event"] = llm_row["Event"]
                discrepant_records.append(record_discrepancies)

        return {
            "discrepancies": (
                pd.DataFrame(discrepant_records) if discrepant_records else None
            ),
            "llm_only": (
                llm_processed[llm_processed["comparison_key"].isin(llm_only_keys)]
                if llm_only_keys
                else None
            ),
        }


# Integration function for use with prompt logging
def evaluate_and_log_accuracy(
    llm_raw_df: pd.DataFrame,
    prompt_call_id: str,
    prompt_metadata: Dict[str, Any],
    output_base_path: str = None,
) -> Dict[str, Any]:
    """
    Convenience function to evaluate accuracy and integrate with prompt logging.

    Args:
        llm_raw_df: Raw LLM extraction DataFrame
        prompt_call_id: ID from prompt logger
        prompt_metadata: Metadata from prompt manager
        output_base_path: Base path for saving evaluation results

    Returns:
        Comprehensive metrics dictionary
    """
    evaluator = AccuracyEvaluator()

    # Add call ID to metadata
    call_metadata = {"prompt_call_id": prompt_call_id, **prompt_metadata}

    # Evaluate accuracy
    metrics = evaluator.evaluate_extraction(llm_raw_df, call_metadata)

    # Save results if path provided
    if output_base_path:
        evaluator.save_evaluation_results(
            metrics, output_base_path, llm_data=llm_raw_df, save_discrepancies=True
        )

    return metrics


if __name__ == "__main__":
    # Test the accuracy evaluator
    print("🧪 Testing AccuracyEvaluator...")

    # Load a sample LLM extraction for testing
    sample_llm_path = (
        Path(__file__).parent.parent / "outputs" / "text_extracted_data2.csv"
    )

    if sample_llm_path.exists():
        sample_llm_df = pd.read_csv(sample_llm_path)

        evaluator = AccuracyEvaluator()
        metrics = evaluator.evaluate_extraction(sample_llm_df)

        print("\n📊 Sample Evaluation Results:")
        print(f"Coverage: {metrics['coverage_rate']}%")
        print(f"Precision: {metrics['precision_rate']}%")
        print(f"Overall Accuracy: {metrics['overall_accuracy']}%")
        print(f"Composite Score: {metrics['composite_score']}%")

        # Save test results
        test_output = Path(__file__).parent.parent / "outputs" / "test_evaluation"
        evaluator.save_evaluation_results(metrics, test_output, llm_data=sample_llm_df)

        print("✅ AccuracyEvaluator test complete!")
    else:
        print(f"❌ Sample LLM data not found: {sample_llm_path}")
