"""
Accuracy Metrics Calculator for LLM extraction validation.
Calculates detailed accuracy metrics from discrepancy analysis for logging.
"""

from typing import Any, Dict, List, Optional

import pandas as pd


class AccuracyMetricsCalculator:
    """
    Calculate accuracy metrics from discrepancy analysis results.
    Designed to work with the output of discrepancy analysis QMD.
    """

    def __init__(self):
        self.metrics_fields = ["TotalCases", "CasesConfirmed", "Deaths", "CFR", "Grade"]

    def calculate_accuracy_metrics(
        self,
        discrepancies_df: pd.DataFrame,
        total_compared_records: int,
        llm_only_count: int = 0,
        baseline_only_count: int = 0,
        prompt_version: str = None,
        model_name: str = None,
        model_source: str = None,  # "tagged" or "legacy"
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive accuracy metrics from discrepancy analysis.

        Args:
            discrepancies_df: DataFrame containing discrepancy analysis results
            total_compared_records: Total number of records that were compared
            llm_only_count: Number of records only found in LLM output
            baseline_only_count: Number of records only found in baseline
            prompt_version: Version of prompt used for this extraction
            model_name: Name of the model used (e.g., "openai_gpt_4o")
            model_source: Whether this is from "tagged" or "legacy" extraction

        Returns:
            Dict containing detailed accuracy metrics
        """
        if total_compared_records == 0:
            return self._empty_metrics(prompt_version, model_name, model_source)

        metrics = {
            "prompt_version": prompt_version,
            "model_name": model_name,
            "model_source": model_source,  # "tagged" or "legacy"
            "total_compared_records": total_compared_records,
            "total_discrepant_records": len(discrepancies_df),
            "llm_only_records": llm_only_count,
            "baseline_only_records": baseline_only_count,
            "overall_accuracy_percent": round(
                (
                    (total_compared_records - len(discrepancies_df))
                    / total_compared_records
                )
                * 100,
                2,
            ),
            "overall_discrepancy_rate_percent": round(
                (len(discrepancies_df) / total_compared_records) * 100, 2
            ),
        }

        # Calculate field-level accuracy metrics
        field_metrics = {}
        if len(discrepancies_df) > 0:
            for field in self.metrics_fields:
                discrepancy_col = f"{field}_discrepancy"
                if discrepancy_col in discrepancies_df.columns:
                    field_discrepancies = discrepancies_df[discrepancy_col].sum()
                    field_accuracy = (
                        (total_compared_records - field_discrepancies)
                        / total_compared_records
                    ) * 100

                    field_metrics[f"{field}_accuracy_percent"] = round(
                        field_accuracy, 2
                    )
                    field_metrics[f"{field}_discrepancy_count"] = int(
                        field_discrepancies
                    )
                    field_metrics[f"{field}_discrepancy_rate_percent"] = round(
                        (field_discrepancies / total_compared_records) * 100, 2
                    )

        metrics["field_accuracy_metrics"] = field_metrics

        # Calculate coverage metrics
        total_baseline_records = total_compared_records + baseline_only_count
        if total_baseline_records > 0:
            metrics["coverage_rate_percent"] = round(
                (total_compared_records / total_baseline_records) * 100, 2
            )
        else:
            metrics["coverage_rate_percent"] = 0.0

        # Identify most problematic fields
        if field_metrics:
            problem_fields = []
            for field in self.metrics_fields:
                discrepancy_rate_key = f"{field}_discrepancy_rate_percent"
                if (
                    discrepancy_rate_key in field_metrics
                    and field_metrics[discrepancy_rate_key] > 10
                ):
                    problem_fields.append(
                        {
                            "field": field,
                            "discrepancy_rate": field_metrics[discrepancy_rate_key],
                        }
                    )

            # Sort by discrepancy rate
            problem_fields.sort(key=lambda x: x["discrepancy_rate"], reverse=True)
            metrics["problematic_fields"] = problem_fields

        return metrics

    def _empty_metrics(
        self,
        prompt_version: str = None,
        model_name: str = None,
        model_source: str = None,
    ) -> Dict[str, Any]:
        """Return empty metrics structure when no data available."""
        return {
            "prompt_version": prompt_version,
            "model_name": model_name,
            "model_source": model_source,
            "total_compared_records": 0,
            "total_discrepant_records": 0,
            "llm_only_records": 0,
            "baseline_only_records": 0,
            "overall_accuracy_percent": 0.0,
            "overall_discrepancy_rate_percent": 0.0,
            "coverage_rate_percent": 0.0,
            "field_accuracy_metrics": {},
            "problematic_fields": [],
        }

    def generate_accuracy_summary_text(self, metrics: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of accuracy metrics.

        Args:
            metrics: Accuracy metrics dictionary

        Returns:
            Formatted summary string
        """
        if metrics["total_compared_records"] == 0:
            return "No records available for accuracy analysis."

        summary_lines = [
            f"🎯 Accuracy Analysis Summary (Prompt {metrics['prompt_version']})",
            f"   Overall Accuracy: {metrics['overall_accuracy_percent']}%",
            f"   Records Compared: {metrics['total_compared_records']}",
            f"   Discrepant Records: {metrics['total_discrepant_records']}",
            f"   Coverage Rate: {metrics['coverage_rate_percent']}%",
            "",
        ]

        if metrics["field_accuracy_metrics"]:
            summary_lines.append("📊 Field-Level Accuracy:")
            for field in self.metrics_fields:
                accuracy_key = f"{field}_accuracy_percent"
                if accuracy_key in metrics["field_accuracy_metrics"]:
                    accuracy = metrics["field_accuracy_metrics"][accuracy_key]
                    summary_lines.append(f"   {field}: {accuracy}%")

        if metrics.get("problematic_fields"):
            summary_lines.extend(["", "🔴 Most Problematic Fields:"])
            for field_info in metrics["problematic_fields"][:3]:  # Top 3
                summary_lines.append(
                    f"   {field_info['field']}: {field_info['discrepancy_rate']}% error rate"
                )

        return "\n".join(summary_lines)
