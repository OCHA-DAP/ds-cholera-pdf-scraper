#!/usr/bin/env python3
"""
Backfill Accuracy Metrics CLI Tool

Automatically discovers all prompt-versioned extraction files and calculates
accuracy metrics for them, updating the corresponding prompt logs.

Usage:
  python backfill_accuracy_metrics.py                    # Process all found versions
  python backfill_accuracy_metrics.py --version v1.1.0   # Process specific version
  python backfill_accuracy_metrics.py --dry-run          # Show what would be processed
  python backfill_accuracy_metrics.py --force            # Overwrite existing metrics
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Add src directory to path
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.append(str(src_dir))

# Import the analysis function from the main package
from compare import perform_discrepancy_analysis


def discover_prompt_versioned_files(outputs_dir: str) -> List[Dict[str, str]]:
    """
    Discover all prompt-versioned extraction files in outputs directory.

    Returns:
        List of dicts with 'version', 'file_path', and 'file_name' keys
    """
    outputs_path = Path(outputs_dir)
    if not outputs_path.exists():
        print(f"❌ Outputs directory not found: {outputs_dir}")
        return []

    # Pattern to match: text_extracted_data_prompt_v1.1.0.csv
    pattern = re.compile(r"text_extracted_data_prompt_(v\d+\.\d+\.\d+)\.csv")

    found_files = []

    for file_path in outputs_path.glob("*.csv"):
        match = pattern.match(file_path.name)
        if match:
            version = match.group(1)
            found_files.append(
                {
                    "version": version,
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                }
            )

    # Sort by version (semantic versioning)
    def version_key(item):
        version = item["version"].lstrip("v")
        return tuple(map(int, version.split(".")))

    found_files.sort(key=version_key)
    return found_files


def load_baseline_data(data_dir: str) -> pd.DataFrame:
    """Load and filter baseline data for comparison."""
    baseline_path = Path(data_dir) / "final_data_for_powerbi_with_kpi.csv"

    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline data not found: {baseline_path}")

    baseline_df = pd.read_csv(baseline_path)

    # Filter to Week 28, 2025 (consistent with QMD analysis)
    baseline_week28 = baseline_df[
        (baseline_df["Year"] == 2025) & (baseline_df["WeekNumber"] == 28)
    ].copy()

    return baseline_week28


def check_existing_accuracy_metrics(logger, prompt_version: str) -> bool:
    """Check if accuracy metrics already exist for a prompt version."""
    latest_log = logger.get_latest_log_for_prompt_version(prompt_version)

    if not latest_log:
        return False

    custom_metrics = latest_log.get("custom_metrics")
    if custom_metrics:
        try:
            metrics_data = (
                json.loads(custom_metrics)
                if isinstance(custom_metrics, str)
                else custom_metrics
            )
            return "accuracy_metrics" in metrics_data
        except:
            return False

    return False


def process_prompt_version(
    version: str,
    file_path: str,
    baseline_data: pd.DataFrame,
    force: bool = False,
    dry_run: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Process a single prompt version and calculate accuracy metrics.

    Returns:
        Dict with accuracy metrics if successful, None if failed
    """
    print(f"\n🎯 Processing prompt version: {version}")
    print(f"   📁 File: {Path(file_path).name}")

    try:
        # Load LLM extraction results
        llm_data = pd.read_csv(file_path)
        print(f"   📊 Loaded {len(llm_data)} LLM records")

        if dry_run:
            print("   🔍 [DRY RUN] Would calculate accuracy metrics")
            return {"dry_run": True, "version": version}

        # Check if accuracy metrics already exist
        from prompt_logger import PromptLogger

        logger = PromptLogger()

        if not force and check_existing_accuracy_metrics(logger, version):
            print("   ⚠️ Accuracy metrics already exist. Use --force to overwrite.")
            return None

        # Perform discrepancy analysis
        print("   🔍 Performing discrepancy analysis...")
        discrepancies_df, llm_common, llm_only_df, baseline_only_df = (
            perform_discrepancy_analysis(llm_data, baseline_data)
        )

        print(f"   📈 Analysis results:")
        print(f"      Common records: {len(llm_common)}")
        print(f"      Discrepant records: {len(discrepancies_df)}")
        print(
            f"      LLM-only: {len(llm_only_df)}, Baseline-only: {len(baseline_only_df)}"
        )

        # Calculate accuracy metrics
        from accuracy_metrics import AccuracyMetricsCalculator

        calculator = AccuracyMetricsCalculator()
        accuracy_metrics = calculator.calculate_metrics_from_qmd_variables(
            discrepancies_df=discrepancies_df,
            llm_common=llm_common,
            llm_only_df=llm_only_df,
            baseline_only_df=baseline_only_df,
            prompt_version=version,
        )

        print(
            f"   📊 Overall Accuracy: {accuracy_metrics['overall_accuracy_percent']}%"
        )
        print(f"   📊 Coverage Rate: {accuracy_metrics['coverage_rate_percent']}%")

        # Log to database
        latest_log = logger.get_latest_log_for_prompt_version(version)

        if latest_log:
            log_id = latest_log["id"]
            print(f"   💾 Updating log entry (ID: {log_id})")

            update_success = logger.update_log_with_accuracy_metrics(
                log_identifier=str(log_id), accuracy_metrics=accuracy_metrics
            )

            if update_success:
                print("   ✅ Accuracy metrics logged to database")
            else:
                print("   ❌ Failed to update database")
        else:
            print("   ⚠️ No log entry found - metrics calculated but not logged")

        return accuracy_metrics

    except Exception as e:
        print(f"   ❌ Error processing {version}: {e}")
        return None


def main():
    """Main function for backfill accuracy metrics CLI."""
    parser = argparse.ArgumentParser(
        description="Backfill accuracy metrics for prompt-versioned extraction files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--version",
        "-v",
        type=str,
        help="Process specific prompt version only (e.g., v1.1.0)",
    )
    parser.add_argument(
        "--outputs",
        "-o",
        type=str,
        default="outputs",
        help="Directory containing extraction files (default: outputs)",
    )
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default="data",
        help="Directory containing baseline data (default: data)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without making changes",
    )
    parser.add_argument(
        "--force", "-f", action="store_true", help="Overwrite existing accuracy metrics"
    )

    args = parser.parse_args()

    print("🚀 Backfill Accuracy Metrics Tool")
    print("=" * 50)

    try:
        # Discover prompt-versioned files
        print(f"📁 Scanning {args.outputs} for prompt-versioned files...")
        found_files = discover_prompt_versioned_files(args.outputs)

        if not found_files:
            print("❌ No prompt-versioned extraction files found")
            print("   Expected pattern: text_extracted_data_prompt_v*.*.*.csv")
            sys.exit(1)

        print(f"✅ Found {len(found_files)} prompt-versioned files:")
        for file_info in found_files:
            print(f"   📄 {file_info['version']}: {file_info['file_name']}")

        # Filter to specific version if requested
        if args.version:
            found_files = [f for f in found_files if f["version"] == args.version]
            if not found_files:
                print(f"❌ No files found for version {args.version}")
                sys.exit(1)
            print(f"\n🎯 Filtering to version: {args.version}")

        # Load baseline data
        print(f"\n📊 Loading baseline data from {args.data}...")
        baseline_data = load_baseline_data(args.data)
        print(f"✅ Baseline data loaded: {len(baseline_data)} records (Week 28, 2025)")

        if args.dry_run:
            print("\n🔍 DRY RUN MODE - No changes will be made")

        # Process each file
        results = []
        for file_info in found_files:
            result = process_prompt_version(
                version=file_info["version"],
                file_path=file_info["file_path"],
                baseline_data=baseline_data,
                force=args.force,
                dry_run=args.dry_run,
            )

            if result:
                results.append({"version": file_info["version"], "metrics": result})

        # Summary
        print("\n" + "=" * 50)
        print("📋 SUMMARY")
        print("=" * 50)

        if args.dry_run:
            print(f"🔍 Would process {len(results)} versions")
        else:
            successful = len(
                [r for r in results if r["metrics"] and not r["metrics"].get("dry_run")]
            )
            print(f"✅ Successfully processed {successful}/{len(found_files)} versions")

            if successful > 0:
                print("\n📊 Accuracy Summary:")
                for result in results:
                    if result["metrics"] and not result["metrics"].get("dry_run"):
                        version = result["version"]
                        accuracy = result["metrics"]["overall_accuracy_percent"]
                        coverage = result["metrics"]["coverage_rate_percent"]
                        print(
                            f"   {version}: {accuracy}% accuracy, {coverage}% coverage"
                        )

        print("\n🎯 Backfill complete!")

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
