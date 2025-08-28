#!/usr/bin/env python3
"""
Prompt comparison utilities for analyzing LLM extraction performance across versions.
Provides easy access to discrepancies, accuracy metrics, and multi-version comparison tools.
Enhanced to support model-tagged files from OpenRouter multi-model testing.

Key Functions:
- get_discrepancies_by_prompt_version(): Get discrepancies for a single prompt version
- show_model_comparison(): Compare models for a specific prompt version
- prompt_model_comparison(): NEW! Compare all prompt/model combinations across versions
- get_model_discrepancies(): Get discrepancies for specific prompt+model combination
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..compare import perform_discrepancy_analysis


def get_discrepancies_by_prompt_version(
    prompt_version: str, outputs_dir: str = "outputs", data_dir: str = "data"
) -> Optional[pd.DataFrame]:
    """
    Get discrepancies DataFrame for a specific prompt version.

    Args:
        prompt_version: Prompt version (e.g., "v1.1.1")
        outputs_dir: Directory containing extraction results
        data_dir: Directory containing baseline data

    Returns:
        DataFrame with discrepancies, or None if files not found
    """
    # Look for extraction files with new naming convention
    outputs_path = Path(outputs_dir)
    extraction_files = list(
        outputs_path.glob(f"extraction_*_prompt_{prompt_version}_*.csv")
    )

    if not extraction_files:
        print(f"❌ No extraction files found for prompt {prompt_version}")
        return None

    # Use the first matching file (or most recent if multiple)
    extraction_path = extraction_files[0]
    if len(extraction_files) > 1:
        # Sort by modification time, use most recent
        extraction_path = max(extraction_files, key=lambda p: p.stat().st_mtime)
        print(f"📁 Multiple files found, using most recent: {extraction_path.name}")

    # Check if baseline data exists
    baseline_path = Path(data_dir) / "final_data_for_powerbi_with_kpi.csv"
    if not baseline_path.exists():
        print(f"❌ Baseline data not found: {baseline_path}")
        return None

    print(f"🔍 Loading discrepancies for prompt {prompt_version}...")

    try:
        # Load data
        llm_data = pd.read_csv(extraction_path)
        baseline_df = pd.read_csv(baseline_path)

        # Filter baseline to Week 28, 2025 (same as QMD)
        baseline_week28 = baseline_df[
            (baseline_df["Year"] == 2025) & (baseline_df["WeekNumber"] == 28)
        ].copy()

        print(f"📊 LLM data: {len(llm_data)} records")
        print(f"📊 Baseline data: {len(baseline_week28)} records")

        # Perform discrepancy analysis
        discrepancies_df, llm_common, llm_only_df, baseline_only_df = (
            perform_discrepancy_analysis(llm_data, baseline_week28)
        )

        print(f"✅ Found {len(discrepancies_df)} discrepant records")
        print(f"   📊 Records compared: {len(llm_common)}")
        print(f"   📊 LLM-only records: {len(llm_only_df)}")
        print(f"   📊 Baseline-only records: {len(baseline_only_df)}")

        return discrepancies_df

    except Exception as e:
        print(f"❌ Error analyzing prompt {prompt_version}: {e}")
        return None


def get_analysis_summary_by_prompt_version(
    prompt_version: str, outputs_dir: str = "outputs", data_dir: str = "data"
) -> Optional[Dict[str, Any]]:
    """
    Get complete analysis summary for a prompt version.

    Returns:
        Dictionary with discrepancies_df, llm_common, llm_only_df, baseline_only_df
    """
    # Look for extraction files with new naming convention first
    outputs_path = Path(outputs_dir)
    extraction_files = list(
        outputs_path.glob(f"extraction_*_prompt_{prompt_version}_*.csv")
    )

    if not extraction_files:
        # Fallback to old naming convention
        extraction_files = list(
            outputs_path.glob(f"text_extracted_data_prompt_{prompt_version}*.csv")
        )

    if not extraction_files:
        print(f"❌ No extraction files found for prompt {prompt_version}")
        return None

    # Use the first matching file (or most recent if multiple)
    extraction_path = extraction_files[0]
    if len(extraction_files) > 1:
        extraction_path = max(extraction_files, key=lambda p: p.stat().st_mtime)

    try:
        # Load data
        llm_data = pd.read_csv(extraction_path)
        baseline_df = pd.read_csv(
            Path(data_dir) / "final_data_for_powerbi_with_kpi.csv"
        )

        # Filter baseline to Week 28, 2025
        baseline_week28 = baseline_df[
            (baseline_df["Year"] == 2025) & (baseline_df["WeekNumber"] == 28)
        ].copy()

        # Perform discrepancy analysis
        discrepancies_df, llm_common, llm_only_df, baseline_only_df = (
            perform_discrepancy_analysis(llm_data, baseline_week28)
        )

        return {
            "discrepancies_df": discrepancies_df,
            "llm_common": llm_common,
            "llm_only_df": llm_only_df,
            "baseline_only_df": baseline_only_df,
            "prompt_version": prompt_version,
            "extraction_file": str(extraction_path),
        }

    except Exception as e:
        print(f"❌ Error analyzing prompt {prompt_version}: {e}")
        return None


def list_available_prompt_versions(outputs_dir: str = "outputs") -> list:
    """
    List all available prompt versions based on extraction files.

    Returns:
        List of prompt versions found
    """
    outputs_path = Path(outputs_dir)
    if not outputs_path.exists():
        print(f"❌ Outputs directory not found: {outputs_path}")
        return []

    # Find all prompt-versioned files
    pattern = "text_extracted_data_prompt_*.csv"
    files = list(outputs_path.glob(pattern))

    versions = []
    for file in files:
        # Extract version from filename
        name = file.stem  # Remove .csv
        if name.startswith("text_extracted_data_prompt_"):
            version = name.replace("text_extracted_data_prompt_", "")
            versions.append(version)

    versions.sort()
    print(f"📁 Found {len(versions)} prompt versions: {versions}")
    return versions


def get_discrepancies_by_model(
    prompt_version: str,
    model_name: str,
    outputs_dir: str = "outputs",
    data_dir: str = "data",
) -> Optional[pd.DataFrame]:
    """
    Get discrepancies DataFrame for a specific prompt version and model.

    Args:
        prompt_version: Prompt version (e.g., "v1.1.2")
        model_name: Model name (e.g., "anthropic_claude_sonnet_4")
        outputs_dir: Directory containing extraction results
        data_dir: Directory containing baseline data

    Returns:
        DataFrame with discrepancies, or None if files not found
    """
    # Look for new naming pattern: extraction_*_prompt_{version}_model_{model}.csv
    outputs_path = Path(outputs_dir)
    extraction_files = list(
        outputs_path.glob(
            f"extraction_*_prompt_{prompt_version}_model_{model_name}.csv"
        )
    )

    if not extraction_files:
        print(f"❌ Model extraction file not found for {model_name}")
        return None

    # Use the most recent file if multiple exist
    extraction_path = max(extraction_files, key=lambda p: p.stat().st_mtime)

    # Check if baseline data exists
    baseline_path = Path(data_dir) / "final_data_for_powerbi_with_kpi.csv"
    if not baseline_path.exists():
        print(f"❌ Baseline data not found: {baseline_path}")
        return None

    print(
        f"🔍 Loading discrepancies for prompt {prompt_version} with model {model_name}..."
    )

    try:
        # Load LLM extraction results
        llm_data = pd.read_csv(extraction_path)
        baseline_df = pd.read_csv(baseline_path)

        # Filter baseline to Week 28, 2025 (consistent with QMD analysis)
        baseline_week28 = baseline_df[
            (baseline_df["Year"] == 2025) & (baseline_df["WeekNumber"] == 28)
        ].copy()

        print(f"📊 LLM data: {len(llm_data)} records")
        print(f"📊 Baseline data: {len(baseline_week28)} records")

        # Perform discrepancy analysis
        discrepancies_df, llm_common, llm_only_df, baseline_only_df = (
            perform_discrepancy_analysis(llm_data, baseline_week28)
        )

        print(f"✅ Found {len(discrepancies_df)} discrepant records")
        print(f"   📊 Records compared: {len(llm_common)}")
        print(f"   📊 LLM-only records: {len(llm_only_df)}")
        print(f"   📊 Baseline-only records: {len(baseline_only_df)}")

        return discrepancies_df

    except Exception as e:
        print(
            f"❌ Error analyzing prompt {prompt_version} with model {model_name}: {e}"
        )
        return None


def get_analysis_summary_by_model(
    prompt_version: str,
    model_name: str,
    outputs_dir: str = "outputs",
    data_dir: str = "data",
) -> Optional[Dict[str, Any]]:
    """
    Get complete analysis summary for a prompt version and model.

    Returns:
        Dictionary with discrepancies_df, llm_common, llm_only_df, baseline_only_df, model info
    """
    # Look for new naming pattern: extraction_*_prompt_{version}_model_{model}.csv
    outputs_path = Path(outputs_dir)
    extraction_files = list(
        outputs_path.glob(
            f"extraction_*_prompt_{prompt_version}_model_{model_name}.csv"
        )
    )

    if not extraction_files:
        print(f"❌ Model extraction file not found for {model_name}")
        return None

    # Use the most recent file if multiple exist
    actual_file = max(extraction_files, key=lambda p: p.stat().st_mtime)
    file_type = "new"

    try:
        # Load data
        llm_data = pd.read_csv(actual_file)
        baseline_df = pd.read_csv(
            Path(data_dir) / "final_data_for_powerbi_with_kpi.csv"
        )

        # Filter baseline to Week 28, 2025
        baseline_week28 = baseline_df[
            (baseline_df["Year"] == 2025) & (baseline_df["WeekNumber"] == 28)
        ].copy()

        # Perform discrepancy analysis
        discrepancies_df, llm_common, llm_only_df, baseline_only_df = (
            perform_discrepancy_analysis(llm_data, baseline_week28)
        )

        return {
            "discrepancies_df": discrepancies_df,
            "llm_common": llm_common,
            "llm_only_df": llm_only_df,
            "baseline_only_df": baseline_only_df,
            "prompt_version": prompt_version,
            "model_name": model_name,
            "extraction_file": str(actual_file),
            "file_type": file_type,  # "tagged" or "legacy"
        }

    except Exception as e:
        print(
            f"❌ Error analyzing prompt {prompt_version} with model {model_name}: {e}"
        )
        return None


def list_available_model_extractions(
    outputs_dir: str = "outputs",
) -> List[Dict[str, str]]:
    """
    List all available model-tagged extraction files AND legacy untagged files.
    Legacy files are assumed to be GPT-4o extractions.

    Returns:
        List of dicts with 'prompt_version', 'model_name', 'file_name' keys
    """
    outputs_path = Path(outputs_dir)
    if not outputs_path.exists():
        print(f"❌ Outputs directory not found: {outputs_path}")
        return []

    # Pattern to match new naming: extraction_38_prompt_v1.1.2_model_meta_llama_llama_4_maverick.csv
    new_pattern = re.compile(
        r"extraction_(\d+)_prompt_(v\d+\.\d+\.\d+)_model_(.+)\.csv"
    )

    model_files = []

    for file_path in outputs_path.glob("*.csv"):
        # Only check for new naming pattern
        new_match = new_pattern.match(file_path.name)
        if new_match:
            call_id = new_match.group(1)
            prompt_version = new_match.group(2)
            model_name = new_match.group(3)

            model_files.append(
                {
                    "prompt_version": prompt_version,
                    "model_name": model_name,
                    "file_name": file_path.name,
                    "call_id": call_id,
                    "naming_style": "new",
                    "is_legacy": False,
                }
            )

    # Sort by prompt version then model name
    def sort_key(item):
        version = item["prompt_version"].lstrip("v")
        return (tuple(map(int, version.split("."))), item["model_name"])

    model_files.sort(key=sort_key)

    print(f"📁 Found {len(model_files)} extraction files:")
    for file_info in model_files:
        print(
            f"   📄 {file_info['prompt_version']} + {file_info['model_name']}: {file_info['file_name']}"
        )

    return model_files


def get_discrepancies_by_model_with_legacy_support(
    prompt_version: str,
    model_name: str,
    outputs_dir: str = "outputs",
    data_dir: str = "data",
) -> Optional[pd.DataFrame]:
    """
    Get discrepancies for a model using new naming format only.

    Args:
        prompt_version: Prompt version (e.g., "v1.1.2")
        model_name: Model name (e.g., "meta_llama_llama_4_maverick")
        outputs_dir: Directory containing extraction results
        data_dir: Directory containing baseline data

    Returns:
        DataFrame with discrepancies, or None if files not found
    """
    outputs_path = Path(outputs_dir)
    extraction_path = None

    # Look for new naming pattern: extraction_*_prompt_{version}_model_{model}.csv
    new_files = list(
        outputs_path.glob(
            f"extraction_*_prompt_{prompt_version}_model_{model_name}.csv"
        )
    )
    if new_files:
        extraction_path = max(new_files, key=lambda p: p.stat().st_mtime)
        print(f"🔍 Using file: {extraction_path.name}")

    if not extraction_path:
        print(
            f"❌ No extraction files found for prompt {prompt_version} + model {model_name}"
        )
        return None

    # Check if baseline data exists
    baseline_path = Path(data_dir) / "final_data_for_powerbi_with_kpi.csv"
    if not baseline_path.exists():
        print(f"❌ Baseline data not found: {baseline_path}")
        return None

    print(
        f"🔍 Loading discrepancies for prompt {prompt_version} + model {model_name}..."
    )

    try:
        # Load data
        llm_data = pd.read_csv(extraction_path)
        baseline_df = pd.read_csv(baseline_path)

        # Filter baseline to Week 28, 2025 (same as QMD)
        baseline_week28 = baseline_df[
            (baseline_df["Year"] == 2025) & (baseline_df["WeekNumber"] == 28)
        ].copy()

        print(f"📊 LLM data: {len(llm_data)} records")
        print(f"📊 Baseline data: {len(baseline_week28)} records")

        # Perform discrepancy analysis
        discrepancies_df, llm_common, llm_only_df, baseline_only_df = (
            perform_discrepancy_analysis(llm_data, baseline_week28)
        )

        print(f"✅ Found {len(discrepancies_df)} discrepant records")
        print(f"   📊 Records compared: {len(llm_common)}")
        print(f"   📊 LLM-only records: {len(llm_only_df)}")
        print(f"   📊 Baseline-only records: {len(baseline_only_df)}")

        return discrepancies_df

    except Exception as e:
        print(
            f"❌ Error analyzing prompt {prompt_version} with model {model_name}: {e}"
        )
        return None


def quick_model_discrepancy_check(prompt_version: str, model_name: str):
    """
    Quick check of discrepancies for a prompt version and model.
    Prints summary statistics.
    """
    discrepancies_df = get_discrepancies_by_model(prompt_version, model_name)

    if discrepancies_df is None:
        return

    if len(discrepancies_df) == 0:
        print(
            f"🎉 No discrepancies found for prompt {prompt_version} with model {model_name}!"
        )
        return

    print(f"\n📊 Discrepancy Summary for {prompt_version} + {model_name}:")
    print(f"   Total discrepant records: {len(discrepancies_df)}")

    # Count discrepancies by field
    fields = ["TotalCases", "CasesConfirmed", "Deaths", "CFR", "Grade"]
    for field in fields:
        discrepancy_col = f"{field}_discrepancy"
        if discrepancy_col in discrepancies_df.columns:
            count = discrepancies_df[discrepancy_col].sum()
            if count > 0:
                print(f"   {field}: {count} discrepancies")

    # Show most common countries with discrepancies
    if "Country" in discrepancies_df.columns:
        country_counts = discrepancies_df["Country"].value_counts().head(3)
        print(f"   Top countries with discrepancies: {dict(country_counts)}")


def compare_models_for_prompt(
    prompt_version: str, outputs_dir: str = "outputs"
) -> pd.DataFrame:
    """
    Compare all available models for a specific prompt version.

    Returns:
        DataFrame with model comparison metrics
    """
    # Find all model files for this prompt version
    available_models = list_available_model_extractions(outputs_dir)
    prompt_models = [
        m for m in available_models if m["prompt_version"] == prompt_version
    ]

    if not prompt_models:
        print(f"❌ No model extractions found for prompt {prompt_version}")
        return pd.DataFrame()

    print(f"🔍 Comparing {len(prompt_models)} models for prompt {prompt_version}...")

    comparison_results = []

    for model_info in prompt_models:
        model_name = model_info["model_name"]

        # Get analysis summary
        analysis = get_analysis_summary_by_model(
            prompt_version, model_name, outputs_dir
        )

        if analysis:
            discrepancies_df = analysis["discrepancies_df"]
            llm_common = analysis["llm_common"]

            # Calculate metrics
            total_compared = len(llm_common)
            total_discrepancies = len(discrepancies_df)
            accuracy_rate = (
                ((total_compared - total_discrepancies) / total_compared * 100)
                if total_compared > 0
                else 0
            )

            comparison_results.append(
                {
                    "model_name": model_name,
                    "total_compared": total_compared,
                    "total_discrepancies": total_discrepancies,
                    "accuracy_rate": round(accuracy_rate, 2),
                    "file_name": model_info["file_name"],
                }
            )

    comparison_df = pd.DataFrame(comparison_results)

    if not comparison_df.empty:
        # Sort by accuracy rate descending
        comparison_df = comparison_df.sort_values("accuracy_rate", ascending=False)

        print(f"\n📊 Model Comparison Results for prompt {prompt_version}:")
        print(comparison_df.to_string(index=False))

    return comparison_df


def quick_discrepancy_check(prompt_version: str):
    """
    Quick check of discrepancies for a prompt version.
    Prints summary statistics.
    """
    discrepancies_df = get_discrepancies_by_prompt_version(prompt_version)

    if discrepancies_df is None:
        return

    if len(discrepancies_df) == 0:
        print(f"🎉 No discrepancies found for prompt {prompt_version}!")
        return

    print(f"\n📊 Discrepancy Summary for {prompt_version}:")
    print(f"   Total discrepant records: {len(discrepancies_df)}")

    # Count discrepancies by field
    fields = ["TotalCases", "CasesConfirmed", "Deaths", "CFR", "Grade"]
    for field in fields:
        discrepancy_col = f"{field}_discrepancy"
        if discrepancy_col in discrepancies_df.columns:
            count = discrepancies_df[discrepancy_col].sum()
            if count > 0:
                print(f"   {field}: {count} discrepancies")

    # Show most common countries with discrepancies
    if "Country" in discrepancies_df.columns:
        country_counts = discrepancies_df["Country"].value_counts().head(3)
        print(f"   Top countries with discrepancies: {dict(country_counts)}")


# Convenience functions for easy interactive use
def get_model_discrepancies(
    prompt_version: str = "v1.1.2", model_name: str = None
) -> Optional[pd.DataFrame]:
    """
    Convenience function to quickly get discrepancies for model-tagged files.
    Supports legacy untagged files (assumed to be GPT-4o).

    Args:
        prompt_version: Prompt version (default: "v1.1.2")
        model_name: Model name (if None, lists available models)

    Returns:
        DataFrame with discrepancies, or None if not found

    Examples:
        # List available models (including legacy GPT-4o)
        get_model_discrepancies()

        # Get Claude 4 discrepancies
        df = get_model_discrepancies("v1.1.2", "anthropic_claude_sonnet_4")

        # Get GPT-4o discrepancies (works with both tagged and legacy files)
        df = get_model_discrepancies("v1.1.2", "openai_gpt_4o")
    """
    if model_name is None:
        print(f"📁 Available model extractions for prompt {prompt_version}:")
        available_models = list_available_model_extractions()
        prompt_models = [
            m for m in available_models if m["prompt_version"] == prompt_version
        ]

        if not prompt_models:
            print(f"❌ No model extractions found for prompt {prompt_version}")
            print("\n💡 Available prompt versions with models:")
            all_versions = list(set(m["prompt_version"] for m in available_models))
            for version in sorted(all_versions):
                version_models = [
                    m for m in available_models if m["prompt_version"] == version
                ]
                model_names = [m["model_name"] for m in version_models]
                print(f"   {version}: {model_names}")
        else:
            for model_info in prompt_models:
                legacy_note = " (legacy)" if model_info.get("is_legacy", False) else ""
                print(f"   📄 {model_info['model_name']}{legacy_note}")

            print(f"\n💡 Usage:")
            print(f"   df = get_model_discrepancies('{prompt_version}', 'model_name')")

        return None

    return get_discrepancies_by_model_with_legacy_support(prompt_version, model_name)


def show_model_comparison(prompt_version: str = "v1.1.2") -> pd.DataFrame:
    """
    Convenience function to show model comparison for a prompt version.

    Args:
        prompt_version: Prompt version (default: "v1.1.2")

    Returns:
        DataFrame with model comparison results
    """
    return compare_models_for_prompt(prompt_version)


def prompt_model_comparison(
    prompt_versions: Optional[List[str]] = None, outputs_dir: str = "outputs"
) -> pd.DataFrame:
    """
    Compare all prompt/model combinations across multiple prompt versions.

    This function extends show_model_comparison to work across multiple prompt versions,
    providing a comprehensive view of all combinations tested.

    Args:
        prompt_versions: List of prompt versions to compare (if None, auto-detect)
        outputs_dir: Directory containing extraction results

    Returns:
        DataFrame with prompt_version, model_name, and accuracy metrics for all combinations
    """
    # Auto-detect available prompt versions if not specified
    if prompt_versions is None:
        available_models = list_available_model_extractions(outputs_dir)
        prompt_versions = sorted(
            list(set([m["prompt_version"] for m in available_models]))
        )
        print(f"🔍 Auto-detected prompt versions: {prompt_versions}")

    if not prompt_versions:
        print(f"❌ No prompt versions found in {outputs_dir}")
        return pd.DataFrame()

    print(f"🚀 Comparing {len(prompt_versions)} prompt versions across all models...")

    all_combinations = []

    for prompt_version in prompt_versions:
        print(f"\n📝 Analyzing prompt {prompt_version}...")

        # Get model comparison for this prompt version
        prompt_comparison = compare_models_for_prompt(prompt_version, outputs_dir)

        if not prompt_comparison.empty:
            # Add prompt_version column
            prompt_comparison["prompt_version"] = prompt_version

            # Reorder columns to put prompt_version first
            cols = ["prompt_version"] + [
                col for col in prompt_comparison.columns if col != "prompt_version"
            ]
            prompt_comparison = prompt_comparison[cols]

            all_combinations.append(prompt_comparison)
            print(f"   ✅ Found {len(prompt_comparison)} model combinations")
        else:
            print(f"   ⚠️ No model results found for prompt {prompt_version}")

    if not all_combinations:
        print("❌ No valid prompt/model combinations found")
        return pd.DataFrame()

    # Combine all results
    combined_df = pd.concat(all_combinations, ignore_index=True)

    # Sort by prompt version, then by accuracy rate
    combined_df = combined_df.sort_values(
        ["prompt_version", "accuracy_rate"], ascending=[True, False]
    )

    print(f"\n🏆 COMPLETE PROMPT/MODEL COMPARISON ({len(combined_df)} combinations)")
    print("=" * 80)
    print(combined_df.to_string(index=False))

    # Add summary insights
    print(f"\n📊 SUMMARY INSIGHTS:")
    print("-" * 40)

    # Best overall combination
    best_combo = combined_df.loc[combined_df["accuracy_rate"].idxmax()]
    print(
        f"🥇 Best combination: {best_combo['prompt_version']} + {best_combo['model_name']} ({best_combo['accuracy_rate']}%)"
    )

    # Best prompt version average
    prompt_averages = (
        combined_df.groupby("prompt_version")["accuracy_rate"]
        .mean()
        .sort_values(ascending=False)
    )
    best_prompt = prompt_averages.index[0]
    print(
        f"🎯 Best prompt version: {best_prompt} (avg: {prompt_averages[best_prompt]:.1f}%)"
    )

    # Best model average
    model_averages = (
        combined_df.groupby("model_name")["accuracy_rate"]
        .mean()
        .sort_values(ascending=False)
    )
    best_model = model_averages.index[0]
    print(
        f"🤖 Best model overall: {best_model} (avg: {model_averages[best_model]:.1f}%)"
    )

    # Show prompt evolution
    if len(prompt_versions) > 1:
        print(f"\n📈 PROMPT EVOLUTION:")
        for prompt in sorted(prompt_versions):
            prompt_data = combined_df[combined_df["prompt_version"] == prompt]
            avg_accuracy = prompt_data["accuracy_rate"].mean()
            max_accuracy = prompt_data["accuracy_rate"].max()
            print(
                f"   {prompt}: avg={avg_accuracy:.1f}%, max={max_accuracy:.1f}% ({len(prompt_data)} models)"
            )

    return combined_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Get prompt comparison analysis results by version"
    )
    parser.add_argument("prompt_version", help="Prompt version (e.g., v1.1.1)")
    parser.add_argument("--outputs", default="outputs", help="Outputs directory")
    parser.add_argument("--data", default="data", help="Data directory")
    parser.add_argument("--summary", action="store_true", help="Show quick summary")

    args = parser.parse_args()

    if args.summary:
        quick_discrepancy_check(args.prompt_version)
    else:
        discrepancies_df = get_discrepancies_by_prompt_version(
            args.prompt_version, args.outputs, args.data
        )
        if discrepancies_df is not None:
            print(f"\nDiscrepancies DataFrame shape: {discrepancies_df.shape}")
            print(f"Columns: {list(discrepancies_df.columns)}")
