#!/usr/bin/env python3
"""
Prompt comparison utilities for analyzing LLM extraction performance across versions.
Provides easy access to discrepancies, accuracy metrics, and multi-version comparison tools.
Enhanced to support model-tagged files from OpenRouter multi-model testing.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.backfill_accuracy_metrics import perform_discrepancy_analysis


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
    # Check if extraction file exists
    extraction_file = f"text_extracted_data_prompt_{prompt_version}.csv"
    extraction_path = Path(outputs_dir) / extraction_file

    if not extraction_path.exists():
        print(f"❌ Extraction file not found: {extraction_path}")
        return None

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
    # Check if extraction file exists
    extraction_file = f"text_extracted_data_prompt_{prompt_version}.csv"
    extraction_path = Path(outputs_dir) / extraction_file

    if not extraction_path.exists():
        print(f"❌ Extraction file not found: {extraction_path}")
        return None

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
    # Check if model-tagged extraction file exists
    extraction_file = (
        f"text_extracted_data_prompt_{prompt_version}_model_{model_name}.csv"
    )
    extraction_path = Path(outputs_dir) / extraction_file

    if not extraction_path.exists():
        print(f"❌ Model extraction file not found: {extraction_path}")
        return None

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
    Supports legacy untagged files (assumed to be GPT-4o).

    Returns:
        Dictionary with discrepancies_df, llm_common, llm_only_df, baseline_only_df, model info
    """
    # Determine extraction file path with legacy support
    if model_name == "openai_gpt_4o":
        # For GPT-4o, try both tagged and legacy formats
        extraction_file = (
            f"text_extracted_data_prompt_{prompt_version}_model_{model_name}.csv"
        )
        legacy_file = f"text_extracted_data_prompt_{prompt_version}.csv"

        extraction_path = Path(outputs_dir) / extraction_file
        legacy_path = Path(outputs_dir) / legacy_file

        if extraction_path.exists():
            actual_file = extraction_path
            file_type = "tagged"
        elif legacy_path.exists():
            actual_file = legacy_path
            file_type = "legacy"
        else:
            print(f"❌ No GPT-4o extraction found for prompt {prompt_version}")
            return None
    else:
        # For other models, only look for tagged files
        extraction_file = (
            f"text_extracted_data_prompt_{prompt_version}_model_{model_name}.csv"
        )
        actual_file = Path(outputs_dir) / extraction_file
        file_type = "tagged"

        if not actual_file.exists():
            print(f"❌ Model extraction file not found: {actual_file}")
            return None

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

    # Pattern to match model-tagged files: text_extracted_data_prompt_v1.1.2_model_anthropic_claude_sonnet_4.csv
    model_pattern = re.compile(
        r"text_extracted_data_prompt_(v\d+\.\d+\.\d+)_model_(.+)\.csv"
    )

    # Pattern to match legacy untagged files: text_extracted_data_prompt_v1.1.2.csv
    legacy_pattern = re.compile(r"text_extracted_data_prompt_(v\d+\.\d+\.\d+)\.csv")

    model_files = []

    for file_path in outputs_path.glob("*.csv"):
        # Check for model-tagged files first
        model_match = model_pattern.match(file_path.name)
        if model_match:
            prompt_version = model_match.group(1)
            model_name = model_match.group(2)

            model_files.append(
                {
                    "prompt_version": prompt_version,
                    "model_name": model_name,
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "is_legacy": False,
                }
            )
        else:
            # Check for legacy untagged files
            legacy_match = legacy_pattern.match(file_path.name)
            if legacy_match:
                prompt_version = legacy_match.group(1)
                # Assume legacy files used GPT-4o (from config default)
                model_name = "openai_gpt_4o"

                model_files.append(
                    {
                        "prompt_version": prompt_version,
                        "model_name": model_name,
                        "file_name": file_path.name,
                        "file_path": str(file_path),
                        "is_legacy": True,
                    }
                )

    # Sort by prompt version then model name
    def sort_key(item):
        version = item["prompt_version"].lstrip("v")
        return (tuple(map(int, version.split("."))), item["model_name"])

    model_files.sort(key=sort_key)

    print(f"📁 Found {len(model_files)} extraction files:")
    for file_info in model_files:
        legacy_tag = " (legacy GPT-4o)" if file_info["is_legacy"] else ""
        print(
            f"   📄 {file_info['prompt_version']} + {file_info['model_name']}{legacy_tag}: {file_info['file_name']}"
        )

    return model_files


def get_discrepancies_by_model_with_legacy_support(
    prompt_version: str,
    model_name: str,
    outputs_dir: str = "outputs",
    data_dir: str = "data",
) -> Optional[pd.DataFrame]:
    """
    Get discrepancies for a model, with support for legacy untagged files.

    Args:
        prompt_version: Prompt version (e.g., "v1.1.2")
        model_name: Model name, or "openai_gpt_4o" for legacy files
        outputs_dir: Directory containing extraction results
        data_dir: Directory containing baseline data

    Returns:
        DataFrame with discrepancies, or None if files not found
    """
    # First try to find model-tagged file
    if model_name != "openai_gpt_4o":
        extraction_file = (
            f"text_extracted_data_prompt_{prompt_version}_model_{model_name}.csv"
        )
    else:
        # For GPT-4o, try both tagged and legacy formats
        extraction_file = (
            f"text_extracted_data_prompt_{prompt_version}_model_{model_name}.csv"
        )
        legacy_file = f"text_extracted_data_prompt_{prompt_version}.csv"

        extraction_path = Path(outputs_dir) / extraction_file
        legacy_path = Path(outputs_dir) / legacy_file

        if extraction_path.exists():
            print(f"🔍 Using model-tagged GPT-4o file: {extraction_file}")
        elif legacy_path.exists():
            print(f"🔍 Using legacy GPT-4o file (assuming GPT-4o): {legacy_file}")
            extraction_file = legacy_file
        else:
            print(f"❌ No GPT-4o extraction found for prompt {prompt_version}")
            print(f"   Checked: {extraction_file} and {legacy_file}")
            return None

    extraction_path = Path(outputs_dir) / extraction_file

    if not extraction_path.exists():
        print(f"❌ Extraction file not found: {extraction_path}")
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
