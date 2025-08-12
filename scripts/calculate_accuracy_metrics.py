#!/usr/bin/env python3
"""
Standalone script to calculate and log accuracy metrics for past LLM runs.
Can be used to retroactively add accuracy metrics to existing prompt logs.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import argparse

# Add src directory to path
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.append(str(src_dir))

from accuracy_metrics import AccuracyMetricsCalculator
from prompt_logger import PromptLogger
from prompt_manager import PromptManager


def load_extraction_results(prompt_version: str, outputs_dir: str) -> pd.DataFrame:
    """Load LLM extraction results for a given prompt version."""
    output_file = f'text_extracted_data_prompt_{prompt_version}.csv'
    output_path = Path(outputs_dir) / output_file
    
    if not output_path.exists():
        raise FileNotFoundError(f"No extraction results found: {output_path}")
    
    return pd.read_csv(output_path)


def load_baseline_data(data_dir: str) -> pd.DataFrame:
    """Load baseline data for comparison."""
    baseline_path = Path(data_dir) / "final_data_for_powerbi_with_kpi.csv"
    
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline data not found: {baseline_path}")
    
    baseline_df = pd.read_csv(baseline_path)
    
    # Filter to Week 28, 2025 (same as QMD)
    baseline_week28 = baseline_df[
        (baseline_df['Year'] == 2025) & 
        (baseline_df['WeekNumber'] == 28)
    ].copy()
    
    return baseline_week28


def perform_discrepancy_analysis(llm_data: pd.DataFrame, baseline_data: pd.DataFrame):
    """Perform the same discrepancy analysis as in the QMD."""
    # Import post-processing pipeline
    sys.path.append(str(src_dir))
    from post_processing import apply_post_processing_pipeline
    
    # Apply post-processing
    llm_processed = apply_post_processing_pipeline(llm_data.copy(), source="llm")
    baseline_processed = apply_post_processing_pipeline(baseline_data.copy(), source="baseline")
    
    # Create comparison keys
    llm_processed['comparison_key'] = llm_processed['Country'] + '_' + llm_processed['Event']
    baseline_processed['comparison_key'] = baseline_processed['Country'] + '_' + baseline_processed['Event']
    
    # Find common and unique records
    llm_keys = set(llm_processed['comparison_key'])
    baseline_keys = set(baseline_processed['comparison_key'])
    
    common_keys = llm_keys & baseline_keys
    llm_only_keys = llm_keys - baseline_keys
    baseline_only_keys = baseline_keys - llm_keys
    
    # Create comparison datasets
    llm_common = llm_processed[llm_processed['comparison_key'].isin(common_keys)].copy()
    baseline_common = baseline_processed[baseline_processed['comparison_key'].isin(common_keys)].copy()
    llm_only_df = llm_processed[llm_processed['comparison_key'].isin(llm_only_keys)].copy()
    baseline_only_df = baseline_processed[baseline_processed['comparison_key'].isin(baseline_only_keys)].copy()
    
    # Sort for alignment
    llm_common = llm_common.sort_values('comparison_key').reset_index(drop=True)
    baseline_common = baseline_common.sort_values('comparison_key').reset_index(drop=True)
    
    # Perform discrepancy analysis using merge (robust approach)
    merged_data = llm_common.merge(
        baseline_common, 
        on='comparison_key', 
        suffixes=('_llm', '_baseline'),
        how='inner'
    )
    
    # Compare fields and create discrepancy records
    discrepant_records = []
    fields_to_compare = ['TotalCases', 'CasesConfirmed', 'Deaths', 'CFR', 'Grade']
    
    def values_match(val1, val2, tolerance=0.01):
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
    
    for i in range(len(merged_data)):
        row = merged_data.iloc[i]
        
        record_discrepancies = {}
        has_discrepancy = False
        
        # Compare each field
        for field in fields_to_compare:
            llm_val = row.get(f'{field}_llm')
            baseline_val = row.get(f'{field}_baseline')
            
            if not values_match(llm_val, baseline_val):
                record_discrepancies[f'{field}_discrepancy'] = True
                record_discrepancies[f'llm_{field}'] = llm_val
                record_discrepancies[f'baseline_{field}'] = baseline_val
                has_discrepancy = True
            else:
                record_discrepancies[f'{field}_discrepancy'] = False
                record_discrepancies[f'llm_{field}'] = llm_val
                record_discrepancies[f'baseline_{field}'] = baseline_val
        
        if has_discrepancy:
            # Add record metadata
            record_discrepancies['comparison_key'] = row['comparison_key']
            record_discrepancies['Country'] = row.get('Country_llm', row.get('Country_baseline'))
            record_discrepancies['Event'] = row.get('Event_llm', row.get('Event_baseline'))
            discrepant_records.append(record_discrepancies)
    
    discrepancies_df = pd.DataFrame(discrepant_records)
    
    return discrepancies_df, llm_common, llm_only_df, baseline_only_df


def main():
    """Main function to calculate and log accuracy metrics."""
    parser = argparse.ArgumentParser(description="Calculate accuracy metrics for LLM extraction")
    parser.add_argument("--prompt-version", "-v", type=str, help="Prompt version to analyze (e.g., v1.1.0)")
    parser.add_argument("--outputs-dir", "-o", type=str, default="outputs", help="Directory containing extraction outputs")
    parser.add_argument("--data-dir", "-d", type=str, default="data", help="Directory containing baseline data")
    parser.add_argument("--current", "-c", action="store_true", help="Use current prompt version from PromptManager")
    
    args = parser.parse_args()
    
    # Determine prompt version
    if args.current:
        prompt_manager = PromptManager()
        current_prompt = prompt_manager.get_current_prompt("health_data_extraction")
        prompt_version = current_prompt["version"]
        print(f"🎯 Using current prompt version: {prompt_version}")
    elif args.prompt_version:
        prompt_version = args.prompt_version
        print(f"🎯 Using specified prompt version: {prompt_version}")
    else:
        print("❌ Must specify --prompt-version or use --current flag")
        sys.exit(1)
    
    try:
        # Load data
        print("📊 Loading extraction results and baseline data...")
        llm_data = load_extraction_results(prompt_version, args.outputs_dir)
        baseline_data = load_baseline_data(args.data_dir)
        
        print(f"   LLM results: {len(llm_data)} records")
        print(f"   Baseline data: {len(baseline_data)} records")
        
        # Perform discrepancy analysis
        print("\n🔍 Performing discrepancy analysis...")
        discrepancies_df, llm_common, llm_only_df, baseline_only_df = perform_discrepancy_analysis(
            llm_data, baseline_data
        )
        
        print(f"   Records compared: {len(llm_common)}")
        print(f"   Discrepant records: {len(discrepancies_df)}")
        print(f"   LLM-only records: {len(llm_only_df)}")
        print(f"   Baseline-only records: {len(baseline_only_df)}")
        
        # Calculate accuracy metrics
        print("\n📊 Calculating accuracy metrics...")
        calculator = AccuracyMetricsCalculator()
        accuracy_metrics = calculator.calculate_metrics_from_qmd_variables(
            discrepancies_df=discrepancies_df,
            llm_common=llm_common,
            llm_only_df=llm_only_df,
            baseline_only_df=baseline_only_df,
            prompt_version=prompt_version
        )
        
        # Display summary
        summary_text = calculator.generate_accuracy_summary_text(accuracy_metrics)
        print(f"\n{summary_text}")
        
        # Log to database
        print("\n💾 Logging accuracy metrics...")
        logger = PromptLogger()
        
        # Find latest log for this prompt version
        latest_log = logger.get_latest_log_for_prompt_version(prompt_version)
        
        if latest_log:
            log_id = latest_log['id']
            print(f"   Found log entry (ID: {log_id}) for prompt {prompt_version}")
            
            # Update with accuracy metrics
            update_success = logger.update_log_with_accuracy_metrics(
                log_identifier=str(log_id),
                accuracy_metrics=accuracy_metrics
            )
            
            if update_success:
                print("✅ Accuracy metrics successfully logged to database")
            else:
                print("❌ Failed to update log with accuracy metrics")
        else:
            print(f"⚠️ No existing log found for prompt version {prompt_version}")
            print("   Consider running the extraction first to create a log entry")
        
        # Save metrics to file for reference
        metrics_file = Path(args.outputs_dir) / f"accuracy_metrics_prompt_{prompt_version}.json"
        import json
        with open(metrics_file, 'w') as f:
            json.dump(accuracy_metrics, f, indent=2)
        print(f"📄 Accuracy metrics saved to: {metrics_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
