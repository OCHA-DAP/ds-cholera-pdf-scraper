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

import sys
import os
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import json

# Add src directory to path
script_dir = Path(__file__).parent
src_dir = script_dir / "src"
sys.path.append(str(src_dir))


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
    pattern = re.compile(r'text_extracted_data_prompt_(v\d+\.\d+\.\d+)\.csv')
    
    found_files = []
    
    for file_path in outputs_path.glob("*.csv"):
        match = pattern.match(file_path.name)
        if match:
            version = match.group(1)
            found_files.append({
                'version': version,
                'file_path': str(file_path),
                'file_name': file_path.name
            })
    
    # Sort by version (semantic versioning)
    def version_key(item):
        version = item['version'].lstrip('v')
        return tuple(map(int, version.split('.')))
    
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
        (baseline_df['Year'] == 2025) & 
        (baseline_df['WeekNumber'] == 28)
    ].copy()
    
    return baseline_week28


def perform_discrepancy_analysis(llm_data: pd.DataFrame, baseline_data: pd.DataFrame):
    """
    Perform the same discrepancy analysis as in the QMD.
    Returns discrepancies_df, llm_common, llm_only_df, baseline_only_df
    """
    # Import post-processing pipeline
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


def check_existing_accuracy_metrics(logger, prompt_version: str) -> bool:
    """Check if accuracy metrics already exist for a prompt version."""
    latest_log = logger.get_latest_log_for_prompt_version(prompt_version)
    
    if not latest_log:
        return False
    
    custom_metrics = latest_log.get('custom_metrics')
    if custom_metrics:
        try:
            metrics_data = json.loads(custom_metrics) if isinstance(custom_metrics, str) else custom_metrics
            return 'accuracy_metrics' in metrics_data
        except:
            return False
    
    return False


def process_prompt_version(
    version: str, 
    file_path: str, 
    baseline_data: pd.DataFrame,
    force: bool = False,
    dry_run: bool = False
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
            return {'dry_run': True, 'version': version}
        
        # Check if accuracy metrics already exist
        from prompt_logger import PromptLogger
        logger = PromptLogger()
        
        if not force and check_existing_accuracy_metrics(logger, version):
            print("   ⚠️ Accuracy metrics already exist. Use --force to overwrite.")
            return None
        
        # Perform discrepancy analysis
        print("   🔍 Performing discrepancy analysis...")
        discrepancies_df, llm_common, llm_only_df, baseline_only_df = perform_discrepancy_analysis(
            llm_data, baseline_data
        )
        
        print(f"   📈 Analysis results:")
        print(f"      Common records: {len(llm_common)}")
        print(f"      Discrepant records: {len(discrepancies_df)}")
        print(f"      LLM-only: {len(llm_only_df)}, Baseline-only: {len(baseline_only_df)}")
        
        # Calculate accuracy metrics
        from accuracy_metrics import AccuracyMetricsCalculator
        calculator = AccuracyMetricsCalculator()
        accuracy_metrics = calculator.calculate_metrics_from_qmd_variables(
            discrepancies_df=discrepancies_df,
            llm_common=llm_common,
            llm_only_df=llm_only_df,
            baseline_only_df=baseline_only_df,
            prompt_version=version
        )
        
        print(f"   📊 Overall Accuracy: {accuracy_metrics['overall_accuracy_percent']}%")
        print(f"   📊 Coverage Rate: {accuracy_metrics['coverage_rate_percent']}%")
        
        # Log to database
        latest_log = logger.get_latest_log_for_prompt_version(version)
        
        if latest_log:
            log_id = latest_log['id']
            print(f"   💾 Updating log entry (ID: {log_id})")
            
            update_success = logger.update_log_with_accuracy_metrics(
                log_identifier=str(log_id),
                accuracy_metrics=accuracy_metrics
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
        epilog=__doc__
    )
    
    parser.add_argument("--version", "-v", type=str, 
                       help="Process specific prompt version only (e.g., v1.1.0)")
    parser.add_argument("--outputs", "-o", type=str, default="outputs",
                       help="Directory containing extraction files (default: outputs)")
    parser.add_argument("--data", "-d", type=str, default="data", 
                       help="Directory containing baseline data (default: data)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be processed without making changes")
    parser.add_argument("--force", "-f", action="store_true",
                       help="Overwrite existing accuracy metrics")
    
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
            found_files = [f for f in found_files if f['version'] == args.version]
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
                version=file_info['version'],
                file_path=file_info['file_path'],
                baseline_data=baseline_data,
                force=args.force,
                dry_run=args.dry_run
            )
            
            if result:
                results.append({
                    'version': file_info['version'],
                    'metrics': result
                })
        
        # Summary
        print("\n" + "=" * 50)
        print("📋 SUMMARY")
        print("=" * 50)
        
        if args.dry_run:
            print(f"🔍 Would process {len(results)} versions")
        else:
            successful = len([r for r in results if r['metrics'] and not r['metrics'].get('dry_run')])
            print(f"✅ Successfully processed {successful}/{len(found_files)} versions")
            
            if successful > 0:
                print("\n📊 Accuracy Summary:")
                for result in results:
                    if result['metrics'] and not result['metrics'].get('dry_run'):
                        version = result['version']
                        accuracy = result['metrics']['overall_accuracy_percent']
                        coverage = result['metrics']['coverage_rate_percent']
                        print(f"   {version}: {accuracy}% accuracy, {coverage}% coverage")
        
        print("\n🎯 Backfill complete!")
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
