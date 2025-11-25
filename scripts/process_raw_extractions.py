#!/usr/bin/env python3
"""
Process raw extraction data and write to processed folders.

This script:
1. Downloads raw CSVs from blob storage (raw/monitoring/{llm,rule_based}_extractions/)
2. Applies post-processing pipeline (standardization, cleaning, etc.)
3. Uploads processed CSVs to blob storage (processed/monitoring/{llm,rule_based}_extractions/)

Usage:
    # Process all raw extractions
    python scripts/process_raw_extractions.py

    # Process only LLM extractions
    python scripts/process_raw_extractions.py --source llm

    # Process only rule-based extractions
    python scripts/process_raw_extractions.py --source rule-based

    # Process a specific week/year
    python scripts/process_raw_extractions.py --week 42 --year 2025

    # Apply experimental gap-filling corrections to LLM extractions
    python scripts/process_raw_extractions.py --source llm --correct-gap-fill

    # Process one file for testing
    python scripts/process_raw_extractions.py --limit 1

Environment variables required:
    DSCI_AZ_BLOB_DEV_SAS_WRITE: Azure Blob SAS token
    STAGE: dev or prod (default: dev)
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Optional

import ocha_stratus as stratus
import pandas as pd
from azure.storage.blob import BlobServiceClient

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.config import Config
from src.post_processing import apply_post_processing_pipeline, validate_post_processing


def list_raw_extractions(source: str, stage: str = "dev") -> List[Dict[str, str]]:
    """
    List all raw extraction CSVs in blob storage.

    Args:
        source: "llm" or "rule-based"
        stage: Azure stage (dev/prod)

    Returns:
        List of dicts with blob_name, filename, last_modified
    """
    container = Config.BLOB_CONTAINER

    # Get the appropriate raw path from centralized config
    if source == "llm":
        blob_prefix = Config.get_blob_paths()["raw_llm_extractions"]
    elif source == "rule-based":
        blob_prefix = Config.get_blob_paths()["raw_rule_based_extractions"]
    else:
        raise ValueError(f"Invalid source: {source}")

    print(f"🔍 Listing raw {source} extractions from: {blob_prefix}")

    try:
        # Get SAS token
        sas_token = os.getenv(f"DSCI_AZ_BLOB_{stage.upper()}_SAS_WRITE")
        if not sas_token:
            print(f"❌ No SAS token found for stage {stage}")
            return []

        # Connect to blob
        account_url = f"https://imb0chd0{stage}.blob.core.windows.net"
        blob_service_client = BlobServiceClient(account_url=account_url, credential=sas_token)
        container_client = blob_service_client.get_container_client(container)

        # List blobs
        blobs = list(container_client.list_blobs(name_starts_with=blob_prefix))

        # Filter for CSV files, excluding those in archive subfolder
        csv_blobs = [
            b for b in blobs
            if b.name.endswith('.csv') and '/archive/' not in b.name
        ]

        if not csv_blobs:
            print(f"ℹ️  No CSV files found in {blob_prefix}")
            return []

        # Sort by last_modified (most recent first)
        csv_blobs.sort(key=lambda x: x.last_modified, reverse=True)

        results = []
        for blob in csv_blobs:
            filename = Path(blob.name).name
            results.append({
                'blob_name': blob.name,
                'filename': filename,
                'last_modified': blob.last_modified,
            })

        print(f"✅ Found {len(results)} CSV files")
        return results

    except Exception as e:
        print(f"❌ Error listing blobs: {e}")
        return []


def download_csv_from_blob(blob_name: str, local_path: Path, stage: str = "dev") -> bool:
    """
    Download a CSV from Azure Blob Storage.

    Args:
        blob_name: Full blob path
        local_path: Local path to save the CSV
        stage: Azure stage (dev/prod)

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"  📥 Downloading {Path(blob_name).name}...")

        # Use stratus to download
        blob_data = stratus.load_blob_data(
            blob_name=blob_name,
            stage=stage,
            container_name=Config.BLOB_CONTAINER,
        )

        # Write to local file
        with open(local_path, 'wb') as f:
            f.write(blob_data)

        # Verify file size
        file_size = local_path.stat().st_size / 1024  # KB
        print(f"  ✅ Downloaded ({file_size:.1f} KB)")

        return True

    except Exception as e:
        print(f"  ❌ Error downloading: {e}")
        return False


def upload_csv_to_blob(
    csv_path: Path,
    source: str,
    original_filename: str,
    stage: str = "dev"
) -> Optional[str]:
    """
    Upload processed CSV to Azure Blob Storage.

    Args:
        csv_path: Local path to processed CSV
        source: "llm" or "rule-based"
        original_filename: Original filename (for naming)
        stage: Azure stage (dev/prod)

    Returns:
        Blob path if successful, None otherwise
    """
    try:
        # Get the appropriate processed path from centralized config
        if source == "llm":
            blob_base_path = Config.get_blob_paths()["processed_llm_extractions"]
        elif source == "rule-based":
            blob_base_path = Config.get_blob_paths()["processed_rule_based_extractions"]
        else:
            raise ValueError(f"Invalid source: {source}")

        # Use original filename with "_processed" suffix
        processed_filename = original_filename.replace('.csv', '_processed.csv')
        blob_path = f"{blob_base_path}{processed_filename}"

        print(f"  📤 Uploading to {blob_path}...")

        # Read CSV as DataFrame
        df = pd.read_csv(csv_path)

        # Upload using stratus
        stratus.upload_csv_to_blob(
            df=df,
            blob_name=blob_path,
            stage=stage,
            container_name=Config.BLOB_CONTAINER,
        )

        print(f"  ✅ Uploaded successfully")

        return blob_path

    except Exception as e:
        print(f"  ❌ Error uploading: {e}")
        return None


def process_single_extraction(
    blob_name: str,
    filename: str,
    source: str,
    temp_dir: Path,
    stage: str = "dev",
    correct_gap_fill_errors: bool = False,
) -> Dict[str, any]:
    """
    Process a single raw extraction CSV.

    Args:
        blob_name: Full blob path to raw CSV
        filename: Filename
        source: "llm" or "rule-based"
        temp_dir: Temporary directory for processing
        stage: Azure stage
        correct_gap_fill_errors: Apply experimental gap-fill corrections (LLM only)

    Returns:
        Dict with processing results
    """
    print(f"\n{'='*60}")
    print(f"Processing: {filename}")
    print(f"{'='*60}")

    result = {
        'filename': filename,
        'source': source,
        'status': 'failed',
        'records_original': 0,
        'records_processed': 0,
        'blob_path': None,
    }

    try:
        # Step 1: Download raw CSV
        raw_path = temp_dir / filename
        if not download_csv_from_blob(blob_name, raw_path, stage):
            result['error'] = 'Download failed'
            return result

        # Step 2: Load CSV
        print(f"  📊 Loading data...")
        df_raw = pd.read_csv(raw_path)
        result['records_original'] = len(df_raw)
        print(f"  ✅ Loaded {len(df_raw)} records")

        # Step 3: Apply post-processing
        print(f"  🧹 Applying post-processing pipeline...")

        # Determine source name for post-processing
        source_name = "llm" if source == "llm" else "baseline"

        # Apply post-processing with optional gap-fill correction
        df_processed = apply_post_processing_pipeline(
            df_raw,
            source=source_name,
            correct_gap_fill_errors=(source == "llm" and correct_gap_fill_errors)
        )

        result['records_processed'] = len(df_processed)

        # Validate processing
        if not validate_post_processing(df_raw, df_processed):
            print(f"  ⚠️  Validation failed - using original data")
            df_processed = df_raw

        # Step 4: Save processed CSV
        processed_path = temp_dir / f"{filename.replace('.csv', '_processed.csv')}"
        df_processed.to_csv(processed_path, index=False)
        print(f"  💾 Saved processed data locally")

        # Step 5: Upload to blob
        blob_path = upload_csv_to_blob(
            processed_path,
            source,
            filename,
            stage
        )

        if blob_path:
            result['status'] = 'success'
            result['blob_path'] = blob_path
        else:
            result['error'] = 'Upload failed'

    except Exception as e:
        print(f"  ❌ Error processing: {e}")
        import traceback
        traceback.print_exc()
        result['error'] = str(e)

    return result


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Process raw extractions and upload to processed folders"
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["llm", "rule-based", "both"],
        default="both",
        help="Which extraction source to process (default: both)"
    )
    parser.add_argument(
        "--correct-gap-fill",
        action="store_true",
        help="Apply experimental gap-filling corrections to LLM extractions"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only process N most recent files (for testing)"
    )
    parser.add_argument(
        "--week",
        type=int,
        help="Process only files from this week number"
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Process only files from this year"
    )

    args = parser.parse_args()

    # Get environment
    stage = os.getenv("STAGE", "dev")

    print("=" * 60)
    print("RAW EXTRACTION PROCESSING PIPELINE")
    print("=" * 60)
    print(f"Stage: {stage}")
    print(f"Source: {args.source}")
    if args.week is not None and args.year is not None:
        print(f"Week/Year filter: Week {args.week}, {args.year}")
    if args.correct_gap_fill:
        print("⚠️  Gap-filling corrections: ENABLED (experimental)")
    print()

    # Determine which sources to process
    sources = []
    if args.source in ["llm", "both"]:
        sources.append("llm")
    if args.source in ["rule-based", "both"]:
        sources.append("rule-based")

    # Process each source
    all_results = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for source in sources:
            print(f"\n{'='*60}")
            print(f"Processing {source.upper()} extractions")
            print(f"{'='*60}")

            # List raw extractions
            raw_files = list_raw_extractions(source, stage)

            if not raw_files:
                print(f"ℹ️  No raw {source} extractions found")
                continue

            # Filter by week/year if specified
            if args.week is not None and args.year is not None:
                import re
                # Match filenames like "OEW42-2025_gpt-4o_1234567890.csv"
                # or "OEW42-2025_rule-based_1234567890.csv"
                week_pattern = f"OEW{args.week:02d}-{args.year}"
                raw_files = [
                    f for f in raw_files
                    if week_pattern in f['filename']
                ]
                if not raw_files:
                    print(f"ℹ️  No files found for Week {args.week}, {args.year}")
                    continue
                print(f"ℹ️  Filtered to Week {args.week}, {args.year}: {len(raw_files)} file(s)")

            # Apply limit if specified
            if args.limit:
                raw_files = raw_files[:args.limit]
                print(f"ℹ️  Limited to {args.limit} most recent files")

            # Process each file
            for file_info in raw_files:
                result = process_single_extraction(
                    blob_name=file_info['blob_name'],
                    filename=file_info['filename'],
                    source=source,
                    temp_dir=temp_path,
                    stage=stage,
                    correct_gap_fill_errors=args.correct_gap_fill,
                )
                all_results.append(result)

    # Final summary
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")

    successful = [r for r in all_results if r['status'] == 'success']
    failed = [r for r in all_results if r['status'] == 'failed']

    print(f"Total files processed: {len(all_results)}")
    print(f"  ✅ Successful: {len(successful)}")
    print(f"  ❌ Failed: {len(failed)}")

    if failed:
        print("\nFailed files:")
        for r in failed:
            error_msg = r.get('error', 'Unknown error')
            print(f"  - {r['filename']}: {error_msg}")

    print()

    # Exit code
    sys.exit(0 if len(failed) == 0 else 1)


if __name__ == "__main__":
    main()
