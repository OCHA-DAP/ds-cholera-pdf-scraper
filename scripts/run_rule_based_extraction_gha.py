#!/usr/bin/env python3
"""
GitHub Actions wrapper for rule-based (Tabula) extraction from WHO cholera PDFs.

This script runs in GitHub Actions to:
1. Download the latest PDF from Azure Blob Storage
2. Run rule-based table extraction using Tabula
3. Log results to JSONL
4. Upload extracted data and logs back to blob

Usage:
    # Extract from latest PDF in blob
    python scripts/run_rule_based_extraction_gha.py --week latest

    # Extract from specific week
    python scripts/run_rule_based_extraction_gha.py --week 42 --year 2025

Environment variables required:
    DSCI_AZ_BLOB_DEV_SAS_WRITE: Azure Blob SAS token
    STAGE: dev or prod (default: dev)
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import ocha_stratus as stratus
import pandas as pd
from azure.storage.blob import BlobServiceClient

from src.config import Config
from src.rule_based_extract import extract_table_from_pdf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ExtractionRunMetadata:
    """Metadata about an extraction run execution."""
    # Bulletin info
    week: Optional[int]
    year: Optional[int]
    pdf_name: str

    # Run context
    run_date: str
    status: str  # "success" or "failed"
    error_message: Optional[str]
    runner: str  # "github-actions" or "local"
    trigger: Optional[str]  # "schedule", "workflow_dispatch", etc.
    run_id: Optional[str]
    run_url: Optional[str]

    # Extraction details
    extraction_method: str  # "rule-based-tabula"
    records_extracted: int
    execution_time_seconds: float

    # Outcome
    blob_uploaded: bool
    csv_blob_path: Optional[str]
    log_blob_path: Optional[str]

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    def to_jsonl(self) -> str:
        """Convert to JSONL format (single line JSON)."""
        return json.dumps(self.to_dict())


def get_run_context() -> Dict[str, Optional[str]]:
    """
    Detect execution context (GitHub Actions vs local).

    Returns dict with runner, run_id, run_url, and trigger.
    """
    github_run_id = os.getenv("GITHUB_RUN_ID")

    if github_run_id:
        github_repo = os.getenv("GITHUB_REPOSITORY", "unknown/unknown")
        return {
            "runner": "github-actions",
            "run_id": github_run_id,
            "run_url": f"https://github.com/{github_repo}/actions/runs/{github_run_id}",
            "trigger": os.getenv("GITHUB_EVENT_NAME"),
        }
    else:
        return {
            "runner": "local",
            "run_id": None,
            "run_url": None,
            "trigger": None,
        }


def get_latest_pdf_from_blob(stage: str = "dev") -> Optional[Dict[str, Any]]:
    """
    Get the latest PDF from Azure Blob Storage.

    Returns:
        Dict with blob_name, filename, week, year, or None if not found
    """
    container = Config.BLOB_CONTAINER
    proj_dir = Config.BLOB_PROJ_DIR

    print("🔍 Looking for latest PDF in blob storage...")

    try:
        # Get SAS token
        sas_token = os.getenv(f"DSCI_AZ_BLOB_{stage.upper()}_SAS_WRITE")
        if not sas_token:
            print(f"❌ No SAS token found for stage {stage}")
            return None

        # Connect to blob
        account_url = f"https://imb0chd0{stage}.blob.core.windows.net"
        blob_service_client = BlobServiceClient(account_url=account_url, credential=sas_token)
        container_client = blob_service_client.get_container_client(container)

        # List blobs in monitoring/pdfs directory (use centralized path)
        blob_prefix = Config.get_blob_paths()["raw_pdfs"]
        blobs = list(container_client.list_blobs(name_starts_with=blob_prefix))

        # Filter for PDF files
        pdf_blobs = [b for b in blobs if b.name.endswith('.pdf')]

        if not pdf_blobs:
            print("❌ No PDFs found in blob storage")
            return None

        # Sort by last_modified (most recent first)
        pdf_blobs.sort(key=lambda x: x.last_modified, reverse=True)
        latest_blob = pdf_blobs[0]

        # Extract week/year from filename (format: OEW42-2025.pdf)
        import re
        filename = Path(latest_blob.name).name
        match = re.match(r'OEW(\d+)-(\d{4})\.pdf', filename)

        if match:
            week = int(match.group(1))
            year = int(match.group(2))
        else:
            week = None
            year = None

        print(f"✅ Found latest PDF: {filename}")
        if week and year:
            print(f"   Week {week}, Year {year}")

        return {
            'blob_name': latest_blob.name,
            'filename': filename,
            'week': week,
            'year': year,
            'last_modified': latest_blob.last_modified,
        }

    except Exception as e:
        print(f"❌ Error listing blobs: {e}")
        return None


def download_pdf_from_blob(blob_name: str, local_path: Path, stage: str = "dev") -> bool:
    """
    Download a PDF from Azure Blob Storage.

    Args:
        blob_name: Full blob path
        local_path: Local path to save the PDF
        stage: Azure stage (dev/prod)

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"📥 Downloading PDF from blob...")

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
        file_size = local_path.stat().st_size / 1024 / 1024  # MB
        print(f"✅ Downloaded {local_path.name} ({file_size:.2f} MB)")

        return True

    except Exception as e:
        print(f"❌ Error downloading PDF: {e}")
        return False


def upload_csv_to_blob(
    csv_path: Path,
    pdf_name: str,
    timestamp: int,
    stage: str = "dev"
) -> Optional[str]:
    """
    Upload extracted CSV to Azure Blob Storage.

    Args:
        csv_path: Local path to CSV file
        pdf_name: Name of the PDF (e.g., "OEW42-2025.pdf")
        timestamp: Timestamp for filename
        stage: Azure stage (dev/prod)

    Returns:
        Blob path if successful, None otherwise
    """
    try:
        # Create filename: <pdf_stem>_rule-based_<timestamp>.csv
        pdf_stem = Path(pdf_name).stem
        new_filename = f"{pdf_stem}_rule-based_{timestamp}.csv"

        # Upload to raw/monitoring/rule_based_extractions/ using centralized config
        blob_base_path = Config.get_blob_paths()["raw_rule_based_extractions"]
        blob_path = f"{blob_base_path}{new_filename}"

        print(f"📤 Uploading extraction CSV to blob...")
        print(f"   Filename: {new_filename}")

        # Read CSV as DataFrame
        df = pd.read_csv(csv_path)

        # Upload using stratus
        stratus.upload_csv_to_blob(
            df=df,
            blob_name=blob_path,
            stage=stage,
            container_name=Config.BLOB_CONTAINER,
        )

        print(f"✅ Uploaded to {blob_path}")

        return blob_path

    except Exception as e:
        print(f"❌ Error uploading CSV: {e}")
        return None


def upload_log_to_blob(log_path: Path, stage: str = "dev") -> Optional[str]:
    """
    Upload JSONL log file to Azure Blob Storage.

    Args:
        log_path: Local path to JSONL log
        stage: Azure stage (dev/prod)

    Returns:
        Blob path if successful, None otherwise
    """
    try:
        proj_dir = Config.BLOB_PROJ_DIR
        blob_path = f"{proj_dir}/processed/logs/rule_based_extraction_log.jsonl"

        print(f"📤 Uploading execution log to blob...")

        with open(log_path, "rb") as f:
            stratus.upload_blob_data(
                data=f,
                blob_name=blob_path,
                stage=stage,
                container_name=Config.BLOB_CONTAINER,
                content_type="application/x-ndjson",
            )

        print(f"✅ Uploaded log to {blob_path}")

        return blob_path

    except Exception as e:
        print(f"❌ Error uploading log: {e}")
        return None


def download_log_from_blob(local_path: Path, stage: str = "dev") -> bool:
    """
    Download existing log file from blob storage.

    Args:
        local_path: Local path to save log
        stage: Azure stage

    Returns:
        True if successful, False otherwise
    """
    proj_dir = Config.BLOB_PROJ_DIR
    blob_path = f"{proj_dir}/processed/logs/rule_based_extraction_log.jsonl"

    try:
        print(f"📥 Downloading existing log from blob...")
        blob_data = stratus.load_blob_data(
            blob_name=blob_path,
            stage=stage,
            container_name=Config.BLOB_CONTAINER,
        )
        with open(local_path, 'wb') as f:
            f.write(blob_data)
        print(f"✅ Downloaded existing log")
        return True
    except Exception:
        print("ℹ️  No existing log found (this is normal for first run)")
        return False


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run rule-based extraction in GitHub Actions"
    )
    parser.add_argument(
        "--week",
        type=str,
        default="latest",
        help="Week number or 'latest' (default: latest)"
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Year (required if week is specified)"
    )

    args = parser.parse_args()

    # Get environment
    stage = os.getenv("STAGE", "dev")

    print("=" * 60)
    print("GitHub Actions Rule-Based Extraction Workflow")
    print("=" * 60)
    print(f"Stage: {stage}")
    print(f"Week: {args.week}")
    if args.year:
        print(f"Year: {args.year}")
    print()

    # Get run context
    run_context = get_run_context()
    start_time = time.time()
    run_date = datetime.now().isoformat()
    timestamp = int(time.time())

    # Initialize run metadata
    run_metadata = None
    pdf_info = None

    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        try:
            # Step 1: Get PDF from blob
            if args.week == "latest":
                pdf_info = get_latest_pdf_from_blob(stage=stage)
                if not pdf_info:
                    raise Exception("No PDF found in blob storage")

                blob_name = pdf_info['blob_name']
                filename = pdf_info['filename']
                week = pdf_info['week']
                year = pdf_info['year']
            else:
                # Construct blob name from week/year
                if not args.year:
                    raise Exception("--year is required when --week is specified")

                week = int(args.week)
                year = args.year
                filename = f"OEW{week:02d}-{year}.pdf"
                # Use centralized blob path from Config
                blob_base_path = Config.get_blob_paths()["raw_pdfs"]
                blob_name = f"{blob_base_path}{filename}"
                pdf_info = {
                    'blob_name': blob_name,
                    'filename': filename,
                    'week': week,
                    'year': year,
                }

            # Step 2: Download PDF
            local_pdf_path = temp_path / filename
            if not download_pdf_from_blob(blob_name, local_pdf_path, stage=stage):
                raise Exception("Failed to download PDF")

            print()

            # Step 3: Run extraction
            print("=" * 60)
            print("🤖 Running Rule-Based Extraction (Tabula)")
            print("=" * 60)
            print(f"PDF: {filename}")
            print(f"Week: {week}, Year: {year}")
            print()

            df = extract_table_from_pdf(
                pdf_path=local_pdf_path,
                week=week,
                year=year
            )

            execution_time = time.time() - start_time

            print()
            print("=" * 60)
            print("✅ Extraction Complete")
            print("=" * 60)
            print(f"Records extracted: {len(df)}")
            print(f"Execution time: {execution_time:.2f}s")
            print()

            # Step 4: Save CSV locally
            csv_filename = f"extraction_{timestamp}_rule_based.csv"
            csv_path = temp_path / csv_filename
            df.to_csv(csv_path, index=False)
            print(f"💾 Saved CSV: {csv_path.name}")

            # Step 5: Upload CSV to blob
            csv_blob_path = upload_csv_to_blob(
                csv_path=csv_path,
                pdf_name=filename,
                timestamp=timestamp,
                stage=stage
            )

            # Create success metadata
            run_metadata = ExtractionRunMetadata(
                week=week,
                year=year,
                pdf_name=filename,
                run_date=run_date,
                status="success",
                error_message=None,
                runner=run_context["runner"],
                trigger=run_context["trigger"],
                run_id=run_context["run_id"],
                run_url=run_context["run_url"],
                extraction_method="rule-based-tabula",
                records_extracted=len(df),
                execution_time_seconds=execution_time,
                blob_uploaded=csv_blob_path is not None,
                csv_blob_path=csv_blob_path,
                log_blob_path=None,  # Will be set after log upload
            )

        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ Extraction failed: {e}")
            import traceback
            traceback.print_exc()

            # Create failed run metadata
            run_metadata = ExtractionRunMetadata(
                week=pdf_info['week'] if pdf_info else (int(args.week) if args.week != 'latest' else None),
                year=pdf_info['year'] if pdf_info else args.year,
                pdf_name=pdf_info['filename'] if pdf_info else f"OEW{args.week}-{args.year}.pdf",
                run_date=run_date,
                status="failed",
                error_message=str(e),
                runner=run_context["runner"],
                trigger=run_context["trigger"],
                run_id=run_context["run_id"],
                run_url=run_context["run_url"],
                extraction_method="rule-based-tabula",
                records_extracted=0,
                execution_time_seconds=execution_time,
                blob_uploaded=False,
                csv_blob_path=None,
                log_blob_path=None,
            )

        finally:
            # Step 6: Append to log and upload
            if run_metadata:
                # Download existing log
                log_path = temp_path / "rule_based_extraction_log.jsonl"
                download_log_from_blob(log_path, stage=stage)

                # Append new run
                with open(log_path, "a") as f:
                    f.write(run_metadata.to_jsonl() + "\n")

                # Upload log
                log_blob_path = upload_log_to_blob(log_path, stage=stage)
                run_metadata.log_blob_path = log_blob_path

                print()
                print("=" * 60)
                if run_metadata.status == "success":
                    print("✅ Workflow Complete!")
                    print("=" * 60)
                    print(f"PDF: {run_metadata.pdf_name}")
                    print(f"Records: {run_metadata.records_extracted}")
                    print(f"Time: {run_metadata.execution_time_seconds:.2f}s")
                    print()
                    print("📊 Files uploaded to blob:")
                    print(f"   CSV: {run_metadata.csv_blob_path}")
                    print(f"   Log: {run_metadata.log_blob_path}")
                else:
                    print("❌ Workflow Failed")
                    print("=" * 60)
                    print(f"Error: {run_metadata.error_message}")
                    sys.exit(1)


if __name__ == "__main__":
    main()
