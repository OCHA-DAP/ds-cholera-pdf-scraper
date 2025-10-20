#!/usr/bin/env python3
"""
GitHub Actions extraction wrapper for WHO cholera PDFs.

This script runs in GitHub Actions to:
1. Download the latest PDF from Azure Blob Storage
2. Run LLM-based extraction using the existing pipeline
3. Log results to Parquet files
4. Upload logs and extracted data back to blob

Usage:
    # Extract from latest PDF in blob
    python scripts/run_extraction_gha.py --week latest

    # Extract from specific week
    python scripts/run_extraction_gha.py --week 42 --year 2025

    # Use specific model
    python scripts/run_extraction_gha.py --week latest --model gpt-5

Environment variables required:
    DSCI_AZ_OPENAI_API_KEY_WHO_CHOLERA: OpenAI API key
    DSCI_AZ_BLOB_DEV_SAS_WRITE: Azure Blob SAS token
    STAGE: dev or prod (default: dev)
"""

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Parse args FIRST to get log_backend before importing src modules
# This is necessary because Config.py reads LOG_BACKEND at import time
parser = argparse.ArgumentParser(description="Run extraction in GitHub Actions")
parser.add_argument("--log-backend", type=str, default="duckdb",
                   choices=["sqlite", "duckdb", "jsonl"],
                   help="Logging backend to use (default: duckdb)")
parser.add_argument("--week", type=str, default="latest")
parser.add_argument("--year", type=int)
parser.add_argument("--model", type=str, default="gpt-5")
parser.add_argument("--prompt-version", type=str, default="v1.4.7")
parser.add_argument("--preprocessor", type=str, default="none-pdf-upload")

# Parse args early
early_args, _ = parser.parse_known_args()

# Set LOG_BACKEND before importing any src modules
os.environ['LOG_BACKEND'] = early_args.log_backend

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import ocha_stratus as stratus
import pandas as pd

from src.config import Config
from src.cloud_logging import DuckDBLogger
from src.prompt_logger import PromptLogger


def get_latest_pdf_from_blob(stage: str = "dev") -> Optional[Dict[str, Any]]:
    """
    Get the latest PDF from Azure Blob Storage.

    Returns:
        Dict with pdf_path, week, year, or None if not found
    """
    container = Config.BLOB_CONTAINER
    proj_dir = Config.BLOB_PROJ_DIR

    print("🔍 Looking for latest PDF in blob storage...")

    # List all PDFs in the monitoring directory
    try:
        from azure.storage.blob import BlobServiceClient

        # Get SAS token
        sas_token = os.getenv(f"DSCI_AZ_BLOB_{stage.upper()}_SAS_WRITE")
        if not sas_token:
            print(f"❌ No SAS token found for stage {stage}")
            return None

        # Connect to blob
        account_url = f"https://imb0chd0{stage}.blob.core.windows.net"
        blob_service_client = BlobServiceClient(account_url=account_url, credential=sas_token)
        container_client = blob_service_client.get_container_client(container)

        # List blobs in monitoring directory
        blob_prefix = f"{proj_dir}/raw/monitoring/"
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
        blob_name: Full blob path (e.g., "ds-cholera-pdf-scraper/raw/monitoring/OEW42-2025.pdf")
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


def upload_parquet_log_to_blob(
    parquet_path: Path,
    log_type: str,
    stage: str = "dev"
) -> bool:
    """
    Upload a parquet log file to Azure Blob Storage.

    Args:
        parquet_path: Local path to parquet file
        log_type: Type of log ("prompt" or "preprocessing")
        stage: Azure stage (dev/prod)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Determine blob path based on log type
        proj_dir = Config.BLOB_PROJ_DIR

        if log_type == "prompt":
            blob_path = f"{proj_dir}/processed/logs/prompt_logs/{parquet_path.name}"
        elif log_type == "preprocessing":
            blob_path = f"{proj_dir}/processed/logs/tabular_preprocessing_logs/{parquet_path.name}"
        else:
            raise ValueError(f"Unknown log_type: {log_type}")

        print(f"📤 Uploading {parquet_path.name} to blob...")

        # Read parquet as DataFrame
        df = pd.read_parquet(parquet_path)

        # Upload using stratus
        stratus.upload_parquet_to_blob(
            df=df,
            blob_name=blob_path,
            stage=stage,
            container_name=Config.BLOB_CONTAINER,
        )

        print(f"✅ Uploaded to {blob_path}")

        return True

    except Exception as e:
        print(f"❌ Error uploading parquet: {e}")
        return False


def upload_csv_to_blob(
    csv_path: Path,
    pdf_name: str,
    run_id: int,
    model: str,
    stage: str = "dev"
) -> bool:
    """
    Upload extracted CSV to Azure Blob Storage with standardized naming.

    Args:
        csv_path: Local path to CSV file
        pdf_name: Name of the PDF that was extracted (e.g., "OEW42-2025.pdf")
        run_id: Run ID from the extraction log
        model: Model name used for extraction
        stage: Azure stage (dev/prod)

    Returns:
        True if successful, False otherwise
    """
    try:
        proj_dir = Config.BLOB_PROJ_DIR

        # Create standardized filename: <pdf_stem>_<model>_<run_id>.csv
        # Example: OEW42-2025_gpt-5_1729468934.csv
        pdf_stem = Path(pdf_name).stem
        model_clean = model.replace("/", "-").replace("_", "-")
        new_filename = f"{pdf_stem}_{model_clean}_{run_id}.csv"

        # Upload to processed/llm_extractions/
        blob_path = f"{proj_dir}/processed/llm_extractions/{new_filename}"

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

        return True

    except Exception as e:
        print(f"❌ Error uploading CSV: {e}")
        return False


def run_extraction_pipeline(
    pdf_path: Path,
    model: str = "gpt-5",
    prompt_version: str = "v1.4.7",
    preprocessor: str = "none-pdf-upload",
    output_dir: Path = None,
) -> Optional[Dict[str, Any]]:
    """
    Run the extraction pipeline on a PDF.

    Args:
        pdf_path: Path to PDF file
        model: Model to use
        prompt_version: Prompt version
        preprocessor: Preprocessor to use
        output_dir: Directory for output CSV

    Returns:
        Dict with extraction results, csv_path, run_id, or None if failed
    """
    try:
        print("=" * 60)
        print("🤖 Running LLM Extraction Pipeline")
        print("=" * 60)
        print(f"PDF: {pdf_path.name}")
        print(f"Model: {model}")
        print(f"Prompt: {prompt_version}")
        print(f"Preprocessor: {preprocessor}")
        print()

        # Set up output directory
        if output_dir is None:
            output_dir = Path(tempfile.gettempdir()) / "cholera_extraction"
            output_dir.mkdir(exist_ok=True)

        # Run extraction based on preprocessor (matching llm_text_extract.py __main__ logic)
        start_time = time.time()

        if preprocessor == "pdfplumber":
            from src.llm_text_extract import extract_data_with_pdfplumber_preprocessing
            print("🔍 Running pdfplumber preprocessing + LLM extraction...")
            extracted_data, call_id = extract_data_with_pdfplumber_preprocessing(
                str(pdf_path), model_name=model, prompt_version=prompt_version
            )
            # Data is returned as list of dicts, convert to DataFrame
            df = pd.DataFrame(extracted_data) if extracted_data else pd.DataFrame()

        elif preprocessor == "blank-treatment":
            from src.llm_text_extract import extract_data_with_blank_treatment
            print("🔧 Running blank field treatment + LLM extraction...")
            extracted_data, call_id = extract_data_with_blank_treatment(
                str(pdf_path), model_name=model
            )
            df = pd.DataFrame(extracted_data) if extracted_data else pd.DataFrame()

        elif preprocessor == "table-focused":
            from src.llm_text_extract import extract_data_with_table_focused_preprocessing
            print("🎯 Running table-focused WHO surveillance extraction + LLM correction...")
            extracted_data, call_id = extract_data_with_table_focused_preprocessing(
                str(pdf_path), model_name=model, prompt_version=prompt_version
            )
            df = pd.DataFrame(extracted_data) if extracted_data else pd.DataFrame()

        elif preprocessor == "none-pdf-upload":
            from src.pdf_upload_extract import extract_data_with_pdf_upload
            print("📤 Running direct PDF upload extraction (no text preprocessing)...")
            extracted_data, call_id = extract_data_with_pdf_upload(
                str(pdf_path), model_name=model, prompt_version=prompt_version
            )
            df = pd.DataFrame(extracted_data) if extracted_data else pd.DataFrame()

        else:
            # Default: text-based extraction (no preprocessing)
            from src.llm_text_extract import process_pdf_with_text_extraction
            print("📄 Running text-based extraction...")
            output_csv_path = output_dir / f"extraction_{pdf_path.stem}.csv"
            df = process_pdf_with_text_extraction(
                pdf_path=str(pdf_path),
                output_csv_path=str(output_csv_path),
                model_name=model,
            )
            call_id = None  # Will get from parquet logs

        execution_time = time.time() - start_time

        print()
        print("=" * 60)

        # Check if extraction was successful
        if len(df) == 0:
            print("⚠️  WARNING: Extraction returned 0 records!")
            print("=" * 60)
            print("This usually means:")
            print("1. The LLM API call failed (check API key)")
            print("2. The PDF format was unreadable")
            print("3. The model couldn't extract data from the text")
            print()
            raise ValueError("Extraction failed: 0 records extracted")

        print("✅ Extraction Complete")
        print("=" * 60)
        print(f"Records extracted: {len(df)}")
        print(f"Execution time: {execution_time:.2f}s")
        print()

        # Get run_id from the most recent parquet log (or use call_id if available)
        run_id = None
        parquet_dir = Config.get_duckdb_logs_dir() / "prompt_logs"
        if parquet_dir.exists():
            parquet_files = sorted(parquet_dir.glob("run_*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
            if parquet_files:
                # Extract run_id from filename (run_<id>.parquet)
                run_id_str = parquet_files[0].stem.replace("run_", "")
                run_id = int(run_id_str)
            else:
                print("⚠️  WARNING: No parquet logs were created!")
                print("Check that LOG_BACKEND=duckdb is set correctly")

        # Fallback to call_id if we got one from the extraction function
        if run_id is None and call_id:
            run_id = int(call_id)

        if run_id is None:
            run_id = int(time.time())  # Last resort fallback

        # Save CSV to output directory with run_id
        model_clean = model.replace("/", "_").replace("-", "_")
        csv_filename = f"extraction_{run_id}_prompt_{prompt_version}_model_{model_clean}.csv"
        actual_csv_path = output_dir / csv_filename

        # Add SourceDocument column if not present
        if len(df) > 0 and "SourceDocument" not in df.columns:
            df["SourceDocument"] = pdf_path.name

        df.to_csv(actual_csv_path, index=False)
        print(f"💾 Saved CSV: {actual_csv_path.name}")

        return {
            'df': df,
            'records_extracted': len(df),
            'csv_path': actual_csv_path,
            'run_id': run_id,
            'execution_time': execution_time,
        }

    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    # Use the global parser (already defined at module level with all args)
    global parser
    args = parser.parse_args()

    # Get environment
    stage = os.getenv("STAGE", "dev")

    print("=" * 60)
    print("GitHub Actions Extraction Workflow")
    print("=" * 60)
    print(f"Stage: {stage}")
    print(f"Week: {args.week}")
    if args.year:
        print(f"Year: {args.year}")
    print()

    # Create temp directory for PDF
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Step 1: Get latest PDF from blob
        if args.week == "latest":
            pdf_info = get_latest_pdf_from_blob(stage=stage)
            if not pdf_info:
                print("❌ No PDF found in blob storage")
                sys.exit(1)

            blob_name = pdf_info['blob_name']
            filename = pdf_info['filename']
        else:
            # Construct blob name from week/year
            if not args.year:
                print("❌ --year is required when --week is specified")
                sys.exit(1)

            filename = f"OEW{int(args.week):02d}-{args.year}.pdf"
            blob_name = f"{Config.BLOB_PROJ_DIR}/raw/monitoring/{filename}"
            pdf_info = {
                'blob_name': blob_name,
                'filename': filename,
                'week': int(args.week),
                'year': args.year,
            }

        # Step 2: Download PDF
        local_pdf_path = temp_path / filename
        if not download_pdf_from_blob(blob_name, local_pdf_path, stage=stage):
            print("❌ Failed to download PDF")
            sys.exit(1)

        print()

        # Step 3: Run extraction
        extraction_result = run_extraction_pipeline(
            pdf_path=local_pdf_path,
            model=args.model,
            prompt_version=args.prompt_version,
            preprocessor=args.preprocessor,
            output_dir=temp_path,  # Use temp directory for CSV output
        )

        if not extraction_result:
            print("❌ Extraction failed")
            sys.exit(1)

        run_id = extraction_result['run_id']
        csv_path = extraction_result.get('csv_path')

        # Step 4: Upload logs to blob
        print("📤 Uploading logs to blob...")

        # Upload parquet logs from DuckDB logger
        parquet_dir = Config.get_duckdb_logs_dir()

        # Upload prompt logs
        prompt_logs_dir = parquet_dir / "prompt_logs"
        if prompt_logs_dir.exists():
            # Find the run file for this extraction
            run_file = prompt_logs_dir / f"run_{run_id}.parquet"
            if run_file.exists():
                upload_parquet_log_to_blob(
                    run_file,
                    log_type="prompt",
                    stage=stage
                )

        # Upload preprocessing logs if any
        preprocessing_logs_dir = parquet_dir / "tabular_preprocessing_logs"
        if preprocessing_logs_dir.exists():
            run_file = preprocessing_logs_dir / f"run_{run_id}.parquet"
            if run_file.exists():
                upload_parquet_log_to_blob(
                    run_file,
                    log_type="preprocessing",
                    stage=stage
                )

        # Step 5: Upload extracted CSV (if it exists)
        if csv_path and csv_path.exists():
            upload_csv_to_blob(
                csv_path=csv_path,
                pdf_name=filename,
                run_id=run_id,
                model=args.model,
                stage=stage
            )

        print()
        print("=" * 60)
        print("✅ Workflow Complete!")
        print("=" * 60)
        print(f"PDF: {filename}")
        print(f"Run ID: {run_id}")
        print(f"Records: {extraction_result['records_extracted']}")
        print(f"Time: {extraction_result['execution_time']:.2f}s")
        print()
        print("📊 Files uploaded to blob:")
        print(f"   Logs: {Config.BLOB_PROJ_DIR}/processed/logs/prompt_logs/run_{run_id}.parquet")
        print(f"   CSV:  {Config.BLOB_PROJ_DIR}/processed/llm_extractions/{Path(filename).stem}_{args.model.replace('/', '-')}_{run_id}.csv")
        print()
        print("💡 Query logs with:")
        print("   from src.cloud_logging import DuckDBCloudQuery")
        print("   query = DuckDBCloudQuery()")
        print("   df = query.get_latest_runs(n=5)")


if __name__ == "__main__":
    main()
