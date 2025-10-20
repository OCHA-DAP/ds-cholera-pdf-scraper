#!/usr/bin/env python3
"""
Batch extraction script that processes the last 12 PDFs using LLM extraction.
Saves each extraction to its own CSV file in outputs/batch_run/.
"""

import sys
import time
from pathlib import Path

import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.config import Config
from src.pdf_upload_extract import extract_data_with_pdf_upload


def get_latest_weeks_to_process(pdf_dir: Path, count: int = 12) -> list[Path]:
    """
    Get the latest weeks from the CSV source that we have locally as PDFs.

    Args:
        pdf_dir: Directory containing PDFs
        count: Number of latest weeks to check

    Returns:
        List of PDF paths sorted by week (newest first), only for weeks we have locally
    """
    # Load the CSV source to get the latest weeks
    csv_url = "https://github.com/CBPFGMS/pfbi-data/raw/main/who_download_log.csv"
    print(f"📊 Loading latest weeks from CSV source...")

    try:
        df = pd.read_csv(csv_url)
        # Sort by Year and WeekNumber to get the latest weeks
        df_sorted = df.sort_values(["Year", "WeekNumber"], ascending=False)
        latest_weeks = df_sorted.head(count)

        print(f"   Found {len(latest_weeks)} latest weeks in CSV")

        # Check which ones we have locally
        available_pdfs = []
        missing_pdfs = []

        for _, row in latest_weeks.iterrows():
            filename = row["FileName"]
            pdf_path = pdf_dir / filename

            if pdf_path.exists() and pdf_path.stat().st_size > 0:
                available_pdfs.append(pdf_path)
            else:
                missing_pdfs.append(filename)

        if missing_pdfs:
            print(f"   ⚠️  Missing {len(missing_pdfs)} PDFs locally:")
            for missing in missing_pdfs[:5]:  # Show first 5
                print(f"      - {missing}")
            if len(missing_pdfs) > 5:
                print(f"      ... and {len(missing_pdfs) - 5} more")

        print(f"   ✅ Found {len(available_pdfs)} PDFs available locally")
        return available_pdfs

    except Exception as e:
        print(f"   ❌ Error loading CSV source: {e}")
        print(f"   🔄 Falling back to modification time method...")

        # Fallback to original method
        pdf_files = list(pdf_dir.glob("*.pdf"))
        pdf_files = [f for f in pdf_files if f.is_file() and f.stat().st_size > 0]
        pdf_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return pdf_files[:count]


def process_single_pdf(
    pdf_path: Path, model_name: str, prompt_version: str, output_dir: Path
) -> tuple[bool, int, str]:
    """
    Process a single PDF with direct PDF upload and save to CSV.

    Args:
        pdf_path: Path to PDF file
        model_name: Model to use for extraction
        prompt_version: Prompt version to use
        output_dir: Directory to save CSV output

    Returns:
        tuple: (success: bool, records_count: int, csv_path or error_message: str)
    """
    # Check if CSV already exists
    csv_filename = f"{pdf_path.stem}.csv"
    csv_path = output_dir / csv_filename

    if csv_path.exists():
        # Count existing records
        try:
            existing_df = pd.read_csv(csv_path)
            return (
                True,
                len(existing_df),
                f"Already processed (skipped) - {str(csv_path)}",
            )
        except Exception:
            return True, 0, f"Already processed (skipped) - {str(csv_path)}"

    try:
        print(f"  📤 Uploading PDF to {model_name}...")
        extracted_records, call_id = extract_data_with_pdf_upload(
            pdf_path=str(pdf_path),
            model_name=model_name,
            prompt_version=prompt_version,
        )

        if extracted_records:
            # Convert to DataFrame
            df = pd.DataFrame(extracted_records)

            # Save to CSV
            csv_filename = f"{pdf_path.stem}.csv"
            csv_path = output_dir / csv_filename
            df.to_csv(csv_path, index=False)

            return True, len(df), str(csv_path)
        else:
            return True, 0, "No records extracted"

    except KeyboardInterrupt:
        raise
    except Exception as e:
        return False, 0, f"Error: {str(e)}"


def main():
    """Run batch extraction on the last 12 PDFs."""

    # Configuration
    pdf_directory = Config.HISTORICAL_PDFS_DIR
    output_dir = Config.OUTPUTS_DIR / "batch_run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configuration - hardcoded values
    prompt_version = "v1.4.7"
    model_name = "gpt-5"

    print(f"📋 Using configuration:")
    print(f"   Prompt version: {prompt_version}")
    print(f"   Model: {model_name}")

    print(f"\n📁 PDF directory: {pdf_directory}")
    print(f"💾 Output directory: {output_dir}")
    print()

    # Get latest weeks that we have locally
    pdf_files = get_latest_weeks_to_process(pdf_directory, count=12)

    if not pdf_files:
        print("❌ No PDF files found")
        return

    print(f"📚 Found {len(pdf_files)} latest weeks available locally:")
    for i, pdf in enumerate(pdf_files, 1):
        mtime = time.strftime("%Y-%m-%d", time.localtime(pdf.stat().st_mtime))
        print(f"  {i:2d}. {pdf.name} (modified: {mtime})")
    print()

    # Process each PDF
    successful_count = 0
    error_count = 0
    total_records = 0
    start_time = time.time()

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i:2d}/{len(pdf_files)}] Processing: {pdf_path.name}")

        success, record_count, result = process_single_pdf(
            pdf_path=pdf_path,
            model_name=model_name,
            prompt_version=prompt_version,
            output_dir=output_dir,
        )

        if success and record_count > 0:
            if "skipped" in result.lower():
                print(f"  ⏭️  Already processed ({record_count} records)\n")
            else:
                successful_count += 1
                total_records += record_count
                csv_name = Path(result).name
                print(f"  ✅ {record_count} records → {csv_name}\n")
        elif success and record_count == 0:
            print(f"  ⚠️  No records extracted\n")
        else:
            error_count += 1
            print(f"  ❌ {result}\n")

    # Summary
    total_time = time.time() - start_time
    print(f"{'='*60}")
    print(f"🏁 BATCH EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"✅ Successful: {successful_count}/{len(pdf_files)}")
    print(f"📊 Total records extracted: {total_records}")
    print(f"💾 CSV files saved to: {output_dir}")

    if error_count > 0:
        print(f"❌ Errors: {error_count}")

    print("✨ Complete!")


if __name__ == "__main__":
    main()
