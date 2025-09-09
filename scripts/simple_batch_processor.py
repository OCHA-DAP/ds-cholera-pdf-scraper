#!/usr/bin/env python3
"""
Simple batch processing script using existing WHOSurveillanceExtractor.
Just loops through PDFs and combines results into master CSV.
Includes error handling and warning suppression for robust processing.
"""

import time
import warnings
import logging
from pathlib import Path
import pandas as pd

from src.config import Config
from src.pre_extraction.who_surveillance_extractor import WHOSurveillanceExtractor

# Suppress the harmless pdfminer color warnings
warnings.filterwarnings("ignore", message=".*Cannot set gray.*invalid float value.*")

# Set up logging to show only important messages
logging.getLogger("pdfminer").setLevel(logging.ERROR)  # Suppress pdfminer warnings


def process_pdf_safely(pdf_path: Path, extractor: WHOSurveillanceExtractor, individual_output_dir: Path) -> tuple:
    """
    Process a single PDF with error handling and save individual CSV.
    
    Args:
        pdf_path: Path to PDF file
        extractor: WHO surveillance extractor instance
        individual_output_dir: Directory to save individual CSV files
    
    Returns:
        tuple: (success: bool, records_count: int, csv_path: str or error_message: str)
    """
    try:
        df = extractor.extract_from_pdf(str(pdf_path), verbose=False)
        
        if not df.empty:
            # Save individual CSV
            csv_filename = f"{pdf_path.stem}.csv"
            csv_path = individual_output_dir / csv_filename
            df.to_csv(csv_path, index=False)
            return True, len(df), str(csv_path)
        else:
            return True, 0, "No records found"
            
    except KeyboardInterrupt:
        # Re-raise keyboard interrupt so user can still stop the process
        raise
    except Exception as e:
        error_msg = str(e)
        # Check if it's a serious error or just PDF corruption
        if any(keyword in error_msg.lower() for keyword in ['memory', 'disk', 'permission']):
            return False, 0, f"SERIOUS: {error_msg}"
        else:
            return False, 0, f"PDF issue: {error_msg}"


def combine_individual_csvs(individual_output_dir: Path, master_output_path: Path) -> pd.DataFrame:
    """
    Combine all individual CSV files into a master DataFrame.
    
    Args:
        individual_output_dir: Directory containing individual CSV files
        master_output_path: Path for the master CSV file
        
    Returns:
        Combined DataFrame
    """
    csv_files = list(individual_output_dir.glob("*.csv"))
    
    if not csv_files:
        print("❌ No individual CSV files found to combine")
        return pd.DataFrame()
    
    print(f"🔗 Combining {len(csv_files)} individual CSV files...")
    
    all_dataframes = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if not df.empty:
                all_dataframes.append(df)
        except Exception as e:
            print(f"⚠️  Error reading {csv_file.name}: {e}")
    
    if all_dataframes:
        master_df = pd.concat(all_dataframes, ignore_index=True)
        master_df.to_csv(master_output_path, index=False)
        print(f"✅ Master CSV saved: {master_output_path}")
        return master_df
    else:
        print("❌ No valid data found in individual CSV files")
        return pd.DataFrame()


def main(clobber: bool = False):
    """Robust batch processing with individual CSV outputs.
    
    Args:
        clobber: If True, remove existing CSV files and reprocess all PDFs
    """
    
    # Setup directories
    pdf_directory = Config.HISTORICAL_PDFS_DIR
    individual_output_dir = Config.OUTPUTS_DIR / "preprocessing_master" / "individual_csv"
    master_output_path = Config.OUTPUTS_DIR / "preprocessing_master" / "preprocessing_master_table.csv"
    
    # Handle clobber option
    if clobber:
        print(f"🗑️  CLOBBER MODE: Removing existing CSV files for clean reprocessing...")
        if individual_output_dir.exists():
            import shutil
            shutil.rmtree(individual_output_dir)
            print(f"✅ Removed {individual_output_dir}")
        if master_output_path.exists():
            master_output_path.unlink()
            print(f"✅ Removed {master_output_path}")
        print(f"🔄 Ready for clean reprocessing with improved fuzzy header matching!\n")
    
    # Create output directories
    individual_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 PDF directory: {pdf_directory}")
    print(f"� Individual CSVs: {individual_output_dir}")
    print(f"📊 Master CSV: {master_output_path}")
    print(f"🔇 Color warnings suppressed for cleaner output")
    print()
    
    # Find PDFs
    pdf_files = list(pdf_directory.glob("*.pdf"))
    pdf_files = [f for f in pdf_files if f.is_file() and f.stat().st_size > 0]
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    if not pdf_files:
        print("❌ No PDF files found")
        return
    
    # Check for existing individual CSVs (for resume capability)
    existing_csvs = set(csv.stem for csv in individual_output_dir.glob("*.csv"))
    pdfs_to_process = [pdf for pdf in pdf_files if pdf.stem not in existing_csvs]
    
    if len(pdfs_to_process) < len(pdf_files):
        skipped = len(pdf_files) - len(pdfs_to_process)
        print(f"🔄 Found {skipped} existing CSV files, processing remaining {len(pdfs_to_process)} PDFs")
    
    # Initialize extractor
    extractor = WHOSurveillanceExtractor()
    
    # Process PDFs (only those not already processed)
    successful_count = 0
    error_count = 0
    error_details = []
    start_time = time.time()
    
    for i, pdf_path in enumerate(pdfs_to_process, 1):
        total_progress = len(pdf_files) - len(pdfs_to_process) + i  # Account for skipped files
        print(f"[{total_progress:3d}/{len(pdf_files)}] {pdf_path.name}", end=" ... ")
        
        success, record_count, result = process_pdf_safely(pdf_path, extractor, individual_output_dir)
        
        if success and record_count > 0:
            successful_count += 1
            print(f"✅ {record_count} records → {Path(result).name}")
        elif success and record_count == 0:
            print(f"⚠️  No records found")
        else:
            error_count += 1
            error_details.append(f"{pdf_path.name}: {result}")
            print(f"❌ {result}")
        
        # Progress update every 25 files
        if i % 25 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = (len(pdfs_to_process) - i) * avg_time
            success_rate = successful_count / i * 100
            print(f"   📊 Progress: {success_rate:.1f}% success rate, ~{remaining/60:.1f} min remaining")
    
    # Combine all individual CSVs into master file
    total_time = time.time() - start_time
    master_df = combine_individual_csvs(individual_output_dir, master_output_path)
    
    # Final summary
    total_processed = successful_count + error_count
    print(f"\n{'='*60}")
    print(f"🏁 BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"📂 Individual CSVs: {successful_count} files in {individual_output_dir}")
    print(f"📊 Master CSV records: {len(master_df)} total records")
    print(f"✅ Success rate: {successful_count}/{len(pdf_files)} ({successful_count/len(pdf_files)*100:.1f}%)")
    
    if error_count > 0:
        print(f"\n❌ {error_count} errors encountered:")
        for error in error_details[:10]:  # Show first 10 errors
            print(f"   • {error}")
        if len(error_details) > 10:
            print(f"   ... and {len(error_details) - 10} more errors")
    
    print(f"\n💾 Files saved:")
    print(f"   📁 Individual CSVs: {individual_output_dir}")
    print(f"   📊 Master CSV: {master_output_path}")
    print("✨ Processing complete!")


if __name__ == "__main__":
    import sys
    
    # Check for clobber argument
    clobber = "--clobber" in sys.argv or "-c" in sys.argv
    
    if clobber:
        print("🧪 Running with CLOBBER mode - will reprocess all PDFs with improved fuzzy header matching")
    
    main(clobber=clobber)
