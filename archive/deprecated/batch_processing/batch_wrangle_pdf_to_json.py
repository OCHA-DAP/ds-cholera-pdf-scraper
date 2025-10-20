#!/usr/bin/env python3
"""
Batch processing script using the new wrangle_pdf_table_to_json module.
Processes WHO surveillance bulletins and combines results into master JSON and CSV files.
Includes error handling and progress tracking for robust processing.
"""

import time
import warnings
import logging
import json
from pathlib import Path
import pandas as pd

from src.config import Config
from src.preprocess.wrangle_pdf_table_to_json import extract_surveillance_table

# Suppress the harmless pdfminer color warnings
warnings.filterwarnings("ignore", message=".*Cannot set gray.*invalid float value.*")

# Set up logging to show only important messages
logging.getLogger("pdfminer").setLevel(logging.ERROR)  # Suppress pdfminer warnings


def process_pdf_safely(pdf_path: Path, individual_output_dir: Path) -> tuple:
    """
    Process a single PDF with error handling and save individual JSON/CSV.
    
    Args:
        pdf_path: Path to PDF file
        individual_output_dir: Directory to save individual output files
    
    Returns:
        tuple: (success: bool, records_count: int, result_info: str)
    """
    try:
        records = extract_surveillance_table(str(pdf_path), verbose=False)
        
        if records:
            # Save individual JSON
            json_filename = f"{pdf_path.stem}.json"
            json_path = individual_output_dir / json_filename
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            
            # Also save as CSV for easy viewing
            csv_filename = f"{pdf_path.stem}.csv"
            csv_path = individual_output_dir / csv_filename
            df = pd.DataFrame(records)
            df.to_csv(csv_path, index=False)
            
            return True, len(records), f"JSON: {json_path.name}, CSV: {csv_path.name}"
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


def combine_individual_files(individual_output_dir: Path, master_json_path: Path, master_csv_path: Path) -> tuple:
    """
    Combine all individual JSON files into master JSON and CSV files.
    
    Args:
        individual_output_dir: Directory containing individual JSON files
        master_json_path: Path for the master JSON file
        master_csv_path: Path for the master CSV file
        
    Returns:
        tuple: (total_records: int, combined_data: list)
    """
    json_files = list(individual_output_dir.glob("*.json"))
    
    if not json_files:
        print("❌ No individual JSON files found to combine")
        return 0, []
    
    print(f"🔗 Combining {len(json_files)} individual JSON files...")
    
    all_records = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                records = json.load(f)
                if records:
                    # Add source file info to each record
                    for record in records:
                        record['SourceFile'] = json_file.stem + '.pdf'
                    all_records.extend(records)
        except Exception as e:
            print(f"⚠️  Error reading {json_file.name}: {e}")
    
    if all_records:
        # Save master JSON
        with open(master_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_records, f, indent=2, ensure_ascii=False)
        print(f"✅ Master JSON saved: {master_json_path}")
        
        # Save master CSV
        master_df = pd.DataFrame(all_records)
        master_df.to_csv(master_csv_path, index=False)
        print(f"✅ Master CSV saved: {master_csv_path}")
        
        return len(all_records), all_records
    else:
        print("❌ No valid data found in individual JSON files")
        return 0, []


def main(clobber: bool = False):
    """Robust batch processing with individual JSON/CSV outputs.
    
    Args:
        clobber: If True, remove existing files and reprocess all PDFs
    """
    
    # Setup directories
    pdf_directory = Config.HISTORICAL_PDFS_DIR
    individual_output_dir = Config.OUTPUTS_DIR / "wrangle_batch" / "individual"
    master_json_path = Config.OUTPUTS_DIR / "wrangle_batch" / "master_surveillance_data.json"
    master_csv_path = Config.OUTPUTS_DIR / "wrangle_batch" / "master_surveillance_data.csv"
    
    # Handle clobber option
    if clobber:
        print(f"🗑️  CLOBBER MODE: Removing existing files for clean reprocessing...")
        if individual_output_dir.exists():
            import shutil
            shutil.rmtree(individual_output_dir)
            print(f"✅ Removed {individual_output_dir}")
        for master_file in [master_json_path, master_csv_path]:
            if master_file.exists():
                master_file.unlink()
                print(f"✅ Removed {master_file}")
        print(f"🔄 Ready for clean reprocessing!\n")
    
    # Create output directories
    individual_output_dir.mkdir(parents=True, exist_ok=True)
    master_json_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 PDF directory: {pdf_directory}")
    print(f"📂 Individual files: {individual_output_dir}")
    print(f"📊 Master JSON: {master_json_path}")
    print(f"📊 Master CSV: {master_csv_path}")
    print(f"🔇 Color warnings suppressed for cleaner output")
    print()
    
    # Find PDFs
    pdf_files = list(pdf_directory.glob("*.pdf"))
    pdf_files = [f for f in pdf_files if f.is_file() and f.stat().st_size > 0]
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    if not pdf_files:
        print("❌ No PDF files found")
        return
    
    # Check for existing individual JSON files (for resume capability)
    existing_jsons = set(json_file.stem for json_file in individual_output_dir.glob("*.json"))
    pdfs_to_process = [pdf for pdf in pdf_files if pdf.stem not in existing_jsons]
    
    if len(pdfs_to_process) < len(pdf_files):
        skipped = len(pdf_files) - len(pdfs_to_process)
        print(f"🔄 Found {skipped} existing JSON files, processing remaining {len(pdfs_to_process)} PDFs")
    
    # Process PDFs (only those not already processed)
    successful_count = 0
    error_count = 0
    total_records = 0
    error_details = []
    start_time = time.time()
    
    for i, pdf_path in enumerate(pdfs_to_process, 1):
        total_progress = len(pdf_files) - len(pdfs_to_process) + i  # Account for skipped files
        print(f"[{total_progress:3d}/{len(pdf_files)}] {pdf_path.name}", end=" ... ")
        
        success, record_count, result = process_pdf_safely(pdf_path, individual_output_dir)
        
        if success and record_count > 0:
            successful_count += 1
            total_records += record_count
            print(f"✅ {record_count} records → {result}")
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
    
    # Combine all individual files into master files
    total_time = time.time() - start_time
    final_records, combined_data = combine_individual_files(
        individual_output_dir, master_json_path, master_csv_path
    )
    
    # Final summary
    total_processed = successful_count + error_count
    print(f"\n{'='*60}")
    print(f"🏁 BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"📂 Individual files: {successful_count} JSON/CSV pairs in {individual_output_dir}")
    print(f"📊 Master file records: {final_records} total records")
    print(f"✅ Success rate: {successful_count}/{len(pdf_files)} ({successful_count/len(pdf_files)*100:.1f}%)")
    
    if error_count > 0:
        print(f"\n❌ {error_count} errors encountered:")
        for error in error_details[:10]:  # Show first 10 errors
            print(f"   • {error}")
        if len(error_details) > 10:
            print(f"   ... and {len(error_details) - 10} more errors")
    
    print(f"\n💾 Files saved:")
    print(f"   📁 Individual files: {individual_output_dir}")
    print(f"   📊 Master JSON: {master_json_path}")
    print(f"   📊 Master CSV: {master_csv_path}")
    
    # Show sample of extracted data
    if combined_data:
        print(f"\n📋 Sample records from master file:")
        for i, record in enumerate(combined_data[:5]):
            country = record.get('Country', 'Unknown')
            event = record.get('Event', 'Unknown')
            cases = record.get('Total cases', 'N/A')
            source = record.get('SourceFile', 'Unknown')
            print(f"  {i+1}. {country} - {event} ({cases} cases) from {source}")
    
    print("✨ Processing complete!")


if __name__ == "__main__":
    import sys
    
    # Check for clobber argument
    clobber = "--clobber" in sys.argv or "-c" in sys.argv
    
    if clobber:
        print("🧪 Running with CLOBBER mode - will reprocess all PDFs")
    
    main(clobber=clobber)