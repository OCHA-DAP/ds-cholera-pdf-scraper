#!/usr/bin/env python3
"""
Test script to verify CSV reading and PDF URL extraction.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

from download_historical_pdfs import HistoricalPDFDownloader


def test_csv_reading():
    """Test reading the CSV and extracting PDF URLs."""
    csv_url = "https://github.com/CBPFGMS/pfbi-data/raw/main/who_download_log.csv"

    downloader = HistoricalPDFDownloader(
        csv_url=csv_url, local_download_dir=Path("./test_downloads"), stage="dev"
    )

    print("Testing CSV reading...")

    # Test metadata loading
    metadata_df = downloader.get_pdf_metadata()
    print(f"Loaded metadata for {len(metadata_df)} PDFs")

    if len(metadata_df) > 0:
        print(f"Columns: {list(metadata_df.columns)}")
        print(f"First few rows:")
        print(metadata_df.head())

    # Test URL extraction
    pdf_urls = downloader.discover_pdf_urls()
    print(f"\nFound {len(pdf_urls)} PDF URLs")

    if pdf_urls:
        print(f"First 5 URLs:")
        for i, url in enumerate(pdf_urls[:5], 1):
            print(f"  {i}. {url}")

    # Test filename generation
    if pdf_urls and len(metadata_df) > 0:
        test_url = pdf_urls[0]
        filename = downloader.get_filename_from_metadata(test_url, metadata_df)
        print(f"\nTest filename generation:")
        print(f"URL: {test_url}")
        print(f"Generated filename: {filename}")


if __name__ == "__main__":
    test_csv_reading()
