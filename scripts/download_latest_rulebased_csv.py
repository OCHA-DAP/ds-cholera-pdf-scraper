#!/usr/bin/env python3
"""
Download the latest rule-based CSV data from the PFBI repository.

This script downloads the final_data_for_powerbi_with_kpi.csv file from
the CBPFGMS/pfbi-data repository and overwrites the local copy in data/.
"""

import requests
from pathlib import Path


def download_latest_csv():
    """Download the latest rule-based CSV file."""
    url = "https://raw.githubusercontent.com/CBPFGMS/pfbi-data/main/final_data_for_powerbi_with_kpi.csv"
    output_path = Path(__file__).parent.parent / "data" / "final_data_for_powerbi_with_kpi.csv"

    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")

    # Download the file
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    # Write to file
    output_path.write_bytes(response.content)

    print(f"✓ Successfully downloaded {len(response.content):,} bytes")


if __name__ == "__main__":
    download_latest_csv()
