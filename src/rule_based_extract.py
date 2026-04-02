#!/usr/bin/env python3
"""
Rule-Based PDF Table Extraction Module

This module provides table extraction using Tabula (lattice detection) instead of LLMs.
Based on Kenny's approach but integrated with the existing codebase infrastructure.

Key differences from LLM extraction:
- Uses tabula-py for table detection (requires Java)
- Deterministic and fast
- No API costs
- Brittle to PDF format changes

Usage:
    from src.rule_based_extract import extract_table_from_pdf

    df = extract_table_from_pdf(
        pdf_path="/path/to/OEW42-2025.pdf",
        week=42,
        year=2025
    )
"""

import logging
import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import PyPDF2
from tabula.io import convert_into

logger = logging.getLogger(__name__)


# Column names for WHO outbreak surveillance tables
COLUMN_NAMES = [
    'Country', 'Event', 'Grade', 'Date notified to WCO',
    'Start of reporting period', 'End of reporting period',
    'Total cases', 'Cases Confirmed', 'Deaths', 'CFR'
]

# CSV cleaning replacements (remove newlines from headers)
CSV_REPLACEMENTS = {
    "Date notified\nto WCO": "Date notified to WCO",
    "Start of\nreporting\nperiod": "Start of reporting period",
    "End of\nreporting\nperiod": "End of reporting period",
    "Cases\nConfirmed": "Cases Confirmed"
}


def find_table_pages(pdf_path: Path) -> Tuple[int, int]:
    """
    Find the page range containing the outbreak surveillance table.

    Searches for "All events currently being monitored" text to identify
    the start of the table section.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Tuple of (start_page, end_page) as 1-indexed page numbers
    """
    try:
        pdf_reader = PyPDF2.PdfReader(str(pdf_path))
        total_pages = len(pdf_reader.pages)
        search_text = "All events currently being monitored"

        start_page = total_pages  # Default to last page if not found
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if re.search(search_text, text, re.IGNORECASE):
                start_page = i + 1  # 1-indexed
                logger.info(f"Found table start at page {start_page}")
                break

        return start_page, total_pages

    except Exception as e:
        logger.error(f"Error finding table pages: {e}")
        # Fallback: assume table starts at page 1
        return 1, 1


def extract_raw_table(pdf_path: Path, start_page: int, end_page: int) -> Optional[pd.DataFrame]:
    """
    Extract raw table data from PDF using Tabula.

    Args:
        pdf_path: Path to PDF file
        start_page: Starting page (1-indexed)
        end_page: Ending page (1-indexed)

    Returns:
        DataFrame with raw extracted data, or None if extraction fails
    """
    import tempfile

    try:
        logger.info(f"Extracting table from pages {start_page}-{end_page}")

        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            tmp_csv_path = tmp_file.name

        # Extract table using Tabula (lattice mode)
        page_range = f"{start_page}-{end_page}"
        convert_into(
            str(pdf_path),
            tmp_csv_path,
            output_format="csv",
            pages=page_range,
            lattice=True  # Use lattice detection for bordered tables
        )

        # Read the extracted CSV
        df = pd.read_csv(tmp_csv_path, encoding="ISO-8859-1")

        # Clean up temp file
        Path(tmp_csv_path).unlink(missing_ok=True)

        logger.info(f"Extracted {len(df)} raw rows")
        return df

    except Exception as e:
        logger.error(f"Error extracting table: {e}")
        return None


def clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column headers by removing newlines and normalizing names.

    Args:
        df: DataFrame with raw headers

    Returns:
        DataFrame with cleaned headers
    """
    # Apply replacements to column names
    cleaned_columns = []
    for col in df.columns:
        col_str = str(col)
        for old, new in CSV_REPLACEMENTS.items():
            col_str = col_str.replace(old, new)
        cleaned_columns.append(col_str)

    df.columns = cleaned_columns

    # Standardize to expected column names if possible
    # This handles cases where Tabula extracted correct data but with slight variations
    if len(df.columns) == len(COLUMN_NAMES):
        df.columns = COLUMN_NAMES

    return df


def clean_data(df: pd.DataFrame, week: int, year: int) -> pd.DataFrame:
    """
    Clean and transform the extracted data.

    Args:
        df: Raw DataFrame from Tabula
        week: Week number
        year: Year

    Returns:
        Cleaned DataFrame with metadata columns
    """
    # Remove completely empty rows
    df = df.dropna(how="all")

    # Remove rows without a grade (filter out non-data rows)
    if 'Grade' in df.columns:
        df = df[df['Grade'].notna()]

    # Clean text fields (remove newlines from country/event names)
    if 'Country' in df.columns:
        df['Country'] = df['Country'].str.replace("\n", " ", regex=False)
        df['Country'] = df['Country'].str.strip()

    if 'Event' in df.columns:
        df['Event'] = df['Event'].str.replace("\n", " ", regex=False)
        df['Event'] = df['Event'].str.strip()

    # Clean numeric fields
    if 'CFR' in df.columns:
        df['CFR'] = (df['CFR'].astype(str)
                     .str.replace("%", "", regex=False)
                     .str.replace(",", ".", regex=False)
                     .str.replace("-", "0", regex=False))

    if 'Cases Confirmed' in df.columns:
        df['Cases Confirmed'] = (df['Cases Confirmed'].astype(str)
                                  .str.replace("-", "0", regex=False)
                                  .str.replace(" ", "", regex=False)
                                  .str.replace(",", "", regex=False))

    if 'Total cases' in df.columns:
        df['Total cases'] = (df['Total cases'].astype(str)
                             .str.replace("-", "0", regex=False)
                             .str.replace(" ", "", regex=False)
                             .str.replace(",", "", regex=False))

    if 'Deaths' in df.columns:
        df['Deaths'] = (df['Deaths'].astype(str)
                        .str.replace("-", "0", regex=False)
                        .str.replace(" ", "", regex=False)
                        .str.replace(",", "", regex=False))

    # Add metadata columns
    df['WeekNumber'] = week
    df['Year'] = year
    df['Month'] = week  # Simple approximation (can be improved with date mapping)

    # Add source document
    df['ExtractionMethod'] = 'rule-based-tabula'

    logger.info(f"Cleaned data: {len(df)} rows")

    return df


def extract_table_from_pdf(
    pdf_path: str | Path,
    week: Optional[int] = None,
    year: Optional[int] = None,
) -> pd.DataFrame:
    """
    Main function to extract outbreak surveillance table from WHO PDF.

    This is the high-level API that orchestrates the extraction pipeline:
    1. Find table pages
    2. Extract raw table with Tabula
    3. Clean headers
    4. Clean data

    Args:
        pdf_path: Path to PDF file
        week: Week number (optional, for metadata)
        year: Year (optional, for metadata)

    Returns:
        Cleaned DataFrame with extracted outbreak data

    Raises:
        ValueError: If extraction fails or produces no data

    Example:
        >>> df = extract_table_from_pdf("OEW42-2025.pdf", week=42, year=2025)
        >>> print(f"Extracted {len(df)} outbreak records")
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info(f"Starting rule-based extraction: {pdf_path.name}")

    # Extract week/year from filename if not provided (format: OEW42-2025.pdf)
    if week is None or year is None:
        filename_match = re.match(r'OEW(\d+)-(\d{4})\.pdf', pdf_path.name)
        if filename_match:
            week = week or int(filename_match.group(1))
            year = year or int(filename_match.group(2))
            logger.info(f"Extracted metadata from filename: Week {week}, Year {year}")

    # Default values if still missing
    week = week or 1
    year = year or 2025

    # Step 1: Find table pages
    start_page, end_page = find_table_pages(pdf_path)

    # Step 2: Extract raw table
    df = extract_raw_table(pdf_path, start_page, end_page)

    if df is None or len(df) == 0:
        raise ValueError(f"Failed to extract any data from {pdf_path.name}")

    # Step 3: Clean headers
    df = clean_headers(df)

    # Step 4: Clean data
    df = clean_data(df, week, year)

    logger.info(f"Extraction complete: {len(df)} records from {pdf_path.name}")

    return df


def calculate_kpis(df: pd.DataFrame, week_date_csv: Optional[Path] = None) -> pd.DataFrame:
    """
    Calculate KPIs from extracted data (case changes, etc.).

    This is based on Kenny's calculateKPI.py logic.

    Args:
        df: DataFrame with extracted outbreak data
        week_date_csv: Optional path to weekDate.csv for date mapping

    Returns:
        DataFrame with additional KPI columns
    """
    # Create YearWeek column
    df['YearWeek'] = df['Year'].astype(str) + '-' + df['WeekNumber'].astype(str)

    # Merge with week date mapping if provided
    if week_date_csv and week_date_csv.exists():
        weekdate_df = pd.read_csv(week_date_csv)
        df = pd.merge(df, weekdate_df, on='YearWeek', how='left')

        # Clean and parse WeekDate
        if 'WeekDate' in df.columns:
            df['WeekDate'] = df['WeekDate'].str.replace(" ", "", regex=False)
            df['WeekDate'] = pd.to_datetime(df['WeekDate'], format='%m/%d/%Y', errors='coerce')

    # Convert Total cases to numeric
    if 'Total cases' in df.columns:
        df['Total cases'] = df['Total cases'].str.replace('[ ,]', '', regex=True)
        df['Total cases'] = pd.to_numeric(df['Total cases'], errors='coerce')

    # Sort by date and country
    sort_cols = ['WeekDate', 'Country', 'Event'] if 'WeekDate' in df.columns else ['Year', 'WeekNumber', 'Country', 'Event']
    df = df.sort_values(sort_cols, ascending=True)

    # Calculate case change (week-over-week difference by Country/Event)
    df['Case Change'] = df.groupby(['Country', 'Event'])['Total cases'].diff()

    # Trim whitespace from country names
    if 'Country' in df.columns:
        df['Country'] = df['Country'].str.strip()

    # Handle long country names (preserve important ones like DRC)
    long_country_mapping = {
        'Democratic Republic of the Congo': 'Democratic Republic of the Congo',
    }

    for long_country, new_name in long_country_mapping.items():
        df.loc[df['Country'] == long_country, 'Country'] = new_name

    # Apply fallback for other long country names (>30 chars)
    df.loc[
        (df['Country'].str.len() > 30) &
        (~df['Country'].isin(long_country_mapping.keys())),
        'Country'
    ] = "Cote d'Ivoire"

    logger.info(f"KPI calculation complete: {len(df)} records")

    return df
