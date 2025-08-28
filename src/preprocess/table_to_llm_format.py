"""
Convert extracted pdfplumber tables to format suitable for LLM processing.
This bridges the gap between the table extraction and LLM ingestion.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def load_extracted_tables(outputs_dir: str = "outputs") -> List[pd.DataFrame]:
    """Load all extracted table CSV files from the outputs directory."""
    outputs_path = Path(outputs_dir)
    table_files = list(outputs_path.glob("table_page_*.csv"))

    tables = []
    for file_path in sorted(table_files):
        try:
            df = pd.read_csv(file_path)
            # Skip tables that are mostly empty or malformed
            if len(df) > 2 and len(df.columns) > 3:
                tables.append(df)
                print(
                    f"📊 Loaded {file_path.name}: {len(df)} rows x {len(df.columns)} columns"
                )
        except Exception as e:
            print(f"⚠️  Failed to load {file_path.name}: {e}")

    return tables


def clean_surveillance_data(tables: List[pd.DataFrame]) -> pd.DataFrame:
    """Clean and standardize surveillance tables into a unified format."""

    all_records = []

    for table_idx, df in enumerate(tables):
        print(f"\n🔍 Processing table {table_idx + 1}...")

        # Skip tables that don't look like surveillance data
        if not has_surveillance_columns(df):
            print("   ⏭️  Skipping - doesn't match surveillance format")
            continue

        # Clean and standardize column names
        df_clean = standardize_columns(df.copy())

        # Extract valid surveillance records
        records = extract_surveillance_records(df_clean, table_idx + 1)
        all_records.extend(records)

        print(f"   ✅ Extracted {len(records)} valid records")

    if not all_records:
        print("⚠️  No valid surveillance records found")
        return pd.DataFrame()

    # Convert to DataFrame
    result_df = pd.DataFrame(all_records)

    # Clean and validate data types
    result_df = clean_data_types(result_df)

    print(
        f"\n🎯 Final result: {len(result_df)} records across {result_df['country'].nunique()} countries"
    )

    return result_df


def has_surveillance_columns(df: pd.DataFrame) -> bool:
    """Check if DataFrame has surveillance data columns."""
    required_patterns = ["country", "event", "cases", "deaths"]
    column_text = " ".join(df.columns).lower()

    matches = sum(1 for pattern in required_patterns if pattern in column_text)
    return matches >= 3


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names for easier processing."""

    # Common column mappings
    column_mappings = {
        "country": ["country", "location"],
        "event": ["event", "disease", "hazard"],
        "grade": ["grade", "level"],
        "date_notified": ["date notified", "notification date", "notified"],
        "start_date": ["start of reporting", "start date", "onset"],
        "end_date": ["end of reporting", "end date", "latest"],
        "total_cases": ["total cases", "cases", "suspected"],
        "confirmed_cases": ["cases confirmed", "confirmed", "lab confirmed"],
        "deaths": ["deaths", "fatalities"],
        "cfr": ["cfr", "case fatality", "mortality rate"],
    }

    # Create new column names mapping
    new_columns = {}
    for col in df.columns:
        col_lower = col.lower().replace("\n", " ").strip()

        # Find best match
        for standard_name, patterns in column_mappings.items():
            if any(pattern in col_lower for pattern in patterns):
                new_columns[col] = standard_name
                break
        else:
            # Keep original if no match
            new_columns[col] = col_lower.replace(" ", "_")

    df.columns = [new_columns.get(col, col) for col in df.columns]
    return df


def extract_surveillance_records(
    df: pd.DataFrame, table_num: int
) -> List[Dict[str, Any]]:
    """Extract valid surveillance records from a cleaned DataFrame."""

    records = []

    for idx, row in df.iterrows():
        # Skip header rows or section dividers
        if is_header_or_divider(row):
            continue

        # Extract basic surveillance data
        record = extract_basic_record(row, table_num, idx)

        if record and is_valid_record(record):
            records.append(record)

    return records


def is_header_or_divider(row: pd.Series) -> bool:
    """Check if row is a header, divider, or non-data row."""

    # Common divider patterns
    divider_patterns = [
        "new events",
        "ongoing events",
        "grade 1 events",
        "grade 2 events",
        "grade 3 events",
        "ungraded events",
        "protracted",
        "legend",
    ]

    first_cell = str(row.iloc[0]).lower().strip()

    # Check for divider patterns
    if any(pattern in first_cell for pattern in divider_patterns):
        return True

    # Check if it's mostly empty
    non_empty = sum(1 for cell in row if pd.notna(cell) and str(cell).strip())
    if non_empty < 3:
        return True

    return False


def extract_basic_record(
    row: pd.Series, table_num: int, row_idx: int
) -> Dict[str, Any]:
    """Extract basic surveillance record from a row."""

    try:
        record = {
            "table_source": f"table_{table_num}",
            "row_index": row_idx,
            "country": clean_text_field(row.get("country", "")),
            "event": clean_text_field(row.get("event", "")),
            "grade": clean_text_field(row.get("grade", "")),
            "date_notified": clean_date_field(row.get("date_notified", "")),
            "start_date": clean_date_field(row.get("start_date", "")),
            "end_date": clean_date_field(row.get("end_date", "")),
            "total_cases": clean_numeric_field(row.get("total_cases", "")),
            "confirmed_cases": clean_numeric_field(row.get("confirmed_cases", "")),
            "deaths": clean_numeric_field(row.get("deaths", "")),
            "cfr": clean_percentage_field(row.get("cfr", "")),
        }

        return record

    except Exception as e:
        print(f"   ⚠️  Error processing row {row_idx}: {e}")
        return None


def clean_text_field(value: Any) -> str:
    """Clean text field, handling various data types."""
    if pd.isna(value):
        return ""

    text = str(value).strip()

    # Remove common artifacts
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())  # Normalize whitespace

    return text


def clean_date_field(value: Any) -> str:
    """Clean date field."""
    text = clean_text_field(value)

    # Handle common date patterns
    if not text or text.lower() in ["", "none", "null", "-"]:
        return ""

    return text


def clean_numeric_field(value: Any) -> int:
    """Clean numeric field and convert to integer."""
    if pd.isna(value):
        return 0

    text = str(value).strip()

    # Handle empty or dash values
    if not text or text in ["-", "", "None"]:
        return 0

    # Remove common number formatting
    text = text.replace(",", "").replace(" ", "")

    # Extract number
    try:
        # Handle cases like "27,16" (which should be 2716)
        if "." in text:
            return int(float(text))
        else:
            return int(text)
    except (ValueError, TypeError):
        return 0


def clean_percentage_field(value: Any) -> float:
    """Clean percentage field and convert to float."""
    if pd.isna(value):
        return 0.0

    text = str(value).strip()

    if not text or text in ["-", "", "None"]:
        return 0.0

    # Remove percentage sign
    text = text.replace("%", "").strip()

    try:
        return float(text)
    except (ValueError, TypeError):
        return 0.0


def is_valid_record(record: Dict[str, Any]) -> bool:
    """Check if record contains valid surveillance data."""

    # Must have country and event
    if not record.get("country") or not record.get("event"):
        return False

    # Must have some numerical data
    numeric_fields = ["total_cases", "confirmed_cases", "deaths"]
    has_numbers = any(record.get(field, 0) > 0 for field in numeric_fields)

    return has_numbers


def clean_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate data types in the final DataFrame."""

    # Ensure numeric columns are proper types
    numeric_columns = ["total_cases", "confirmed_cases", "deaths", "cfr"]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Clean text columns
    text_columns = ["country", "event", "grade"]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df


def export_for_llm(
    df: pd.DataFrame, output_path: str = "outputs/surveillance_data_for_llm.json"
):
    """Export cleaned data in format suitable for LLM processing."""

    # Convert to records format
    records = df.to_dict("records")

    # Create structured output for LLM
    llm_data = {
        "metadata": {
            "source": "WHO Weekly Bulletin",
            "extraction_method": "pdfplumber",
            "total_records": len(records),
            "countries": sorted(df["country"].unique().tolist()),
            "events": sorted(df["event"].unique().tolist()),
        },
        "surveillance_data": records,
    }

    # Save to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(llm_data, f, indent=2, ensure_ascii=False)

    print(f"💾 Exported {len(records)} records to {output_path}")

    return llm_data


if __name__ == "__main__":
    # Load extracted tables
    print("🔍 Loading extracted tables...")
    tables = load_extracted_tables()

    if not tables:
        print("❌ No tables found to process")
        exit(1)

    # Clean and process surveillance data
    print("\n🧹 Cleaning surveillance data...")
    surveillance_df = clean_surveillance_data(tables)

    if surveillance_df.empty:
        print("❌ No valid surveillance data extracted")
        exit(1)

    # Export for LLM processing
    print("\n📤 Exporting for LLM processing...")
    llm_data = export_for_llm(surveillance_df)

    # Also save as CSV for inspection
    csv_path = "outputs/surveillance_data_cleaned.csv"
    surveillance_df.to_csv(csv_path, index=False)
    print(f"💾 Also saved CSV to {csv_path}")

    print("\n✅ Processing complete!")
    print(f"   • {len(surveillance_df)} surveillance records")
    print(f"   • {surveillance_df['country'].nunique()} countries")
    print(f"   • {surveillance_df['event'].nunique()} event types")
