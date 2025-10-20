"""
Reconstruct the main WHO surveillance table by treating it as a single logical table
that spans multiple pages with interspersed descriptive text.

The WHO bulletin contains ONE main surveillance table that is broken across pages 9-15,
with each country/event having both tabular data and descriptive text.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import pandas as pd
import pdfplumber


class SurveillanceTableReconstructor:
    """Reconstruct the main surveillance table from WHO bulletin."""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.surveillance_pages = list(
            range(8, 16)
        )  # Pages 9-16 in 1-indexed (8-15 in 0-indexed)

    def extract_surveillance_table(self) -> pd.DataFrame:
        """Extract the main surveillance table spanning multiple pages."""

        print("🔍 Extracting WHO surveillance table...")

        # Step 1: Extract all text and identify table patterns
        page_contents = self._extract_page_contents()

        # Step 2: Identify table headers and structure
        table_structure = self._identify_table_structure(page_contents)

        # Step 3: Extract surveillance records
        records = self._extract_surveillance_records(page_contents, table_structure)

        # Step 4: Clean and validate records
        cleaned_records = self._clean_records(records)

        # Step 5: Convert to DataFrame
        df = pd.DataFrame(cleaned_records)

        print(f"✅ Extracted {len(df)} surveillance records")

        return df

    def _extract_page_contents(self) -> List[Dict]:
        """Extract content from surveillance pages."""

        page_contents = []

        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num in self.surveillance_pages:
                if page_num < len(pdf.pages):
                    page = pdf.pages[page_num]

                    # Extract text with position information
                    text_objects = page.extract_words()

                    # Extract tables
                    tables = page.extract_tables()

                    page_content = {
                        "page_num": page_num + 1,  # Convert to 1-indexed
                        "text_objects": text_objects,
                        "tables": tables,
                        "full_text": page.extract_text(),
                    }

                    page_contents.append(page_content)
                    print(
                        f"📄 Processed page {page_num + 1}: {len(text_objects)} text objects, {len(tables)} tables"
                    )

        return page_contents

    def _identify_table_structure(self, page_contents: List[Dict]) -> Dict:
        """Identify the surveillance table structure and headers."""

        # Look for table headers in the first surveillance page
        headers = []

        for page_content in page_contents:
            if page_content["tables"]:
                for table in page_content["tables"]:
                    if table and len(table) > 0:
                        potential_headers = table[0]

                        # Check if this looks like surveillance headers
                        header_text = " ".join(
                            str(h) for h in potential_headers if h
                        ).lower()

                        if self._is_surveillance_header(header_text):
                            headers = [
                                self._clean_header(h) for h in potential_headers if h
                            ]
                            break

                if headers:
                    break

        # Default headers if not found
        if not headers:
            headers = [
                "Country",
                "Event",
                "Grade",
                "Date_Notified",
                "Start_Date",
                "End_Date",
                "Total_Cases",
                "Confirmed_Cases",
                "Deaths",
                "CFR",
            ]

        print(f"🏷️  Table headers: {headers}")

        return {"headers": headers, "num_columns": len(headers)}

    def _is_surveillance_header(self, header_text: str) -> bool:
        """Check if text looks like surveillance table headers."""

        surveillance_keywords = [
            "country",
            "event",
            "grade",
            "cases",
            "deaths",
            "cfr",
            "date",
            "reporting",
            "confirmed",
        ]

        matches = sum(1 for keyword in surveillance_keywords if keyword in header_text)
        return matches >= 4

    def _clean_header(self, header: str) -> str:
        """Clean and standardize header text."""
        if not header:
            return ""

        # Clean the header text
        clean = str(header).strip().replace("\n", " ")
        clean = re.sub(r"\s+", " ", clean)

        # Standardize common headers
        header_mapping = {
            "country": "Country",
            "event": "Event",
            "grade": "Grade",
            "date notified": "Date_Notified",
            "start of reporting": "Start_Date",
            "end of reporting": "End_Date",
            "total cases": "Total_Cases",
            "cases confirmed": "Confirmed_Cases",
            "deaths": "Deaths",
            "cfr": "CFR",
        }

        clean_lower = clean.lower()
        for pattern, standard in header_mapping.items():
            if pattern in clean_lower:
                return standard

        return clean.replace(" ", "_")

    def _extract_surveillance_records(
        self, page_contents: List[Dict], table_structure: Dict
    ) -> List[Dict]:
        """Extract surveillance records from all pages."""

        records = []

        for page_content in page_contents:
            page_records = self._extract_records_from_page(
                page_content, table_structure
            )
            records.extend(page_records)

        return records

    def _extract_records_from_page(
        self, page_content: Dict, table_structure: Dict
    ) -> List[Dict]:
        """Extract records from a single page."""

        records = []

        # Method 1: Try to extract from structured tables
        table_records = self._extract_from_tables(
            page_content["tables"], table_structure
        )
        records.extend(table_records)

        # Method 2: Extract from text patterns (for cases where table structure is broken)
        text_records = self._extract_from_text_patterns(
            page_content["full_text"], table_structure
        )
        records.extend(text_records)

        return records

    def _extract_from_tables(self, tables: List, table_structure: Dict) -> List[Dict]:
        """Extract records from structured table data."""

        records = []
        headers = table_structure["headers"]

        for table in tables:
            if not table or len(table) < 2:
                continue

            # Skip header row and process data rows
            for row in table[1:]:
                if self._is_valid_data_row(row):
                    record = self._create_record(row, headers)
                    if record:
                        records.append(record)

        return records

    def _extract_from_text_patterns(
        self, text: str, table_structure: Dict
    ) -> List[Dict]:
        """Extract records from text using pattern matching."""

        records = []

        # Pattern for surveillance entries: Country + Disease + numerical data
        pattern = r"([A-Z][a-zA-Z\s,]+?)\s+(Cholera|Mpox|Measles|Malaria|Lassa Fever|Meningitis|Diphtheria|Anthrax|Yellow Fever|Dengue|Chikungunya|Poliomyelitis|SARS-CoV-2|Marburg|Plague|West Nile|Hepatitis|Food Poisoning|Leishmaniasis|Crimean-Congo)\s+(Grade \d|Ungraded|Protracted \d?)\s+(\d{1,2}-\w{3}-\d{2})?.*?(\d{1,3}(?:,\d{3})*)\s+.*?(\d{1,3}(?:,\d{3})*)\s+.*?(\d{1,4})\s+.*?(\d{1,2}\.\d)%"

        matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)

        for match in matches:
            try:
                record = {
                    "Country": match.group(1).strip(),
                    "Event": match.group(2).strip(),
                    "Grade": match.group(3).strip(),
                    "Date_Notified": match.group(4) if match.group(4) else "",
                    "Total_Cases": self._clean_number(match.group(5)),
                    "Confirmed_Cases": self._clean_number(match.group(6)),
                    "Deaths": self._clean_number(match.group(7)),
                    "CFR": self._clean_percentage(match.group(8)),
                }

                if self._is_valid_record(record):
                    records.append(record)

            except Exception as e:
                print(f"⚠️  Error parsing text pattern: {e}")
                continue

        return records

    def _is_valid_data_row(self, row: List) -> bool:
        """Check if a table row contains valid surveillance data."""

        if not row or len(row) < 3:
            return False

        # First column should be a country name
        first_col = str(row[0]).strip()
        if not first_col or len(first_col) < 2:
            return False

        # Should contain some numerical data
        has_numbers = any(self._contains_number(str(cell)) for cell in row)

        return has_numbers

    def _contains_number(self, text: str) -> bool:
        """Check if text contains numerical data."""
        return bool(re.search(r"\d", text))

    def _create_record(self, row: List, headers: List[str]) -> Optional[Dict]:
        """Create a record dictionary from table row."""

        try:
            record = {}

            for i, header in enumerate(headers):
                value = row[i] if i < len(row) else ""
                record[header] = self._clean_cell_value(value, header)

            return record if self._is_valid_record(record) else None

        except Exception as e:
            print(f"⚠️  Error creating record: {e}")
            return None

    def _clean_cell_value(self, value, header: str) -> str:
        """Clean individual cell value based on header type."""

        if not value:
            return ""

        text = str(value).strip()

        # Handle numeric fields
        if any(field in header for field in ["Cases", "Deaths"]):
            return str(self._clean_number(text))
        elif "CFR" in header:
            return str(self._clean_percentage(text))
        else:
            return text

    def _clean_number(self, text: str) -> int:
        """Clean and convert text to number."""
        if not text:
            return 0

        # Remove common formatting
        clean = re.sub(r"[^\d,.]", "", str(text))
        clean = clean.replace(",", "")

        try:
            if "." in clean:
                return int(float(clean))
            else:
                return int(clean) if clean else 0
        except (ValueError, TypeError):
            return 0

    def _clean_percentage(self, text: str) -> float:
        """Clean and convert text to percentage."""
        if not text:
            return 0.0

        clean = re.sub(r"[^\d.]", "", str(text))

        try:
            return float(clean) if clean else 0.0
        except (ValueError, TypeError):
            return 0.0

    def _is_valid_record(self, record: Dict) -> bool:
        """Check if record contains basic surveillance data structure."""

        # Only basic validation - let LLM handle data quality filtering
        if not record.get("Country") or not record.get("Event"):
            return False

        # Must have some numerical data
        numeric_fields = ["Total_Cases", "Confirmed_Cases", "Deaths"]
        has_numbers = any(
            isinstance(record.get(field), (int, str))
            and str(record.get(field, 0)).replace(",", "").isdigit()
            and int(str(record.get(field, 0)).replace(",", "")) > 0
            for field in numeric_fields
        )

        return has_numbers

    def _clean_records(self, records: List[Dict]) -> List[Dict]:
        """Clean and deduplicate records."""

        cleaned = []
        seen = set()

        for record in records:
            # Create a key for deduplication
            key = f"{record.get('Country', '')}-{record.get('Event', '')}-{record.get('Grade', '')}"

            if key not in seen and self._is_valid_record(record):
                seen.add(key)
                cleaned.append(record)

        return cleaned


def extract_who_surveillance_data(pdf_path: str) -> pd.DataFrame:
    """Main function to extract WHO surveillance data."""

    reconstructor = SurveillanceTableReconstructor(pdf_path)
    return reconstructor.extract_surveillance_table()


if __name__ == "__main__":
    # Test with the WHO bulletin
    from pathlib import Path

    from src.config import Config

    pdf_path = str(
        Path(Config.LOCAL_DIR_BASE)
        / "Cholera - General"
        / "WHO_bulletins_historical"
        / "Week_28__7_-_13_July_2025.pdf"
    )

    print("🔍 Reconstructing WHO surveillance table...")

    try:
        df = extract_who_surveillance_data(pdf_path)

        if not df.empty:
            print(f"\n✅ Successfully extracted {len(df)} surveillance records")
            print(f"📊 Countries: {df['Country'].nunique()}")
            print(f"🦠 Events: {df['Event'].nunique()}")

            # Save results
            output_path = "outputs/reconstructed_surveillance_table.csv"
            df.to_csv(output_path, index=False)
            print(f"💾 Saved to {output_path}")

            # Show sample
            print(f"\n📋 Sample records:")
            print(df.head())

        else:
            print("❌ No surveillance data extracted")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
