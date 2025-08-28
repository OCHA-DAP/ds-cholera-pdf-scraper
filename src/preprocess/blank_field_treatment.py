"""
Simple blank field treatment processor for WHO health surveillance PDFs.
Identifies blank tabular fields and standardizes them with "-" characters.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pdfplumber


def extract_text_with_blank_treatment(pdf_path: str) -> str:
    """
    Extract text from PDF and apply blank field treatment.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Processed text with standardized blank fields
    """
    print(f"🔍 Extracting text with blank field treatment: {pdf_path}")

    raw_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        print(f"PDF has {len(pdf.pages)} pages")

        for page_num, page in enumerate(pdf.pages, 1):
            page_text = page.extract_text()
            if page_text:
                raw_text += f"\n--- PAGE {page_num} ---\n"
                raw_text += page_text

    print(f"Total raw text extracted: {len(raw_text)} characters")

    # Apply blank field treatment
    treated_text = apply_blank_field_treatment(raw_text)

    print(f"✅ Blank field treatment complete: {len(treated_text)} characters")
    return treated_text


def apply_blank_field_treatment(text: str) -> tuple[str, int]:
    """
    Apply blank field treatment to standardize various blank field indicators.

    Args:
        text: Raw text to process

    Returns:
        Tuple of (processed_text, number_of_treatments_applied)
    """
    lines = text.split("\n")
    total_treatments = 0
    processed_lines = []

    for line in lines:
        processed_line, line_treatments = standardize_blank_fields_in_line(line)
        processed_lines.append(processed_line)
        total_treatments += line_treatments

    return "\n".join(processed_lines), total_treatments


def extract_text_with_table_aware_blanks(pdf_path: str) -> tuple[str, int]:
    """
    Extract text from PDF with table-aware blank field treatment.

    This function extracts both text and table data, then merges them
    to ensure empty table cells are properly represented as "-".

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Tuple of (processed_text, number_of_treatments_applied)
    """
    print(f"🔍 EXTRACTING TEXT WITH TABLE-AWARE BLANK TREATMENT")
    print(f"📄 Processing: {pdf_path}")

    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        full_text_parts = []
        total_treatments = 0

        for page_num, page in enumerate(pdf.pages, 1):
            print(f"📄 Processing page {page_num}")

            # Extract regular text
            page_text = page.extract_text() or ""

            # Extract tables and identify empty cells
            tables = page.extract_tables()
            if tables:
                print(f"   Found {len(tables)} tables on page {page_num}")

                for table_idx, table in enumerate(tables):
                    if table:
                        print(f"   📊 Processing table {table_idx + 1}")

                        # Process each row in the table
                        for row_idx, row in enumerate(table):
                            if row:
                                # Convert empty cells to "-"
                                processed_row = []
                                row_treatments = 0

                                for cell in row:
                                    if (
                                        cell is None
                                        or cell == ""
                                        or (
                                            isinstance(cell, str) and cell.strip() == ""
                                        )
                                    ):
                                        processed_row.append("-")
                                        row_treatments += 1
                                    else:
                                        processed_row.append(str(cell))

                                if row_treatments > 0:
                                    print(
                                        f"      Row {row_idx}: Applied {row_treatments} blank treatments"
                                    )
                                    total_treatments += row_treatments

                                    # Replace the original row text in page_text if possible
                                    original_row_text = " ".join(
                                        [
                                            str(cell) if cell is not None else ""
                                            for cell in row
                                        ]
                                    )
                                    processed_row_text = " ".join(processed_row)

                                    if original_row_text.strip() in page_text:
                                        page_text = page_text.replace(
                                            original_row_text, processed_row_text
                                        )

            # Apply standard text-based blank field treatment
            processed_page_text, text_treatments = apply_blank_field_treatment(
                page_text
            )
            total_treatments += text_treatments

            full_text_parts.append(processed_page_text)

        print(f"✅ TABLE-AWARE EXTRACTION COMPLETE")
        print(f"📊 Total blank treatments applied: {total_treatments}")

        return "\n\n".join(full_text_parts), total_treatments


def standardize_blank_fields_in_line(line: str) -> tuple[str, int]:
    """
    Standardize blank fields in a single line.

    Args:
        line: Single line of text

    Returns:
        Tuple of (processed_line, number_of_treatments_applied)
    """
    # Skip if line is too short or doesn't look like table data
    if len(line.strip()) < 10:
        return line, 0

    # Look for patterns that indicate this is a table row
    numbers = re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", line)
    if len(numbers) < 2:  # Not likely a data row
        return line, 0

    # Standardize various blank field indicators
    treated_line = line
    treatments_applied = 0

    # Replace various dash types with standard dash
    if re.search(r"–+", treated_line):
        treated_line = re.sub(r"–+", "-", treated_line)
        treatments_applied += 1
    if re.search(r"—+", treated_line):
        treated_line = re.sub(r"—+", "-", treated_line)
        treatments_applied += 1
    if re.search(r"---+", treated_line):
        treated_line = re.sub(r"---+", "-", treated_line)
        treatments_applied += 1

    # Replace N/A variants with standard dash
    if re.search(r"\bN/?A\b", treated_line, flags=re.IGNORECASE):
        treated_line = re.sub(r"\bN/?A\b", "-", treated_line, flags=re.IGNORECASE)
        treatments_applied += 1
    if re.search(r"\bnull\b", treated_line, flags=re.IGNORECASE):
        treated_line = re.sub(r"\bnull\b", "-", treated_line, flags=re.IGNORECASE)
        treatments_applied += 1

    # Handle multiple spaces that might indicate empty cells
    # Be conservative - only replace 4+ spaces with a single dash
    if re.search(r"\s{4,}", treated_line):
        treated_line = re.sub(r"\s{4,}", " - ", treated_line)
        treatments_applied += 1

    # Clean up any double dashes that might have been created
    if re.search(r"-\s*-+", treated_line):
        treated_line = re.sub(r"-\s*-+", "-", treated_line)
        treatments_applied += 1

    return treated_line, treatments_applied


def identify_table_sections(text_content: str) -> List[Dict]:
    """
    Identify sections that contain table data for targeted treatment.

    Args:
        text_content: Full PDF text

    Returns:
        List of table section metadata
    """
    lines = text_content.split("\n")

    # Look for table headers
    table_headers = [
        r"Country.*Event.*Grade",
        r"Event.*Date.*Cases.*Deaths",
        r"Total cases.*Confirmed.*Deaths.*CFR",
        r"Country\s+Event\s+WHO risk assessment",
    ]

    table_sections = []

    for i, line in enumerate(lines):
        line_clean = line.strip()

        for pattern in table_headers:
            if re.search(pattern, line_clean, re.IGNORECASE):
                # Found a table header
                section = {"start_line": i, "header": line_clean, "type": "data_table"}
                table_sections.append(section)
                break

    return table_sections


def process_with_blank_treatment(pdf_path: str) -> Dict:
    """
    Complete blank treatment processing for a PDF.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Processing result dictionary
    """
    try:
        # Extract text with blank treatment
        treated_text = extract_text_with_blank_treatment(pdf_path)

        # Identify table sections for statistics
        table_sections = identify_table_sections(treated_text)

        # Count blank field treatments
        original_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    original_text += page_text

        # Count treatments applied
        original_lines = original_text.split("\n")
        treated_lines = treated_text.split("\n")

        treatments_applied = 0
        for orig, treated in zip(original_lines, treated_lines):
            if orig != treated:
                treatments_applied += 1

        result = {
            "success": True,
            "treated_text": treated_text,
            "metadata": {
                "pdf_path": pdf_path,
                "total_lines": len(treated_lines),
                "table_sections_found": len(table_sections),
                "blank_treatments_applied": treatments_applied,
                "original_text_length": len(original_text),
                "treated_text_length": len(treated_text),
            },
            "table_sections": table_sections,
        }

        print(f"✅ Blank treatment processing complete:")
        print(f"   Table sections found: {len(table_sections)}")
        print(f"   Blank treatments applied: {treatments_applied}")
        print(f"   Text length: {len(original_text)} → {len(treated_text)} chars")

        return result

    except Exception as e:
        print(f"❌ Blank treatment processing failed: {e}")
        return {"success": False, "error": str(e), "treated_text": None, "metadata": {}}


def preview_blank_treatments(pdf_path: str, max_examples: int = 10) -> None:
    """
    Preview what blank treatments would be applied without processing the full text.

    Args:
        pdf_path: Path to PDF file
        max_examples: Maximum number of examples to show
    """
    print(f"🔍 Preview of blank treatments for: {pdf_path}")

    with pdfplumber.open(pdf_path) as pdf:
        examples_found = 0

        for page_num, page in enumerate(pdf.pages, 1):
            if examples_found >= max_examples:
                break

            page_text = page.extract_text()
            if not page_text:
                continue

            lines = page_text.split("\n")

            for line_num, line in enumerate(lines):
                if examples_found >= max_examples:
                    break

                original_line = line
                treated_line = standardize_blank_fields_in_line(line)

                if treated_line != original_line:
                    examples_found += 1
                    print(
                        f"\n📝 Example {examples_found} (Page {page_num}, Line {line_num}):"
                    )
                    print(f"   Before: {original_line[:80]}...")
                    print(f"   After:  {treated_line[:80]}...")

        if examples_found == 0:
            print("ℹ️ No blank field treatments would be applied to this PDF")
        else:
            print(f"\n✅ Found {examples_found} examples of blank field treatments")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python blank_field_treatment.py <pdf_path> [preview]")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if len(sys.argv) > 2 and sys.argv[2] == "preview":
        # Preview mode
        preview_blank_treatments(pdf_path)
    else:
        # Full processing
        result = process_with_blank_treatment(pdf_path)

        if result["success"]:
            print("\n" + "=" * 60)
            print("BLANK TREATMENT SUMMARY")
            print("=" * 60)
            metadata = result["metadata"]
            for key, value in metadata.items():
                print(f"{key}: {value}")
        else:
            print(f"❌ Processing failed: {result['error']}")
