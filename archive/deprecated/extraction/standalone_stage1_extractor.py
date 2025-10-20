#!/usr/bin/env python3
"""
Standalone Stage 1 extractor for WHO AFRO surveillance bulletins.
Properly handles table extraction and narrative text association.
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import pdfplumber

AFRICAN_COUNTRIES = {
    "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cameroon",
    "Cape Verde", "Central African Republic", "Chad", "Comoros", "Congo",
    "Democratic Republic of the Congo", "Côte d'Ivoire", "Djibouti", "Egypt", 
    "Equatorial Guinea", "Eritrea", "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea", 
    "Guinea-Bissau", "Kenya", "Lesotho", "Liberia", "Libya", "Madagascar", "Malawi",
    "Mali", "Mauritania", "Mauritius", "Morocco", "Mozambique", "Namibia", "Niger",
    "Nigeria", "Rwanda", "Sao Tome and Principe", "Senegal", "Seychelles",
    "Sierra Leone", "Somalia", "South Africa", "South Sudan", "Sudan", "Eswatini",
    "Tanzania", "United Republic of Tanzania", "Togo", "Tunisia", "Uganda", "Zambia", "Zimbabwe"
}

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', str(text).strip())

def is_country_name(text: str) -> bool:
    """Check if text is a valid country name."""
    text = clean_text(text)
    if not text or len(text) > 50:  # Too long to be a country name
        return False
    
    # Direct match
    if text in AFRICAN_COUNTRIES:
        return True
    
    # Partial match for variations like "Tanzania, United Republic of"
    return any(country.lower() in text.lower() or text.lower() in country.lower() 
               for country in AFRICAN_COUNTRIES)

def is_valid_table_row(row: List[str]) -> bool:
    """Validate if a row is a real table row vs narrative text."""
    if not row or len(row) < 3:
        return False
    
    country_field = clean_text(row[0])
    
    # Check if first field is a country name
    if not is_country_name(country_field):
        return False
    
    # Should have multiple non-empty fields
    non_empty_fields = sum(1 for cell in row if clean_text(cell))
    return non_empty_fields >= 3

def extract_narrative_from_text(page_text: str, country: str, event: str) -> str:
    """Extract narrative text for a specific record from page text."""
    if not page_text or not country:
        return ""
    
    lines = page_text.split('\n')
    
    # Look for lines that mention the country and contain descriptive text
    narrative_candidates = []
    for i, line in enumerate(lines):
        line_clean = clean_text(line)
        if (country.lower() in line_clean.lower() and 
            len(line_clean) > 50 and  # Long enough to be descriptive
            not any(header in line_clean.lower() for header in ['country', 'event', 'grade', 'total cases'])):
            
            # Take this line and potentially the next few lines
            narrative_block = [line_clean]
            for j in range(i + 1, min(i + 4, len(lines))):  # Up to 3 more lines
                next_line = clean_text(lines[j])
                if len(next_line) > 30 and not is_country_name(next_line.split()[0] if next_line.split() else ""):
                    narrative_block.append(next_line)
                else:
                    break
            
            candidate = ' '.join(narrative_block)
            if len(candidate) > len(narrative_candidates[0] if narrative_candidates else ""):
                narrative_candidates = [candidate]
    
    return narrative_candidates[0] if narrative_candidates else ""

def extract_surveillance_table(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract surveillance data from WHO bulletin PDF."""
    records = []
    table_found = False
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            page_text = page.extract_text()
            
            # Find the surveillance table
            if not table_found and page_text and "All events currently being monitored" in page_text:
                table_found = True
                print(f"📍 Found surveillance table on page {page_num}")
            
            if not table_found:
                continue
            
            # Extract tables using pdfplumber
            tables = page.extract_tables()
            
            for table in tables:
                if not table:
                    continue
                
                print(f"🔍 Processing table with {len(table)} rows on page {page_num}")
                
                for row_idx, row in enumerate(table):
                    if not row:
                        continue
                    
                    # Clean the row data
                    cleaned_row = [clean_text(cell) for cell in row]
                    
                    # Skip header rows
                    if any(header in cleaned_row[0].lower() for header in ['country', 'new events', 'ongoing events']):
                        print(f"⏭️  Skipping header row: {cleaned_row[0][:50]}")
                        continue
                    
                    # Validate if this is a real table row
                    if not is_valid_table_row(cleaned_row):
                        print(f"❌ Invalid row (not a country): {cleaned_row[0][:50]}")
                        continue
                    
                    # Extract fields
                    country = cleaned_row[0]
                    event = cleaned_row[1] if len(cleaned_row) > 1 else ""
                    grade = cleaned_row[2] if len(cleaned_row) > 2 else ""
                    date_notified = cleaned_row[3] if len(cleaned_row) > 3 else ""
                    start_period = cleaned_row[4] if len(cleaned_row) > 4 else ""
                    end_period = cleaned_row[5] if len(cleaned_row) > 5 else ""
                    total_cases = cleaned_row[6] if len(cleaned_row) > 6 else ""
                    cases_confirmed = cleaned_row[7] if len(cleaned_row) > 7 else ""
                    deaths = cleaned_row[8] if len(cleaned_row) > 8 else ""
                    cfr = cleaned_row[9] if len(cleaned_row) > 9 else ""
                    
                    # Extract narrative text
                    narrative = extract_narrative_from_text(page_text, country, event)
                    
                    # Create record
                    record = {
                        "Country": country,
                        "Event": event,
                        "Grade": grade,
                        "Date notified to WCO": date_notified,
                        "Start of reporting period": start_period,
                        "End of reporting period": end_period,
                        "Total cases": total_cases,
                        "Cases Confirmed": cases_confirmed,
                        "Deaths": deaths,
                        "CFR": cfr,
                        "PageNumber": page_num,
                        "NarrativeText": narrative[:300] + "..." if len(narrative) > 300 else narrative
                    }
                    
                    records.append(record)
                    print(f"✅ Extracted record: {country} - {event}")
    
    print(f"🎉 Extraction complete: {len(records)} records found")
    return records

def main():
    parser = argparse.ArgumentParser(description="Extract WHO surveillance bulletin data")
    parser.add_argument("--pdf", required=True, help="Path to WHO bulletin PDF")
    parser.add_argument("--output", default="result.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    print(f"🔍 Processing PDF: {pdf_path.name}")
    
    try:
        records = extract_surveillance_table(str(pdf_path))
        
        # Write results
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {output_path}")
        print(f"📊 Records extracted: {len(records)}")
        
        # Preview results
        if records:
            print("\n📋 Sample records:")
            for i, record in enumerate(records[:3]):
                print(f"  {i+1}. {record['Country']} - {record['Event']} ({record.get('Total cases', 'N/A')} cases)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()