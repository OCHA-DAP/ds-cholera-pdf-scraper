#!/usr/bin/env python3
"""
PDF table wrangling module for WHO AFRO surveillance bulletins.
Properly handles table extraction and narrative text association.
"""

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

def standardize_country_names(country_name: str) -> str:
    """Standardize country names to fix common PDF extraction errors.
    
    Args:
        country_name: Raw country name from PDF extraction
        
    Returns:
        Standardized country name matching AFRICAN_COUNTRIES list
    """
    if not country_name:
        return country_name
    
    # Clean the input
    cleaned_name = clean_text(country_name)
    
    # Direct match - already correct
    if cleaned_name in AFRICAN_COUNTRIES:
        return cleaned_name
    
    # Common PDF extraction error patterns
    extraction_errors = {
        # Missing 'R' in Republic
        "Democratic epublic of the Congo": "Democratic Republic of the Congo",
        "Democratic epublic of Congo": "Democratic Republic of the Congo",
        "emocratic Republic of the Congo": "Democratic Republic of the Congo",
        "emocratic Republic of Congo": "Democratic Republic of the Congo",
        
        # Alternative names and variations
        "DRC": "Democratic Republic of the Congo",
        "DR Congo": "Democratic Republic of the Congo",
        "Congo DRC": "Democratic Republic of the Congo",
        "Congo, Democratic Republic": "Democratic Republic of the Congo",
        "Democratic Republic Congo": "Democratic Republic of the Congo",
        
        # Republic of Congo (Brazzaville) vs DRC
        "Congo Republic": "Congo",
        "Republic of Congo": "Congo",
        "Congo Brazzaville": "Congo",
        
        # Tanzania variations
        "Tanzania, United Republic of": "United Republic of Tanzania",
        "United Republic Tanzania": "United Republic of Tanzania",
        "Tanzania United Republic": "United Republic of Tanzania",
        
        # Côte d'Ivoire variations
        "Cote d'Ivoire": "Côte d'Ivoire",
        "Cote dIvoire": "Côte d'Ivoire",
        "Ivory Coast": "Côte d'Ivoire",
        "Côte d Ivoire": "Côte d'Ivoire",
        
        # Cape Verde variations
        "Cape Verde Islands": "Cape Verde",
        "Cabo Verde": "Cape Verde",
        
        # Sao Tome variations
        "Sao Tome & Principe": "Sao Tome and Principe",
        "São Tomé and Príncipe": "Sao Tome and Principe",
        "Sao Tome & Principe": "Sao Tome and Principe",
        "São Tomé & Príncipe": "Sao Tome and Principe",
        
        # Eswatini variations
        "Swaziland": "Eswatini",
        "Kingdom of Eswatini": "Eswatini",
        
        # Other common variations
        "Guinea Bissau": "Guinea-Bissau",
        "Central African Rep": "Central African Republic",
        "Central African Rep.": "Central African Republic",
        "CAR": "Central African Republic",
        
        # Truncation/OCR errors
        "Democratic Republic of th": "Democratic Republic of the Congo",
        "Democratic Republic of t": "Democratic Republic of the Congo",
        "emocratic Republic of th": "Democratic Republic of the Congo",
    }
    
    # Check direct mappings first
    if cleaned_name in extraction_errors:
        return extraction_errors[cleaned_name]
    
    # Case-insensitive matching for extraction errors
    for error_pattern, correct_name in extraction_errors.items():
        if cleaned_name.lower() == error_pattern.lower():
            return correct_name
    
    # Fuzzy matching for partial matches - look for country names within the text
    for standard_country in AFRICAN_COUNTRIES:
        # Check if the standard country name is contained in the extracted name
        if standard_country.lower() in cleaned_name.lower():
            return standard_country
        
        # Check if the extracted name is contained in the standard country name
        if cleaned_name.lower() in standard_country.lower() and len(cleaned_name) > 3:
            return standard_country
    
    # Handle cases where common words indicate specific countries
    fuzzy_patterns = {
        "congo": {
            "democratic": "Democratic Republic of the Congo",
            "republic": "Congo",
            "brazzaville": "Congo",
            "kinshasa": "Democratic Republic of the Congo",
            "default": "Democratic Republic of the Congo"  # Default to DRC if unclear
        },
        "tanzania": {
            "united": "United Republic of Tanzania",
            "republic": "United Republic of Tanzania",
            "default": "United Republic of Tanzania"
        }
    }
    
    lower_name = cleaned_name.lower()
    for key_word, patterns in fuzzy_patterns.items():
        if key_word in lower_name:
            for pattern_key, country_name in patterns.items():
                if pattern_key != "default" and pattern_key in lower_name:
                    return country_name
            # Return default if no specific pattern matched
            return patterns["default"]
    
    # If no standardization found, return original cleaned name
    return cleaned_name

def is_country_name(text: str) -> bool:
    """Check if text is a valid country name."""
    text = clean_text(text)
    if not text or len(text) > 50:  # Too long to be a country name
        return False
    
    # First try to standardize the name
    standardized_name = standardize_country_names(text)
    
    # Direct match with standardized name
    if standardized_name in AFRICAN_COUNTRIES:
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

def extract_narrative_from_text_by_position(page, country: str, event: str, table_row_bbox: dict = None) -> str:
    """Extract narrative text positioned directly below the table row using spatial coordinates."""
    if not page or not country:
        return ""
    
    try:
        # Try spatial approach first
        text_objects = page.extract_words()
        
        # Check if we have valid coordinates
        valid_coords = any(word.get('y0') is not None for word in text_objects)
        
        if valid_coords:
            # Use spatial extraction
            return extract_narrative_spatial(page, country, event, text_objects)
        else:
            # Coordinates not available, fall back to text-based
            return extract_narrative_improved_text(page.extract_text(), country, event)
            
    except Exception as e:
        # Fallback to simple text extraction if spatial method fails
        return extract_narrative_improved_text(page.extract_text() if hasattr(page, 'extract_text') else "", country, event)


def extract_narrative_spatial(page, country: str, event: str, text_objects: list) -> str:
    """Spatial-based narrative extraction using coordinates."""
    page_height = page.height
    
    # Group text objects by lines (similar y-coordinate)
    lines_with_y = {}
    for word_obj in text_objects:
        y_coord = word_obj.get('y0', 0)
        
        # Group words into lines (within 3 points y-tolerance)
        line_y = None
        for existing_y in lines_with_y.keys():
            if abs(y_coord - existing_y) <= 3:
                line_y = existing_y
                break
        
        if line_y is None:
            line_y = y_coord
            lines_with_y[line_y] = []
        
        lines_with_y[line_y].append(word_obj)
    
    # Sort lines by y-coordinate (top to bottom)
    sorted_lines = sorted(lines_with_y.items())
    
    # Reconstruct text lines
    text_lines = []
    for y_coord, words in sorted_lines:
        # Sort words by x-coordinate (left to right)
        sorted_words = sorted(words, key=lambda w: w.get('x0', 0))
        line_text = ' '.join(word['text'] for word in sorted_words)
        text_lines.append(clean_text(line_text))
    
    # Find narrative text that mentions this country
    return find_narrative_in_lines(text_lines, country, event)


def extract_narrative_improved_text(page_text: str, country: str, event: str) -> str:
    """Improved text-based narrative extraction."""
    if not page_text or not country:
        return ""
    
    lines = [clean_text(line) for line in page_text.split('\n')]
    return find_narrative_in_lines(lines, country, event)


def find_narrative_in_lines(text_lines: list, country: str, event: str) -> str:
    """Find narrative text in lines array - works for both spatial and text-based extraction."""
    country_lower = country.lower()
    event_lower = event.lower() if event else ""
    
    # First, try to find text that mentions both country AND the specific disease/event
    disease_specific_narrative = find_disease_specific_narrative(text_lines, country_lower, event_lower)
    if disease_specific_narrative:
        return disease_specific_narrative
    
    # Fallback: find general country narrative (but try to filter by event)
    return find_general_country_narrative(text_lines, country_lower, event_lower)


def find_disease_specific_narrative(text_lines: list, country_lower: str, event_lower: str) -> str:
    """Look for narrative text that mentions both country and specific disease."""
    if not event_lower:
        return ""
    
    # Build event-specific keywords to look for
    event_keywords = get_event_keywords(event_lower)
    
    narrative_lines = []
    
    # Look for lines that mention the country AND the specific event/disease
    for i, line in enumerate(text_lines):
        if not line or len(line) < 30:
            continue
        
        line_lower = line.lower()
        
        # Skip table headers
        if any(header in line_lower for header in [
            'country', 'event', 'grade', 'date notified', 'total cases',
            'start of reporting', 'end of reporting', 'cases confirmed',
            'ongoing events', 'new events', 'cfr', 'deaths'
        ]):
            continue
        
        # Check if this line mentions our country AND our specific disease
        if (country_lower in line_lower and 
            any(keyword in line_lower for keyword in event_keywords)):
            
            # This looks like disease-specific text
            narrative_lines.append(line)
            
            # Look ahead for continuation lines about this disease
            for j in range(i + 1, min(i + 5, len(text_lines))):
                next_line = text_lines[j]
                if not next_line or len(next_line) < 20:
                    continue
                    
                next_lower = next_line.lower()
                
                # Stop if we hit another country or different disease context
                if (is_country_name(next_line.split()[0] if next_line.split() else "") or
                    any(header in next_lower for header in ['country', 'event', 'grade'])):
                    break
                
                # Add continuation line if it seems related
                narrative_lines.append(next_line)
            
            break  # Found disease-specific narrative, stop looking
    
    return ' '.join(narrative_lines).strip()


def find_general_country_narrative(text_lines: list, country_lower: str, event_lower: str) -> str:
    """Find general country narrative as fallback."""
    narrative_lines = []
    
    for i, line in enumerate(text_lines):
        if not line or len(line) < 30:
            continue
        
        line_lower = line.lower()
        
        # Skip table headers
        if any(header in line_lower for header in [
            'country', 'event', 'grade', 'date notified', 'total cases',
            'start of reporting', 'end of reporting', 'cases confirmed',
            'ongoing events', 'new events', 'cfr', 'deaths'
        ]):
            continue
        
        # Check if this line mentions our country
        if country_lower in line_lower and not line_lower.strip() == country_lower:
            narrative_lines.append(line)
            
            # Look ahead for continuation lines
            for j in range(i + 1, min(i + 3, len(text_lines))):  # Shorter for general narrative
                next_line = text_lines[j]
                if not next_line or len(next_line) < 20:
                    continue
                    
                next_lower = next_line.lower()
                
                # Stop if we hit another country
                if (is_country_name(next_line.split()[0] if next_line.split() else "") or
                    any(header in next_lower for header in ['country', 'event', 'grade'])):
                    break
                
                narrative_lines.append(next_line)
            
            break  # Found our narrative, stop looking
    
    narrative = ' '.join(narrative_lines).strip()
    
    # Limit length
    if len(narrative) > 500:
        narrative = narrative[:497] + "..."
    
    return narrative


def get_event_keywords(event_lower: str) -> list:
    """Get keywords to identify specific disease/event mentions."""
    if not event_lower:
        return []
    
    # Base keyword is the event itself
    keywords = [event_lower]
    
    # Add disease-specific variations
    if 'covid' in event_lower:
        keywords.extend(['covid', 'coronavirus', 'sars-cov', 'covid-19'])
    elif 'cholera' in event_lower:
        keywords.extend(['cholera', 'vibrio cholerae'])
    elif 'measles' in event_lower:
        keywords.extend(['measles', 'rubeola'])
    elif 'polio' in event_lower:
        keywords.extend(['polio', 'poliovirus', 'cvdpv', 'wild poliovirus', 'vaccine-derived'])
    elif 'ebola' in event_lower:
        keywords.extend(['ebola', 'evd', 'ebola virus'])
    elif 'yellow fever' in event_lower:
        keywords.extend(['yellow fever', 'yf'])
    elif 'dengue' in event_lower:
        keywords.extend(['dengue'])
    elif 'lassa' in event_lower:
        keywords.extend(['lassa', 'lassa fever'])
    elif 'monkeypox' in event_lower or 'mpox' in event_lower:
        keywords.extend(['monkeypox', 'mpox', 'orthopoxvirus'])
    elif 'meningitis' in event_lower:
        keywords.extend(['meningitis', 'neisseria meningitidis'])
    elif 'hepatitis' in event_lower:
        keywords.extend(['hepatitis'])
    elif 'chikungunya' in event_lower:
        keywords.extend(['chikungunya'])
    
    return keywords


def extract_records_with_alternating_narrative(table: list, page_num: int, verbose: bool = True, page_text: str = "") -> List[Dict[str, Any]]:
    """Extract records using alternating data row → narrative row pattern."""
    records = []
    
    if not table:
        return records
    
    # Find header row and skip it
    header_found = False
    start_idx = 0
    
    for i, row in enumerate(table):
        if not row:
            continue
        
        cleaned_row = [clean_text(cell) for cell in row]
        
        # Check if this is a header row
        if any(header in cleaned_row[0].lower() for header in ['country', 'new events', 'ongoing events']):
            if verbose:
                print(f"⏭️  Found header row at index {i}: {cleaned_row[0][:50]}")
            header_found = True
            start_idx = i + 1
            break
    
    # Process rows in pairs: data row + narrative row
    i = start_idx
    while i < len(table):
        data_row = table[i] if i < len(table) else None
        narrative_row = table[i + 1] if i + 1 < len(table) else None
        
        if not data_row:
            i += 2
            continue
        
        # Clean the data row
        cleaned_data_row = [clean_text(cell) for cell in data_row]
        
        # Skip non-country rows (might be section headers or noise)
        if not is_valid_table_row(cleaned_data_row):
            if verbose:
                print(f"❌ Skipping invalid data row at {i}: {cleaned_data_row[0][:50]}")
            i += 1  # Only advance by 1, don't skip potential narrative
            continue
        
        # Extract fields from data row
        country = standardize_country_names(cleaned_data_row[0])
        event = cleaned_data_row[1] if len(cleaned_data_row) > 1 else ""
        grade = cleaned_data_row[2] if len(cleaned_data_row) > 2 else ""
        date_notified = cleaned_data_row[3] if len(cleaned_data_row) > 3 else ""
        start_period = cleaned_data_row[4] if len(cleaned_data_row) > 4 else ""
        end_period = cleaned_data_row[5] if len(cleaned_data_row) > 5 else ""
        total_cases = cleaned_data_row[6] if len(cleaned_data_row) > 6 else ""
        cases_confirmed = cleaned_data_row[7] if len(cleaned_data_row) > 7 else ""
        deaths = cleaned_data_row[8] if len(cleaned_data_row) > 8 else ""
        cfr = cleaned_data_row[9] if len(cleaned_data_row) > 9 else ""
        
        # Extract narrative from next row (might be empty) or from page text
        narrative = ""
        if narrative_row:
            # Join all cells in the narrative row to get full text
            narrative_parts = [clean_text(cell) for cell in narrative_row if clean_text(cell)]
            narrative = ' '.join(narrative_parts)
        elif page_text:
            # Fallback: extract narrative from page text for single-row tables
            narrative = extract_narrative_fallback(page_text, country, event)
        
        # Limit narrative length
        if len(narrative) > 500:
            narrative = narrative[:497] + "..."
        
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
            "NarrativeText": narrative
        }
        
        records.append(record)
        if verbose:
            narrative_preview = narrative[:100] + "..." if len(narrative) > 100 else narrative or "(no narrative)"
            print(f"✅ Extracted: {country} - {event}")
            print(f"   Narrative: \"{narrative_preview}\"")
        
        # Move to next pair (skip both data and narrative rows)
        i += 2
    
    return records


def extract_narrative_fallback(page_text: str, country: str, event: str) -> str:
    """Fallback method for narrative extraction when spatial method fails."""
    if not page_text or not country:
        return ""
    
    lines = page_text.split('\n')
    narrative_lines = []
    
    # Look for lines that mention this specific country AND event combination
    event_keywords = event.lower().split()[:2]  # First 2 words of event
    
    for i, line in enumerate(lines):
        line_clean = clean_text(line).lower()
        
        # Skip table header rows and short lines
        if (len(line_clean) < 50 or 
            any(header in line_clean for header in ['country', 'event', 'grade', 'date notified']) or
            line_clean.count('\t') > 5):  # Skip table data rows
            continue
            
        # Look for narrative that mentions the country and event context
        if (country.lower() in line_clean and 
            (any(keyword in line_clean for keyword in event_keywords) or
             any(keyword in line_clean for keyword in ['cases', 'reported', 'deaths', 'outbreak', 'confirmed']))):
            
            # Found relevant narrative - collect this and following lines
            current_line = clean_text(lines[i])
            if len(current_line) > 50:
                narrative_lines.append(current_line)
                
                # Collect continuation lines
                for j in range(i + 1, min(i + 5, len(lines))):
                    next_line = clean_text(lines[j])
                    if (len(next_line) > 30 and 
                        not any(header in next_line.lower() for header in ['country', 'event']) and
                        not is_country_name(next_line.split()[0] if next_line.split() else "")):
                        narrative_lines.append(next_line)
                    else:
                        break
                break
    
    narrative = ' '.join(narrative_lines)
    return narrative[:300] + "..." if len(narrative) > 300 else narrative

def detect_bulletin_format(pdf_path: str) -> str:
    """Detect the format of the WHO bulletin based on structure patterns.
    
    Args:
        pdf_path: Path to the WHO bulletin PDF
        
    Returns:
        Format type: 'modern', 'intermediate', or 'legacy'
    """
    with pdfplumber.open(pdf_path) as pdf:
        # Extract text from multiple pages to better detect format
        full_text = ""
        table_page_found = False
        table_start_page = None
        
        for page_num, page in enumerate(pdf.pages[:10], 1):  # Check first 10 pages
            page_text = page.extract_text()
            if page_text:
                full_text += page_text.lower()
                
                # Track where surveillance table appears
                if ("all events currently being monitored" in page_text.lower() and 
                    not table_page_found):
                    table_page_found = True
                    table_start_page = page_num
        
        if not full_text or not table_page_found:
            return "modern"  # Default
        
        # Legacy format: "by WHO AFRO" suffix and table typically starts later (page 5+)
        if ("all events currently being monitored by who afro" in full_text):
            return "legacy"
        
        # Intermediate format: Has both "new events" and "ongoing events" sections
        # and may have different table column structures
        if ("new events" in full_text and "ongoing events" in full_text):
            # Check for intermediate-specific patterns
            if any(pattern in full_text for pattern in [
                "new events reported during the reporting period",
                "ongoing events reported during the reporting period", 
                "events reported during the reporting period"
            ]):
                return "intermediate"
        
        # Modern format: Standard "all events currently being monitored" 
        # without "by WHO AFRO" and typically starts early (pages 2-4)
        if ("all events currently being monitored" in full_text and 
            table_start_page and table_start_page <= 4):
            return "modern"
        
        # Fallback: if table starts later in document, likely legacy format
        if table_start_page and table_start_page >= 5:
            return "legacy"
    
    # Default to modern if unclear
    return "modern"

def extract_surveillance_table(pdf_path: str, verbose: bool = True) -> List[Dict[str, Any]]:
    """Extract surveillance data from WHO bulletin PDF.
    
    Args:
        pdf_path: Path to the WHO bulletin PDF
        verbose: Whether to print progress messages
        
    Returns:
        List of surveillance records as dictionaries
    """
    # Detect bulletin format
    format_type = detect_bulletin_format(pdf_path)
    if verbose:
        print(f"🔍 Detected bulletin format: {format_type}")
    
    if format_type == "legacy":
        return extract_surveillance_table_legacy(pdf_path, verbose)
    elif format_type == "intermediate":
        return extract_surveillance_table_intermediate(pdf_path, verbose)
    else:
        return extract_surveillance_table_modern(pdf_path, verbose)

def extract_surveillance_table_modern(pdf_path: str, verbose: bool = True) -> List[Dict[str, Any]]:
    """Extract surveillance data from modern WHO bulletin PDF (2022+)."""
    records = []
    table_found = False
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            page_text = page.extract_text()
            
            # Find the surveillance table
            if not table_found and page_text and "All events currently being monitored" in page_text:
                table_found = True
                if verbose:
                    print(f"📍 Found surveillance table on page {page_num}")
            
            if not table_found:
                continue
            
            # Extract tables using pdfplumber
            tables = page.extract_tables()
            
            for table in tables:
                if not table:
                    continue
                
                if verbose:
                    print(f"🔍 Processing table with {len(table)} rows on page {page_num}")
                
                # Use alternating row extraction (data row → narrative row pattern)
                extracted_records = extract_records_with_alternating_narrative(
                    table, page_num, verbose, page_text
                )
                records.extend(extracted_records)
    
    if verbose:
        print(f"🎉 Extraction complete: {len(records)} records found")
    return records

def extract_surveillance_table_intermediate(pdf_path: str, verbose: bool = True) -> List[Dict[str, Any]]:
    """Extract surveillance data from intermediate WHO bulletin PDF (transitional format)."""
    records = []
    table_found = False
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            page_text = page.extract_text()
            
            # Find the surveillance table - intermediate format
            if not table_found and page_text:
                # Look for various intermediate format table headers
                if any(header in page_text for header in [
                    "All events currently being monitored",
                    "New events reported during the reporting period",
                    "Ongoing events reported during the reporting period"
                ]):
                    table_found = True
                    if verbose:
                        print(f"📍 Found intermediate surveillance table on page {page_num}")
            
            if not table_found:
                continue
            
            # Extract tables using pdfplumber
            tables = page.extract_tables()
            
            for table in tables:
                if not table:
                    continue
                
                if verbose:
                    print(f"🔍 Processing intermediate table with {len(table)} rows on page {page_num}")
                
                # Use alternating row extraction (data row → narrative row pattern)
                extracted_records = extract_records_with_alternating_narrative(
                    table, page_num, verbose, page_text
                )
                records.extend(extracted_records)
    
    if verbose:
        print(f"🎉 Intermediate extraction complete: {len(records)} records found")
    return records

def extract_surveillance_table_legacy(pdf_path: str, verbose: bool = True) -> List[Dict[str, Any]]:
    """Extract surveillance data from legacy WHO bulletin PDF (2020-2021)."""
    records = []
    table_found = False
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            page_text = page.extract_text()
            
            # Find the surveillance table - legacy format
            if not table_found and page_text and "All events currently being monitored by WHO AFRO" in page_text:
                table_found = True
                if verbose:
                    print(f"📍 Found legacy surveillance table on page {page_num}")
            
            if not table_found:
                continue
            
            # Extract tables using pdfplumber
            tables = page.extract_tables()
            
            for table in tables:
                if not table:
                    continue
                
                if verbose:
                    print(f"🔍 Processing legacy table with {len(table)} rows on page {page_num}")
                
                # Use alternating row extraction (data row → narrative row pattern)  
                extracted_records = extract_records_with_alternating_narrative(
                    table, page_num, verbose, page_text
                )
                records.extend(extracted_records)
    
    if verbose:
        print(f"🎉 Legacy extraction complete: {len(records)} records found")
    return records

def extract_to_json(pdf_path: str, output_path: Optional[str] = None, verbose: bool = True) -> str:
    """Extract surveillance data and save to JSON file.
    
    Args:
        pdf_path: Path to the WHO bulletin PDF
        output_path: Path for output JSON file (optional)
        verbose: Whether to print progress messages
        
    Returns:
        Path to the saved JSON file
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if verbose:
        print(f"🔍 Processing PDF: {pdf_path.name}")
    
    records = extract_surveillance_table(str(pdf_path), verbose=verbose)
    
    # Determine output path
    if output_path is None:
        output_path = pdf_path.with_suffix('.json')
    else:
        output_path = Path(output_path)
    
    # Write results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    
    if verbose:
        print(f"💾 Results saved to: {output_path}")
        print(f"📊 Records extracted: {len(records)}")
        
        # Preview results
        if records:
            print("\n📋 Sample records:")
            for i, record in enumerate(records[:3]):
                print(f"  {i+1}. {record['Country']} - {record['Event']} ({record.get('Total cases', 'N/A')} cases)")
    
    return str(output_path)