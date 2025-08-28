"""
WHO Surveillance Bulletin Extractor.
Extracts complete surveillance table records with narrative text from WHO bulletins.
"""

import pandas as pd
import pdfplumber
import re
from typing import List, Dict, Optional
from pathlib import Path


class WHOSurveillanceExtractor:
    """
    Extracts surveillance records from WHO health emergency bulletins.
    
    Handles multi-page tables that span pages without repeated headers,
    capturing both structured data and associated narrative text.
    """
    
    def __init__(self):
        self.table_headers = [
            'Country', 'Event', 'Grade', 'Date_Notified', 
            'Start_Date', 'End_Date', 'Total_Cases', 
            'Confirmed_Cases', 'Deaths', 'CFR'
        ]
    
    def extract_from_pdf(self, pdf_path: str, verbose: bool = True) -> pd.DataFrame:
        """
        Extract all surveillance records from a WHO bulletin PDF.
        
        Args:
            pdf_path: Path to WHO bulletin PDF file
            verbose: Whether to print extraction progress
            
        Returns:
            DataFrame with surveillance records plus narrative_text column
        """
        if verbose:
            print(f"🔍 Starting WHO surveillance extraction: {Path(pdf_path).name}")
        
        all_records = []
        
        with pdfplumber.open(pdf_path) as pdf:
            # Process main table pages (typically 9-15)
            for page_num in range(8, min(16, len(pdf.pages))):
                page = pdf.pages[page_num]
                records = self._extract_page_records(page, page_num + 1)
                all_records.extend(records)
                
                if verbose:
                    print(f"📄 Page {page_num + 1}: extracted {len(records)} records")
        
        df = pd.DataFrame(all_records)
        
        if verbose:
            print(f"✅ Total records extracted: {len(df)}")
            
        return df
    
    def _extract_page_records(self, page, page_num: int) -> List[Dict]:
        """Extract surveillance records from a single page."""
        records = []
        
        # Extract tables from page
        tables = page.extract_tables()
        
        if tables:
            for table in tables:
                page_records = self._parse_table_records(table, page_num)
                records.extend(page_records)
        
        return records
    
    def _parse_table_records(self, table: List[List], page_num: int) -> List[Dict]:
        """Parse table rows into surveillance records with their narrative text."""
        records = []
        
        i = 0
        while i < len(table):
            row = table[i]
            if not row or len(row) < 3:
                i += 1
                continue
            
            # Clean the row
            clean_row = self._clean_table_row(row)
            
            # Check if this is a valid surveillance record
            if self._is_valid_surveillance_row(clean_row):
                record = self._parse_row_to_record(clean_row, page_num)
                if record:
                    # Look for narrative text in the next row(s)
                    narrative_text = self._extract_narrative_for_record(table, i, record)
                    record['narrative_text'] = narrative_text
                    records.append(record)
            
            i += 1
        
        return records
    
    def _clean_table_row(self, row: List) -> List[str]:
        """Clean and normalize a table row."""
        clean_row = []
        
        for cell in row:
            if cell is None:
                clean_row.append("")
            else:
                # Remove newlines and normalize whitespace
                clean_cell = str(cell).replace('\n', ' ').strip()
                # Remove extra spaces
                clean_cell = re.sub(r'\s+', ' ', clean_cell)
                clean_row.append(clean_cell)
        
        return clean_row
    
    def _is_valid_surveillance_row(self, row: List[str]) -> bool:
        """Check if a row represents a valid surveillance record."""
        if len(row) < 3:
            return False
        
        # Must have a country name (first column)
        country = row[0].strip()
        if not country or len(country) < 2:
            return False
        
        # Skip header rows
        if any(header_word in country.lower() 
               for header_word in ['country', 'event', 'ongoing', 'new events']):
            return False
        
        # Must have an event (second column)
        event = row[1].strip() if len(row) > 1 else ""
        if not event:
            return False
        
        # Must have a grade (third column) 
        grade = row[2].strip() if len(row) > 2 else ""
        if not grade:
            return False
        
        # Skip narrative-only rows (long text in first column)
        if len(country) > 100:
            return False
        
        return True
    
    def _parse_row_to_record(self, row: List[str], page_num: int) -> Optional[Dict]:
        """Convert a table row into a surveillance record."""
        try:
            record = {}
            
            # Map row cells to column headers
            for i, header in enumerate(self.table_headers):
                if i < len(row):
                    value = row[i].strip()
                    # Clean up specific value formats
                    if header in ['Total_Cases', 'Confirmed_Cases'] and value:
                        # Preserve number formatting (commas, etc.)
                        value = self._clean_numeric_field(value)
                    record[header] = value
                else:
                    record[header] = ""
            
            # Add metadata
            record['source_page'] = page_num
            
            # Basic validation
            if not record['Country'] or not record['Event']:
                return None
            
            return record
            
        except Exception as e:
            print(f"⚠️  Error parsing row on page {page_num}: {e}")
            return None
    
    def _clean_numeric_field(self, value: str) -> str:
        """Clean numeric fields while preserving formatting."""
        if not value:
            return ""
        
        # Remove extra spaces but preserve commas in numbers
        cleaned = re.sub(r'\s+', ' ', value.strip())
        
        # Handle specific formatting issues
        # e.g., "27,16" should stay as "27,16" for narrative correction later
        
        return cleaned
    
    def _extract_narrative_for_record(self, table: List[List], row_index: int, record: Dict) -> str:
        """Extract narrative text that follows a specific table row."""
        narrative_parts = []
        
        # Look at the next few rows after the current record
        for i in range(row_index + 1, min(row_index + 5, len(table))):
            next_row = table[i]
            if not next_row:
                continue
            
            # Check if this row contains narrative text
            first_cell = str(next_row[0] or "").strip()
            
            # Skip if this looks like another data record
            if self._is_valid_surveillance_row(self._clean_table_row(next_row)):
                break
            
            # If the first cell has substantial text and looks narrative-like
            if len(first_cell) > 50 and self._looks_like_narrative(first_cell):
                # Clean up the narrative text
                clean_narrative = re.sub(r'\s+', ' ', first_cell).strip()
                narrative_parts.append(clean_narrative)
            elif len(first_cell) > 0 and len(first_cell) < 50:
                # Might be a short continuation, skip it
                break
        
        return ' '.join(narrative_parts)
    
    def _looks_like_narrative(self, text: str) -> bool:
        """Check if text looks like a narrative description."""
        if not text or len(text) < 20:
            return False
        
        text_lower = text.lower()
        
        # Must contain narrative words (expanded list)
        narrative_indicators = [
            'reported', 'cases', 'deaths', 'outbreak', 'health', 
            'ministry', 'confirmed', 'suspected', 'from', 'to',
            'has', 'have', 'been', 'were', 'was', 'during', 'since',
            'people', 'million', 'assistance', 'crisis', 'affected',
            'humanitarian', 'need', 'include', 'total', 'over',
            'across', 'with', 'and', 'the', 'in', 'of'
        ]
        
        indicator_count = sum(1 for word in narrative_indicators if word in text_lower)
        
        # Must have at least 3 narrative indicators (reduced threshold)
        if indicator_count < 3:
            return False
        
        # Must not be mostly numbers (increased tolerance)
        number_count = len(re.findall(r'\b\d+\b', text))
        word_count = len(text.split())
        
        if word_count > 0 and (number_count / word_count) > 0.4:
            return False
        
        # Must not contain obvious table headers/structure
        if any(header in text_lower for header in ['grade', 'cfr', 'country', 'event']):
            # Only exclude if it looks like a header row, not if these words appear naturally
            if len(text.split()) < 10:
                return False
        
        # Additional check: must be substantial text (at least 15 words)
        if word_count < 15:
            return False
            
        return True
    
    def _add_narrative_text(self, records: List[Dict], page):
        """Add narrative text to records based on page content."""
        if not records:
            return
        
        # Extract full page text
        page_text = page.extract_text() or ""
        
        # Split into lines first, then group into meaningful segments
        lines = page_text.split('\n')
        narrative_segments = []
        
        current_segment = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line looks like a narrative description
            if self._is_narrative_line(line):
                current_segment.append(line)
            else:
                # End current segment if we have content
                if current_segment:
                    segment_text = ' '.join(current_segment)
                    narrative_segments.append(segment_text)
                    current_segment = []
        
        # Add final segment
        if current_segment:
            segment_text = ' '.join(current_segment)
            narrative_segments.append(segment_text)
        
        # Match segments to records
        for record in records:
            country = record.get('Country', '')
            
            # Find relevant narrative segments
            relevant_segments = []
            
            for segment in narrative_segments:
                if self._segment_matches_record(segment, country):
                    relevant_segments.append(segment)
            
            # Join matched segments
            record['narrative_text'] = ' '.join(relevant_segments)
    
    def _is_narrative_line(self, line: str) -> bool:
        """Check if a line looks like narrative text (not table data)."""
        if not line or len(line) < 20:
            return False
        
        # Skip lines that look like table headers or data
        if re.search(r'\b(Country|Event|Grade [123]|CFR|\d{1,2}[-/]\w{3}[-/]\d{2,4})\b', line):
            return False
        
        # Skip lines with mostly numbers/structured data
        if len(re.findall(r'\b\d+\b', line)) > 5:
            return False
        
        # Must contain narrative indicators
        narrative_words = [
            'reported', 'cases', 'deaths', 'outbreak', 'health', 
            'ministry', 'confirmed', 'suspected', 'from', 'to',
            'has', 'have', 'been', 'were', 'was'
        ]
        
        line_lower = line.lower()
        narrative_word_count = sum(1 for word in narrative_words if word in line_lower)
        
        return narrative_word_count >= 2
    
    def _segment_matches_record(self, segment: str, country: str) -> bool:
        """Check if a narrative segment is relevant to a specific record."""
        if not country or not segment:
            return False
        
        # Must mention the country
        if country.lower() not in segment.lower():
            return False
        
        # Must be substantial narrative text
        if len(segment.split()) < 10:
            return False
        
        return True
    
    def _paragraph_matches_record(self, paragraph: str, country: str, event: str) -> bool:
        """Check if a paragraph contains narrative relevant to a record."""
        if not country or not paragraph:
            return False
        
        para_lower = paragraph.lower()
        country_lower = country.lower()
        
        # Must mention the country
        if country_lower not in para_lower:
            return False
        
        # Skip table-formatted content
        if re.search(r'\b(Grade [123]|CFR|\d{1,2}[-/]\w{3}[-/]\d{2,4})\b', paragraph):
            return False
        
        # Must be substantial narrative text
        if len(paragraph.split()) < 15:
            return False
        
        # Look for narrative indicators
        narrative_indicators = [
            'reported', 'cases', 'deaths', 'outbreak', 'health', 
            'ministry', 'confirmed', 'suspected'
        ]
        
        if any(indicator in para_lower for indicator in narrative_indicators):
            return True
        
        return False
    
    def save_results(self, df: pd.DataFrame, output_path: str) -> str:
        """Save extraction results to CSV."""
        output_path = Path(output_path)
        df.to_csv(output_path, index=False)
        print(f"💾 Results saved to: {output_path}")
        return str(output_path)
    
    def get_extraction_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary statistics about the extraction."""
        if len(df) == 0:
            return {"total_records": 0}
        
        summary = {
            "total_records": len(df),
            "countries": df['Country'].nunique(),
            "events": df['Event'].nunique(),
            "pages_processed": df['source_page'].nunique(),
            "records_with_narrative": (df['narrative_text'].str.len() > 0).sum(),
            "top_events": df['Event'].value_counts().head(5).to_dict(),
            "top_countries": df['Country'].value_counts().head(5).to_dict()
        }
        
        return summary


def main():
    """Example usage of WHOSurveillanceExtractor."""
    pdf_path = "/Users/zackarno/Library/CloudStorage/GoogleDrive-Zachary.arno@humdata.org/Shared drives/Data Science/CERF Anticipatory Action/Cholera - General/WHO_bulletins_historical/Week_28__7_-_13_July_2025.pdf"
    
    extractor = WHOSurveillanceExtractor()
    df = extractor.extract_from_pdf(pdf_path)
    
    # Show results
    print(f"\n📊 EXTRACTION SUMMARY:")
    summary = extractor.get_extraction_summary(df)
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    # Show Angola records specifically
    angola_records = df[df['Country'].str.contains('Angola', na=False, case=False)]
    print(f"\n🇦🇴 Angola records found: {len(angola_records)}")
    for _, record in angola_records.iterrows():
        cases = record['Total_Cases']
        deaths = record['Deaths']
        print(f"   {record['Event']}: {cases} cases, {deaths} deaths (page {record['source_page']})")
        
        # Show narrative preview for the cholera case
        if 'cholera' in record['Event'].lower() and record['narrative_text']:
            narrative_preview = record['narrative_text'][:200] + "..."
            print(f"     Narrative: {narrative_preview}")
    
    # Save results
    output_path = "outputs/who_surveillance_extraction.csv"
    extractor.save_results(df, output_path)


if __name__ == "__main__":
    main()