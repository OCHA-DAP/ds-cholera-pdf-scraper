"""
LLM-based extraction module for cholera PDF data.

This module handles calling OpenAI API with PDF text or structured input
to extract table data from cholera reports.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import openai
import pandas as pd

logger = logging.getLogger(__name__)

PROMPT_USE = """
You are an expert data extraction assistant for WHO outbreak report PDFs. Extract structured data from the uploaded PDF and return ONLY valid JSON with no markdown formatting, no code blocks, no commentary, and no additional text.

CRITICAL: Your response must be pure JSON only. Do not wrap in ```json or ``` blocks. Do not include any explanatory text before or after the JSON.

Output Schema:
Return exactly one JSON object with this structure:

{
  "records": [
    {
      "Country": "string",
      "Event": "string", 
      "Grade": "string",
      "Date notified to WCO": "string",
      "Start of reporting period": "string",
      "End of reporting period": "string",
      "Total cases": number,
      "Cases Confirmed": number,
      "Deaths": number,
      "CFR": number,
      "WeekNumber": number,
      "Year": number,
      "Month": number,
      "PageNumber": number
    }
  ]
}

Field Guidelines:
- Country: Full country name as stated in document
- Event: Disease/outbreak type (e.g., "Cholera", "COVID-19", "Measles") 
- Grade: Emergency classification - IMPORTANT: Use exact format from document but clean up formatting:
  * "Grade 1", "Grade 2", "Grade 3" for emergency grades
  * "Ungraded" for ungraded events
  * "Protracted 1", "Protracted 2", "Protracted 3" for protracted emergencies
  * Remove any extra line breaks or formatting issues
- "Date notified to WCO": Date when WHO was notified (preserve original format, use YYYY-MM-DD if possible)
- "Start of reporting period": Start date of reporting period (preserve original format, use YYYY-MM-DD if possible)
- "End of reporting period": End date of reporting period (preserve original format, use YYYY-MM-DD if possible)
- "Total cases": Total number of cases (convert to number, use 0 for missing/dash/empty)
- "Cases Confirmed": Confirmed cases count (convert to number, use 0 for missing/dash/empty)
- Deaths: Number of deaths (convert to number, use 0 for missing/dash/empty)
- CFR: Case Fatality Rate as decimal number (e.g., 1.5% becomes 1.5, use 0.0 for missing/dash/empty)
- WeekNumber: Epidemiological week number (extract from document context, filename, or date references)
- Year: Year (extract from document context, filename, or date references)
- Month: Month number 1-12 (extract from document context, filename, or date references)
- PageNumber: The page number where this record was found (1, 2, 3, etc.)

Data Conversion Rules:
1. Convert all numeric fields to actual numbers, not strings
2. Convert dashes (-), missing values, empty cells, or "N/A" to appropriate defaults:
   - Numbers: use 0
   - Strings: use empty string ""
3. For percentages: convert "1.5%" to 1.5 (remove % symbol)
4. For large numbers: convert "1,234" to 1234 (remove commas)
5. For Grade field: Clean up any formatting issues like line breaks but preserve the semantic meaning
6. Extract WeekNumber, Year, Month from document metadata, filename, dates, or any time references in the document

CRITICAL EXTRACTION REQUIREMENTS:
- This document likely contains LARGE TABLES with 100+ rows across multiple pages
- EXTRACT EVERY SINGLE ROW from every table - do not skip, summarize, or sample
- Scan ALL PAGES of the document thoroughly for outbreak/emergency data tables
- Look for continuation tables that may span multiple pages
- Some tables may have page breaks - ensure you capture data from ALL pages
- Expected output: 100+ records for documents like WHO emergency bulletins
- If you only find 10-20 records, you are missing data - look harder for more tables

MANDATORY PAGE-BY-PAGE SCANNING PROCESS:
1. Start at PAGE 1 and systematically examine EVERY SINGLE PAGE
2. For each page, record the page number (1, 2, 3, etc.) for each record found
3. Look for tables on EVERY page - don't skip pages assuming they have no data
4. Continue through ALL pages until you reach the end of the document
5. Tables often continue across multiple pages - capture ALL continuation rows
6. Some pages may have partial tables or summary sections - extract those too

Table Scanning Strategy:
- SCAN EVERY PAGE from beginning to end (pages 1, 2, 3, 4, 5, 6, 7, 8, 9, 10+)
- Look for ANY tabular data containing country names and outbreak information
- Common table headers: Country, Event, Grade, Cases, Deaths, CFR, Dates
- Tables may be titled: "All emergencies currently being monitored", "Ongoing outbreaks", "New events", etc.
- Don't stop after finding one table - there may be multiple tables throughout the document
- Pay special attention to continuation markers like "continued on next page"
- Record the PageNumber field for each record based on which page it appears on

Data Completeness Check:
- If you extract fewer than 50 records from a WHO emergency bulletin, review the document again
- Ensure you've scanned all pages for tabular outbreak data
- Multiple countries should have multiple events/time periods
- Your output should include records from multiple different page numbers (1, 2, 3, 4, 5+)

Response Format: Pure JSON only, no other text whatsoever."""


class LLMExtractor:
    """Handles LLM-based extraction of data from cholera PDFs."""

    def __init__(self, api_key: str, model: str = "gpt-4o", temperature: float = 0.1):
        """
        Initialize the LLM extractor.

        Args:
            api_key: OpenAI API key
            model: OpenAI model to use
            temperature: Temperature for generation (low for consistent)
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def upload_pdf_to_openai(self, pdf_path: Path) -> str:
        """
        Upload PDF file to OpenAI and return file ID.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            OpenAI file ID for the uploaded PDF
        """
        logger.info(f"Uploading PDF to OpenAI: {pdf_path}")

        try:
            with open(pdf_path, "rb") as pdf_file:
                file_response = self.client.files.create(
                    file=pdf_file, purpose="user_data"
                )

            file_id = file_response.id
            logger.info(f"Successfully uploaded PDF, file ID: {file_id}")
            return file_id

        except Exception as e:
            logger.error(f"Failed to upload PDF to OpenAI: {e}")
            raise

    def extract_data_from_pdf(self, file_id: str) -> Dict[str, Any]:
        """
        Extract structured data from uploaded PDF using LLM.

        Args:
            file_id: OpenAI file ID for the uploaded PDF

        Returns:
            Extracted data as dictionary
        """
        try:
            logger.info("Calling OpenAI API for PDF data extraction")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise data extraction expert. "
                            "Return ONLY valid JSON with no markdown formatting. "
                            "Never use ```json blocks or any other text "
                            "formatting. Your entire response must be valid JSON "
                            "that can be parsed directly."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPT_USE},
                            {"type": "file", "file": {"file_id": file_id}},
                        ],
                    },
                ],
                temperature=self.temperature,
                max_tokens=16000,  # Increased for very large table extraction
            )

            # Parse the JSON response
            import json

            raw_content = response.choices[0].message.content
            logger.info(f"Raw OpenAI response length: {len(raw_content)} characters")
            logger.info(f"Raw OpenAI response preview: {repr(raw_content[:500])}...")
            logger.info(f"Response usage tokens: {response.usage}")

            if not raw_content or raw_content.strip() == "":
                logger.error("OpenAI returned empty response")
                raise ValueError("Empty response from OpenAI")

            # Clean markdown formatting if present
            content = raw_content.strip()
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            if content.startswith("```"):
                content = content[3:]  # Remove ```
            if content.endswith("```"):
                content = content[:-3]  # Remove trailing ```
            content = content.strip()

            # Check if response was truncated (incomplete JSON)
            try:
                extracted_data = json.loads(content)
            except json.JSONDecodeError as json_err:
                # Try to salvage partial data if response was truncated
                logger.warning(
                    f"JSON parsing failed, attempting to salvage truncated response: {json_err}"
                )

                # Check if it's a truncation issue (common with large responses)
                if "Unterminated string" in str(json_err) or "Expecting" in str(
                    json_err
                ):
                    logger.warning(
                        "Response appears to be truncated due to token limit"
                    )

                    # Try to find the last complete record
                    try:
                        # Find the last complete record by looking for closing braces
                        lines = content.split("\n")
                        salvaged_content = ""
                        bracket_count = 0
                        in_records = False

                        for line in lines:
                            if '"records":' in line:
                                in_records = True

                            if in_records:
                                bracket_count += line.count("{") - line.count("}")
                                salvaged_content += line + "\n"

                                # If we're back to balanced brackets and have a closing array
                                if bracket_count <= 1 and "]" in line:
                                    # Add the final closing brace if needed
                                    if not salvaged_content.strip().endswith("}"):
                                        salvaged_content += "\n}"
                                    break
                            else:
                                salvaged_content += line + "\n"

                        # Clean up and try parsing again
                        if not salvaged_content.strip().endswith("}"):
                            salvaged_content = salvaged_content.strip() + "\n}"

                        logger.info(
                            f"Attempting to parse salvaged content ({len(salvaged_content)} chars)"
                        )
                        extracted_data = json.loads(salvaged_content)
                        logger.info(
                            f"Successfully salvaged {len(extracted_data.get('records', []))} records from truncated response"
                        )

                    except Exception as salvage_err:
                        logger.error(
                            f"Failed to salvage truncated response: {salvage_err}"
                        )
                        logger.error(f"Last 500 chars of content: {content[-500:]}")
                        raise json_err
                else:
                    raise json_err

            logger.info("Successfully extracted data from PDF")
            return extracted_data

        except Exception as e:
            logger.error(f"Failed to extract data via LLM: {e}")
            raise

    def save_to_csv(self, extracted_data: Dict[str, Any], output_path: Path) -> None:
        """
        Save extracted records to CSV file.

        Args:
            extracted_data: Dictionary containing 'records' key with list
            output_path: Path where to save the CSV file
        """
        if "records" not in extracted_data or not extracted_data["records"]:
            logger.warning("No records found to save to CSV")
            return

        records = extracted_data["records"]
        df = pd.DataFrame(records)

        # Add metadata columns
        if "source_file" in extracted_data:
            df["source_file"] = extracted_data["source_file"]
        if "extraction_timestamp" in extracted_data:
            df["extraction_timestamp"] = extracted_data["extraction_timestamp"]

        df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved {len(records)} records to CSV: {output_path}")

    def process_pdf_file(
        self,
        pdf_path: Path,
        save_csv: bool = False,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Process a single PDF file end-to-end.

        Args:
            pdf_path: Path to the PDF file
            save_csv: Whether to save results to CSV file
            output_dir: Directory to save CSV file (defaults to current)

        Returns:
            Extracted data dictionary
        """
        logger.info(f"Processing PDF file: {pdf_path}")

        # Upload PDF to OpenAI
        file_id = self.upload_pdf_to_openai(pdf_path)

        try:
            # Extract structured data using LLM
            extracted_data = self.extract_data_from_pdf(file_id)

            # Add metadata
            extracted_data["source_file"] = pdf_path.name
            timestamp = pd.Timestamp.now().isoformat()
            extracted_data["extraction_timestamp"] = timestamp

            # Save to CSV if requested
            if save_csv:
                if output_dir is None:
                    output_dir = Path.cwd()

                # Generate CSV filename based on PDF name
                csv_name = pdf_path.stem + "_extracted.csv"
                csv_path = output_dir / csv_name

                self.save_to_csv(extracted_data, csv_path)

            return extracted_data

        finally:
            # Clean up uploaded file
            try:
                self.client.files.delete(file_id)
                logger.info(f"Cleaned up uploaded file: {file_id}")
            except Exception as e:
                logger.warning(f"Failed to delete uploaded file {file_id}: {e}")

    def process_multiple_pdfs(self, pdf_paths: List[Path]) -> List[Dict[str, Any]]:
        """
        Process multiple PDF files.

        Args:
            pdf_paths: List of PDF file paths

        Returns:
            List of extracted data dictionaries
        """
        results = []

        for i, pdf_path in enumerate(pdf_paths, 1):
            try:
                logger.info(f"Processing PDF {i}/{len(pdf_paths)}: {pdf_path}")

                extracted_data = self.process_pdf_file(pdf_path)
                results.append(extracted_data)

            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {e}")
                # Continue with other files
                continue

        success_msg = f"Successfully processed {len(results)}/{len(pdf_paths)} PDFs"
        logger.info(success_msg)
        return results


def main():
    """Main execution function for testing."""
    logging.basicConfig(level=logging.INFO)

    # TODO: Set up API key from environment or config
    import os

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        return

    extractor = LLMExtractor(api_key)

    # Example usage
    pdf_path = Path("example.pdf")
    if pdf_path.exists():
        result = extractor.process_pdf_file(pdf_path)
        print(f"Extracted data: {result}")
    else:
        logger.warning(f"PDF file not found: {pdf_path}")


if __name__ == "__main__":
    main()
