"""
LLM-based extraction from PDF text using OpenAI API.
This module processes extracted PDF text instead of using vision models.
"""

import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import pdfplumber
from openai import OpenAI


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using pdfplumber."""
    print(f"Extracting text from PDF: {pdf_path}")

    text_content = ""
    with pdfplumber.open(pdf_path) as pdf:
        print(f"PDF has {len(pdf.pages)} pages")

        for page_num, page in enumerate(pdf.pages, 1):
            page_text = page.extract_text()
            if page_text:
                text_content += f"\n--- PAGE {page_num} ---\n"
                text_content += page_text

    print(f"Total text extracted: {len(text_content)} characters")
    return text_content


def extract_data_from_text(text_content: str) -> List[Dict[str, Any]]:
    """
    Use OpenAI LLM to extract structured data from PDF text.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Enhanced prompt for text-based extraction
    prompt = f"""
You are extracting health emergency data from a WHO emergency bulletin text. The text contains tabular data about disease outbreaks across different countries.

CRITICAL INSTRUCTIONS:
1. Extract ALL records from the text - there should be approximately 30-50+ disease outbreak records
2. Look for table structures with headers like: Country, Event, Grade, Date notified, Start/End periods, Total cases, Deaths, CFR
3. Each record represents a country-disease combination (e.g., "Angola Cholera", "Burundi Measles")
4. SCAN THE ENTIRE TEXT - do not stop after finding a few records
5. Include page numbers when available to track coverage

Expected JSON schema for EACH record:
{{
    "Country": "string",
    "Event": "string (disease name like Cholera, Mpox, Measles, etc.)",
    "Grade": "string (e.g., Grade 3, Grade 2, Ungraded)",
    "DateNotified": "string (date when WHO was notified)",
    "StartReportingPeriod": "string",
    "EndReportingPeriod": "string", 
    "TotalCases": "string or number",
    "CasesConfirmed": "string or number",
    "Deaths": "string or number",
    "CFR": "string (case fatality rate)",
    "PageNumber": "number (if identifiable from text)"
}}

Return ONLY a JSON array containing ALL extracted records. No markdown, no explanations.

TEXT TO PROCESS:
{text_content}
"""

    try:
        print("Sending text to OpenAI for extraction...")
        print(f"Text length: {len(text_content)} characters")

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a data extraction expert. Extract all health emergency records from the provided text into structured JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=16000,
            temperature=0,
        )

        response_text = response.choices[0].message.content
        print(f"Received response: {len(response_text)} characters")

        # Clean up response if it has markdown
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        # Parse JSON
        try:
            extracted_data = json.loads(response_text)
            print(f"Successfully extracted {len(extracted_data)} records")
            return extracted_data
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response content: {response_text[:500]}...")
            # Try to find JSON in the response
            json_start = response_text.find("[")
            json_end = response_text.rfind("]") + 1
            if json_start != -1 and json_end != 0:
                json_content = response_text[json_start:json_end]
                extracted_data = json.loads(json_content)
                print(
                    f"Successfully extracted {len(extracted_data)} records after cleanup"
                )
                return extracted_data
            else:
                raise e

    except Exception as e:
        print(f"Error during LLM extraction: {e}")
        raise


def process_pdf_with_text_extraction(
    pdf_path: str, output_csv_path: str = None
) -> pd.DataFrame:
    """
    Complete pipeline: extract text from PDF, process with LLM, return DataFrame.
    """
    print("=== Starting Text-Based PDF Extraction ===")

    # Step 1: Extract text from PDF
    text_content = extract_text_from_pdf(pdf_path)

    # Step 2: Process text with LLM
    extracted_data = extract_data_from_text(text_content)

    # Step 3: Convert to DataFrame
    df = pd.DataFrame(extracted_data)
    print(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")

    if len(df) > 0:
        print("Column names:", list(df.columns))
        print("\nFirst few records:")
        print(df.head())

        # Save to CSV if path provided
        if output_csv_path:
            df.to_csv(output_csv_path, index=False)
            print(f"Saved results to: {output_csv_path}")
    else:
        print("WARNING: No data extracted!")

    return df


if __name__ == "__main__":
    # Test the text-based extraction
    pdf_path = "/Users/zackarno/Library/CloudStorage/GoogleDrive-Zachary.arno@humdata.org/Shared drives/Data Science/CERF Anticipatory Action/Cholera - General/WHO_bulletins_historical/Week_28__7_-_13_July_2025.pdf"
    output_path = "/Users/zackarno/Documents/CHD/repos/ds-cholera-pdf-scraper/outputs/text_extracted_data.csv"

    if os.path.exists(pdf_path):
        df = process_pdf_with_text_extraction(pdf_path, output_path)
        print(f"\n=== FINAL RESULTS ===")
        print(f"Total records extracted: {len(df)}")
    else:
        print(f"PDF file not found: {pdf_path}")
