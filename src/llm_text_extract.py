"""
LLM-based extraction from PDF text using OpenAI API.
This module processes extracted PDF text instead of using vision models.
"""

import json
import os
import time
from typing import Any, Dict, List

import pandas as pd
import pdfplumber
from openai import OpenAI

from accuracy_evaluator import evaluate_and_log_accuracy
from prompt_logger import PromptLogger
from prompt_manager import PromptManager


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
    Now with integrated prompt logging.
    """
    # Initialize prompt management and logging
    prompt_manager = PromptManager()
    prompt_logger = PromptLogger()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Build prompt using prompt manager
    system_prompt, user_prompt, prompt_metadata, prompt_for_logging = (
        prompt_manager.build_prompt(
            prompt_type="health_data_extraction", text_content=text_content
        )
    )

    # Model configuration
    model_name = "gpt-4o"
    model_parameters = {"max_tokens": 16384, "temperature": 0}  # Max allowed for gpt-4o

    start_time = time.time()

    try:
        print("Sending text to OpenAI for extraction...")
        print(f"Text length: {len(text_content)} characters")
        print(
            f"Using prompt: {prompt_metadata['prompt_type']} v{prompt_metadata['version']}"
        )

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": user_prompt},
            ],
            **model_parameters,
        )

        raw_response = response.choices[0].message.content
        execution_time = time.time() - start_time

        print(f"Received response: {len(raw_response)} characters")
        print(f"Execution time: {execution_time:.2f} seconds")

        # Clean up response if it has markdown
        response_text = raw_response
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        # Parse JSON
        parsing_errors = None
        parsed_success = False
        extracted_data = []

        try:
            extracted_data = json.loads(response_text)
            parsed_success = True
            print(f"Successfully extracted {len(extracted_data)} records")
        except json.JSONDecodeError as e:
            parsing_errors = f"JSON parsing error: {e}"
            print(f"JSON parsing error: {e}")
            print(f"Response content: {response_text[:500]}...")

            # Try to find JSON in the response
            json_start = response_text.find("[")
            json_end = response_text.rfind("]") + 1
            if json_start != -1 and json_end != 0:
                json_content = response_text[json_start:json_end]
                try:
                    extracted_data = json.loads(json_content)
                    parsed_success = True
                    parsing_errors = f"Recovered after cleanup: {e}"
                    print(
                        f"Successfully extracted {len(extracted_data)} records after cleanup"
                    )
                except json.JSONDecodeError as e2:
                    parsing_errors = f"JSON parsing failed even after cleanup: {e2}"
                    raise e

        # Log the LLM call
        records_extracted = len(extracted_data) if parsed_success else 0
        custom_metrics = {
            "text_length_chars": len(text_content),
            "response_length_chars": len(raw_response),
            "cleanup_required": "```json" in raw_response or "```" in raw_response,
        }

        call_id = prompt_logger.log_llm_call(
            prompt_metadata=prompt_metadata,
            model_name=model_name,
            model_parameters=model_parameters,
            system_prompt=system_prompt,
            user_prompt=prompt_for_logging,  # Use the version with content summary
            raw_response=raw_response,
            parsed_success=parsed_success,
            records_extracted=records_extracted,
            parsing_errors=parsing_errors,
            execution_time_seconds=execution_time,
            custom_metrics=custom_metrics,
        )

        # Evaluate accuracy against baseline if extraction was successful
        if parsed_success and len(extracted_data) > 0:
            try:
                print("🎯 Evaluating accuracy against baseline...")
                accuracy_metrics = evaluate_and_log_accuracy(
                    llm_raw_df=pd.DataFrame(extracted_data),
                    prompt_call_id=call_id,
                    prompt_metadata=prompt_metadata,
                    output_base_path=f"/Users/zackarno/Documents/CHD/repos/ds-cholera-pdf-scraper/logs/accuracy/evaluation_{call_id}",
                )

                print(f"📊 Accuracy Summary:")
                print(f"   Coverage: {accuracy_metrics['coverage_rate']}%")
                print(f"   Precision: {accuracy_metrics['precision_rate']}%")
                print(f"   Overall Accuracy: {accuracy_metrics['overall_accuracy']}%")
                print(f"   Composite Score: {accuracy_metrics['composite_score']}%")

            except Exception as acc_error:
                print(f"⚠️ Accuracy evaluation failed: {acc_error}")

        return extracted_data

    except Exception as e:
        execution_time = time.time() - start_time
        error_message = str(e)

        # Log failed call
        prompt_logger.log_llm_call(
            prompt_metadata=prompt_metadata,
            model_name=model_name,
            model_parameters=model_parameters,
            system_prompt=system_prompt,
            user_prompt=prompt_for_logging,  # Use the version with content summary
            raw_response=f"ERROR: {error_message}",
            parsed_success=False,
            records_extracted=0,
            parsing_errors=error_message,
            execution_time_seconds=execution_time,
            custom_metrics={"text_length_chars": len(text_content)},
        )

        print(f"Error during LLM extraction: {e}")
        raise


def process_pdf_with_text_extraction(
    pdf_path: str, output_csv_path: str = None
) -> pd.DataFrame:
    """
    Complete pipeline: extract text from PDF, process with LLM, return raw DataFrame.
    Post-processing should be applied separately during experimental phase.
    """
    print("=== Starting Text-Based PDF Extraction ===")

    # Step 1: Extract text from PDF
    text_content = extract_text_from_pdf(pdf_path)

    # Step 2: Process text with LLM
    extracted_data = extract_data_from_text(text_content)

    # Step 3: Convert to DataFrame (RAW OUTPUT)
    df = pd.DataFrame(extracted_data)
    print(f"Created RAW DataFrame with {len(df)} rows and {len(df.columns)} columns")

    if len(df) > 0:
        print("Column names:", list(df.columns))
        print("\nFirst few records:")
        print(df.head())

        # Save RAW results to CSV if path provided
        if output_csv_path:
            # Get current prompt version info for filename tagging
            prompt_manager = PromptManager()
            current_prompt = prompt_manager.get_current_prompt("health_data_extraction")
            prompt_version = current_prompt["version"]

            # Add prompt version to filename
            if output_csv_path.endswith(".csv"):
                base_path = output_csv_path[:-4]
                tagged_path = f"{base_path}_prompt_{prompt_version}.csv"
            else:
                tagged_path = f"{output_csv_path}_prompt_{prompt_version}.csv"

            df.to_csv(tagged_path, index=False)
            print(f"Saved RAW LLM results to: {tagged_path}")
            print(f"💡 Tagged with prompt version: {prompt_version}")
            print(
                "💡 Apply post-processing separately using apply_post_processing_pipeline()"
            )
    else:
        print("WARNING: No data extracted!")

    return df


if __name__ == "__main__":
    # Test the text-based extraction
    pdf_path = "/Users/zackarno/Library/CloudStorage/GoogleDrive-Zachary.arno@humdata.org/Shared drives/Data Science/CERF Anticipatory Action/Cholera - General/WHO_bulletins_historical/Week_28__7_-_13_July_2025.pdf"

    # Base output path - will be automatically tagged with prompt version
    base_output_path = "/Users/zackarno/Documents/CHD/repos/ds-cholera-pdf-scraper/outputs/text_extracted_data"

    # Get current prompt version for display
    prompt_manager = PromptManager()
    current_prompt = prompt_manager.get_current_prompt("health_data_extraction")
    prompt_version = current_prompt["version"]

    print(f"🎯 Running extraction with prompt version: {prompt_version}")
    print(f"📁 Output will be saved as: {base_output_path}_prompt_{prompt_version}.csv")

    if os.path.exists(pdf_path):
        df = process_pdf_with_text_extraction(pdf_path, f"{base_output_path}.csv")
        print(f"\n=== FINAL RESULTS ===")
        print(f"Total records extracted: {len(df)}")
    else:
        print(f"PDF file not found: {pdf_path}")
