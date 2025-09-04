"""
LLM-based extraction from PDF text using configurable LLM providers.
This module processes extracted PDF text and supports OpenAI and OpenRouter.
Enhanced with pdfplumber preprocessing for structured table extraction.
"""

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pdfplumber

from src.accuracy_evaluator import evaluate_and_log_accuracy
from src.config import Config
from src.llm_client import LLMClient
from src.prompt_logger import PromptLogger
from src.prompt_manager import PromptManager

# Import table reconstruction preprocessing
try:
    from src.preprocess.llm_narrative_linking import LLMNarrativeLinking
    from src.preprocess.simple_surveillance_processor import (
        process_surveillance_bulletin,
    )

    PREPROCESSOR_AVAILABLE = True
except ImportError:
    PREPROCESSOR_AVAILABLE = False

# Import blank field treatment preprocessing
try:
    from src.preprocess.blank_field_treatment import process_with_blank_treatment

    BLANK_TREATMENT_AVAILABLE = True
except ImportError:
    BLANK_TREATMENT_AVAILABLE = False


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


def extract_data_from_text(
    text_content: str, model_name: str = None
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Use LLM to extract structured data from PDF text.
    Now supports both OpenAI and OpenRouter with integrated prompt logging.

    Args:
        text_content: PDF text content to process
        model_name: Optional model override (e.g., "anthropic/claude-3.5-sonnet")

    Returns:
        Tuple of (extracted records list, database call_id)
    """
    # Initialize prompt management and logging
    prompt_manager = PromptManager()
    prompt_logger = PromptLogger()

    # Initialize LLM client with intelligent provider selection
    if model_name:
        # Create client for specific model (auto-selects OpenAI API for OpenAI models)
        llm_client = LLMClient.create_client_for_model(model_name)
    else:
        # Use default configuration
        llm_client = LLMClient()

    # Build prompt using prompt manager
    system_prompt, user_prompt, prompt_metadata, prompt_for_logging = (
        prompt_manager.build_prompt(
            prompt_type="health_data_extraction", text_content=text_content
        )
    )

    start_time = time.time()

    try:
        # Get model info for logging
        model_info = llm_client.get_model_info()
        actual_model_name = model_info["model_name"]
        provider = model_info["provider"]

        print(f"Sending text to {provider} for extraction...")
        print(f"Model: {actual_model_name}")
        print(f"Text length: {len(text_content)} characters")
        print(
            f"Using prompt: {prompt_metadata['prompt_type']} "
            f"v{prompt_metadata['version']}"
        )

        # Make LLM call with appropriate token allocation
        # GPT-5 and Grok-4: Must have enough tokens for reasoning + response
        is_reasoning_model = (
            "gpt-5" in actual_model_name.lower() or "grok" in actual_model_name.lower()
        )
        if is_reasoning_model:
            max_tokens = 100000  # Higher limit for reasoning models
        else:
            max_tokens = 16384  # Standard limit for other models

        raw_response, api_metadata = llm_client.create_chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=0,
        )

        execution_time = time.time() - start_time

        print(f"✅ API call successful: {len(raw_response)} characters received")
        print(f"Execution time: {execution_time:.2f} seconds")
        print("🔄 Starting JSON parsing...")

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
            print(
                f"✅ JSON parsing successful: {len(extracted_data)} records extracted"
            )
        except json.JSONDecodeError as e:
            parsing_errors = f"JSON parsing error: {e}"
            print(f"❌ JSON parsing failed: {e}")
            print(f"📄 Response preview: {response_text[:500]}...")
            print("🔧 Attempting recovery...")

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
                        f"✅ Recovery successful: {len(extracted_data)} records extracted"
                    )
                except json.JSONDecodeError as e2:
                    parsing_errors = f"JSON parsing failed even after cleanup: {e2}"
                    print(f"❌ Recovery failed: {e2}")
                    # Don't raise - preserve the original response for potential manual recovery
                    print(
                        f"💾 Preserving original response for potential recovery ({len(response_text)} chars)"
                    )
                    parsed_success = False

        # Prepare metrics and model info for logging
        records_extracted = len(extracted_data) if parsed_success else 0
        custom_metrics = {
            "text_length_chars": len(text_content),
            "response_length_chars": len(raw_response),
            "cleanup_required": "```json" in raw_response or "```" in raw_response,
            "provider": provider,
            "usage_tokens": api_metadata.get("usage"),
        }

        call_id = prompt_logger.log_llm_call(
            prompt_metadata=prompt_metadata,
            model_name=actual_model_name,
            model_parameters=api_metadata["model_parameters"],
            system_prompt=system_prompt,
            user_prompt=prompt_for_logging,
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

                # Create output path based on prompt version and model
                prompt_version = prompt_metadata.get("version", "unknown")
                model_safe = actual_model_name.replace("/", "_").replace("-", "_")
                accuracy_output_path = str(
                    Config.LOGS_DIR / "accuracy" / f"{prompt_version}_{model_safe}"
                )

                accuracy_metrics = evaluate_and_log_accuracy(
                    llm_raw_df=pd.DataFrame(extracted_data),
                    prompt_call_id=call_id,
                    prompt_metadata=prompt_metadata,
                    output_base_path=accuracy_output_path,
                )

                print("📊 Accuracy Summary:")
                print(f"   Coverage: {accuracy_metrics['coverage_rate']}%")
                print(f"   Precision: {accuracy_metrics['precision_rate']}%")
                accuracy = accuracy_metrics["overall_accuracy"]
                print(f"   Overall Accuracy: {accuracy}%")
                composite = accuracy_metrics["composite_score"]
                print(f"   Composite Score: {composite}%")
                print(
                    f"💾 Accuracy evaluation saved (overwrites previous): {accuracy_output_path}*"
                )

            except Exception as acc_error:
                print(f"⚠️ Accuracy evaluation failed: {acc_error}")

        # Save final LLM output with proper naming convention
        if extracted_data:
            df = pd.DataFrame(extracted_data)

            # Use database call_id for better linking between CSV and database
            # New format: extraction_<call_id>_prompt_<version>_model_<model>.csv
            from src.config import Config

            config = Config.get_llm_client_config()
            model_for_filename = config["model"].replace("/", "_").replace("-", "_")

            # Get prompt version from metadata
            prompt_version = prompt_metadata.get("version", "unknown")

            output_dir = Config.OUTPUTS_DIR
            tagged_path = (
                output_dir
                / f"extraction_{call_id}_prompt_{prompt_version}_model_{model_for_filename}.csv"
            )

            df.to_csv(tagged_path, index=False)
            print(f"✅ Saved final LLM results to: {tagged_path}")
            print(f"💡 Linked to database record ID: {call_id}")
            print(f"💡 Tagged with prompt version: {prompt_version}")
            print(f"💡 Tagged with model: {model_for_filename}")

        return extracted_data, call_id

    except Exception as e:
        execution_time = time.time() - start_time
        error_message = str(e)

        # Preserve original response if available, otherwise store error
        response_to_store = f"ERROR: {error_message}"
        raw_response_value = locals().get("raw_response")
        if raw_response_value:
            response_to_store = raw_response_value
            print(
                f"💾 Preserving original response despite error ({len(raw_response_value)} chars)"
            )

        # Log failed call
        call_id = prompt_logger.log_llm_call(
            prompt_metadata=prompt_metadata,
            model_name=llm_client.model_name,
            model_parameters={"max_tokens": 16384, "temperature": 0},
            system_prompt=system_prompt,
            user_prompt=prompt_for_logging,
            raw_response=response_to_store,
            parsed_success=False,
            records_extracted=0,
            parsing_errors=error_message,
            execution_time_seconds=execution_time,
            custom_metrics={"text_length_chars": len(text_content)},
        )

        print(f"Error during LLM extraction: {e}")
        print(f"🚨 This was an API-level failure (not a parsing issue)")
        raise


def extract_data_with_blank_treatment(
    pdf_path: str, model_name: str = None
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Extract data using blank field treatment preprocessing.
    Simple approach that standardizes blank fields before LLM processing.

    Args:
        pdf_path: Path to PDF file
        model_name: Optional model override

    Returns:
        Tuple of (extracted records, call_id)
    """
    if not BLANK_TREATMENT_AVAILABLE:
        print("⚠️ Blank field treatment not available, falling back to raw text")
        text_content = extract_text_from_pdf(pdf_path)
        return extract_data_from_text(text_content, model_name=model_name)

    try:
        print("🔧 Running table-aware blank field treatment preprocessing...")

        # Use the new table-aware blank treatment
        from src.preprocess.blank_field_treatment import (
            extract_text_with_table_aware_blanks,
        )

        treated_text, treatments_applied = extract_text_with_table_aware_blanks(
            pdf_path
        )

        print(f"✅ Table-aware blank treatment successful:")
        print(f"   Treatments applied: {treatments_applied}")

        # Process treated text with LLM
        print("🧠 Sending treated text to LLM...")
        extracted_data, call_id = extract_data_from_text(
            treated_text, model_name=model_name
        )

        print(
            f"✅ Table-aware blank treatment extraction completed: {len(extracted_data)} records"
        )

        # Save the extracted data with appropriate naming
        if extracted_data:
            import pandas as pd

            from src.config import Config

            df = pd.DataFrame(extracted_data)

            # Get model info for filename
            if model_name:
                model_for_filename = model_name.replace("/", "_").replace("-", "_")
            else:
                config = Config.get_llm_client_config()
                model_for_filename = config["model"].replace("/", "_").replace("-", "_")

            # Get prompt version
            from src.prompt_manager import PromptManager

            prompt_manager = PromptManager()
            current_prompt = prompt_manager.get_current_prompt("health_data_extraction")
            prompt_version = current_prompt["version"]

            output_dir = Config.OUTPUTS_DIR

            if call_id:
                # Standard naming with table-aware-blank-treatment identifier
                tagged_path = (
                    output_dir
                    / f"extraction_{call_id}_prompt_{prompt_version}_model_{model_for_filename}_table_aware_blank_treatment.csv"
                )
            else:
                # Fallback naming
                import time

                timestamp = int(time.time())
                tagged_path = (
                    output_dir
                    / f"extraction_fallback_{timestamp}_prompt_{prompt_version}_model_{model_for_filename}_table_aware_blank_treatment.csv"
                )

            df.to_csv(tagged_path, index=False)
            print(f"✅ Saved table-aware blank treatment results to: {tagged_path}")
            if call_id:
                print(f"💡 Linked to database record ID: {call_id}")
            print(f"💡 Tagged with prompt version: {prompt_version}")
            print(f"💡 Tagged with model: {model_for_filename}")
            print(f"💡 Preprocessing: table-aware-blank-treatment")

        return extracted_data, call_id

    except Exception as e:
        print(f"❌ Blank treatment preprocessing error: {e}")
        print("🔄 Falling back to raw text extraction...")
        text_content = extract_text_from_pdf(pdf_path)
        return extract_data_from_text(text_content, model_name=model_name)


def extract_data_with_pdfplumber_preprocessing(
    pdf_path: str, model_name: str = None, prompt_version: str = "v1.2.2"
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Enhanced LLM extraction using pdfplumber table preprocessing.

    Args:
        pdf_path: Path to PDF file
        model_name: Optional model override

    Returns:
        Tuple of (extracted records, call_id)
    """
    if not PREPROCESSOR_AVAILABLE:
        print("⚠️ pdfplumber preprocessor not available, falling back to raw text")
        text_content = extract_text_from_pdf(pdf_path)
        return extract_data_from_text(text_content, model_name=model_name)

    try:
        print("🔍 Running pdfplumber table preprocessing...")
        preprocess_result = process_surveillance_bulletin(pdf_path)

        if preprocess_result["success"]:
            records = preprocess_result["surveillance_data"]["records"]
            preprocessing_id = preprocess_result.get(
                "preprocessing_log_id"
            )  # Get ID from new system
            print(f"✅ pdfplumber preprocessing successful: {records} records")
            print(f"📝 Preprocessing logged with ID: {preprocessing_id}")

            # Get DataFrame directly from preprocessing result (no intermediate CSV)
            table_data = preprocess_result["surveillance_data"]["data"]

            # Skip complex narrative linking and go directly to LLM filtering
            print("🧠 Sending table data to LLM for garbage filtering...")

            # Format table data for LLM processing with v1.2.2 prompt
            structured_inputs = {
                "table_data": table_data.to_dict("records"),
                "narrative_corrections": [],  # No corrections needed for filtering
                "narrative_text": "",  # Not needed for filtering
            }

            # Process with LLM using v1.2.2 prompt for intelligent filtering
            extracted_data, call_id = extract_data_from_structured_content(
                structured_inputs,
                model_name=model_name,
                prompt_type="health_data_extraction",
                preprocessing_id=preprocessing_id,  # Pass preprocessing ID for linking
            )

            # Get prompt metadata for CSV saving
            from src.prompt_manager import PromptManager

            prompt_manager = PromptManager()
            prompt_metadata = prompt_manager.get_prompt_version(
                "health_data_extraction", prompt_version
            )

            print(f"✅ LLM filtering completed: {len(extracted_data)} records remain")

            # Save final results as CSV with proper naming convention
            if extracted_data:
                import pandas as pd

                from src.config import Config

                df = pd.DataFrame(extracted_data)

                # Get model info for filename - use the actual model that was used
                if model_name:
                    model_for_filename = model_name.replace("/", "_").replace("-", "_")
                else:
                    # Fallback to config if no model specified
                    config = Config.get_llm_client_config()
                    model_for_filename = (
                        config["model"].replace("/", "_").replace("-", "_")
                    )

                # Get prompt version from metadata or default
                prompt_version = prompt_metadata.get("version", "v1.2.2")

                output_dir = Config.OUTPUTS_DIR

                if call_id:
                    # Standard naming with call_id
                    tagged_path = (
                        output_dir
                        / f"extraction_{call_id}_prompt_{prompt_version}_model_{model_for_filename}.csv"
                    )

                    # ALSO save original preprocessing data for comparison
                    preprocessing_path = (
                        output_dir / f"preprocessing_{call_id}_original_data.csv"
                    )
                    table_data.to_csv(preprocessing_path, index=False)

                    print(f"📊 FILTERING SUMMARY:")
                    print(
                        f"   Original preprocessing: {len(table_data)} records → {preprocessing_path}"
                    )
                    print(f"   LLM filtered output: {len(df)} records → {tagged_path}")
                    print(
                        f"   Records removed: {len(table_data) - len(df)} (potential garbage entries)"
                    )

                else:
                    # Fallback naming without call_id
                    import time

                    timestamp = int(time.time())
                    tagged_path = (
                        output_dir
                        / f"extraction_fallback_{timestamp}_prompt_{prompt_version}_model_{model_for_filename}.csv"
                    )

                df.to_csv(tagged_path, index=False)
                print(f"✅ Saved final LLM results to: {tagged_path}")
                if call_id:
                    print(f"💡 Linked to database record ID: {call_id}")
                print(f"💡 Tagged with prompt version: {prompt_version}")
                print(f"💡 Tagged with model: {model_for_filename}")

            return extracted_data, call_id

        else:
            raise Exception(
                f"pdfplumber preprocessing failed: {preprocess_result.get('error', 'Unknown error')}"
            )

    except Exception as e:
        print(f"❌ pdfplumber preprocessing error: {e}")
        raise Exception(f"Preprocessing failed: {e}")


def apply_narrative_corrections(
    table_df: pd.DataFrame, corrections: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Apply narrative corrections to the table data.

    Args:
        table_df: Original table DataFrame
        corrections: List of corrections with row keys and field updates

    Returns:
        Updated DataFrame with corrections applied
    """
    if not corrections:
        return table_df

    # Make a copy to avoid modifying the original
    updated_df = table_df.copy()

    for correction in corrections:
        if (
            correction.get("confidence", 0) < 0.6
        ):  # Only apply high-confidence corrections
            continue

        row_key = correction["row_key"]
        field = correction["field"]
        new_value = correction["new_value"]

        # Find the matching row(s)
        country, event = row_key
        mask = (updated_df["Country"] == country) & (updated_df["Event"] == event)

        if mask.any():
            old_value = correction.get("old_value", "unknown")
            print(
                f"   🔧 Applying correction: {country} {event} {field}: {old_value} → {new_value}"
            )
            updated_df.loc[mask, field] = new_value

    return updated_df


def extract_text_blocks_for_linking(pdf_path: str) -> List[Dict]:
    """Extract text blocks for narrative linking analysis."""
    text_blocks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            page_text = page.extract_text()
            if page_text:
                # Split into paragraphs for analysis
                paragraphs = page_text.split("\n\n")
                for para in paragraphs:
                    if para.strip():
                        text_blocks.append(
                            {
                                "text": para.strip(),
                                "page": page_num,
                                "type": "paragraph",
                            }
                        )
    return text_blocks


def extract_data_from_structured_content(
    structured_inputs,
    model_name: str = None,
    prompt_type: str = None,
    preprocessing_id: int = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Extract data from structured inputs (table + narrative corrections).

    Args:
        structured_inputs: Dict with table_data, narrative_corrections, narrative_text OR legacy string
        model_name: Optional model override
        prompt_type: Optional prompt type override

    Returns:
        Tuple of (extracted records list, database call_id)
    """
    # Initialize prompt management and logging
    prompt_manager = PromptManager()
    prompt_logger = PromptLogger()

    # Initialize LLM client
    if model_name:
        llm_client = LLMClient.create_client_for_model(model_name)
    else:
        llm_client = LLMClient()

    # Handle both new dict format and legacy string format
    if isinstance(structured_inputs, dict):
        # New format with separate inputs
        prompt_kwargs = structured_inputs
    else:
        # Legacy format - convert to text_content
        prompt_kwargs = {"text_content": structured_inputs}

    # Use specialized structured data prompt or fallback
    prompt_type_to_use = prompt_type or "health_data_extraction"
    try:
        system_prompt, user_prompt, prompt_metadata, prompt_for_logging = (
            prompt_manager.build_prompt(prompt_type=prompt_type_to_use, **prompt_kwargs)
        )
    except Exception as e:
        # Fallback to regular prompt if structured prompt not found
        print(f"⚠️ Prompt '{prompt_type_to_use}' not found: {e}")

        # Convert to fallback format
        if isinstance(structured_inputs, dict):
            fallback_content = f"""
TABLE DATA:
{structured_inputs.get('table_data', 'No table data')}

NARRATIVE CORRECTIONS:
{structured_inputs.get('narrative_corrections', 'No corrections')}

NARRATIVE CONTEXT:
{structured_inputs.get('narrative_text', 'No narrative context')}
"""
        else:
            fallback_content = structured_inputs

        system_prompt, user_prompt, prompt_metadata, prompt_for_logging = (
            prompt_manager.build_prompt(
                prompt_type="health_data_extraction", text_content=fallback_content
            )
        )

    # Call LLM
    import time

    start_time = time.time()

    # GPT-5 and Grok-4: Must have enough tokens for reasoning + response
    actual_model_name = llm_client.get_model_info().get("model_name", "")
    is_reasoning_model = (
        "gpt-5" in actual_model_name.lower() or "grok" in actual_model_name.lower()
    )
    if is_reasoning_model:
        max_tokens = 100000  # Higher limit for reasoning models
    else:
        max_tokens = 16384  # Standard limit for other models

    raw_response, api_metadata = llm_client.create_chat_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        temperature=0,
    )

    execution_time = time.time() - start_time
    print(
        f"✅ Structured LLM call successful: "
        f"{len(raw_response)} characters received"
    )
    print(f"Execution time: {execution_time:.2f} seconds")

    # Log the call
    call_id = prompt_logger.log_llm_call(
        prompt_metadata=prompt_metadata,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        raw_response=raw_response,
        parsed_success=True,  # Will update if parsing fails
        execution_time_seconds=execution_time,
        model_name=llm_client.get_model_info().get("model_name", "unknown"),
        model_parameters=api_metadata.get("model_parameters", {}),
        preprocessing_id=preprocessing_id,  # Link to tabular preprocessing
    )

    # Parse response
    try:
        extracted_data = parse_extracted_data(raw_response)
        return extracted_data, call_id
    except Exception as e:
        print(f"Error parsing structured LLM response: {e}")
        # Update log to reflect parsing failure
        prompt_logger.update_log_entry(call_id, parsed_success=False)
        return [], call_id


def convert_table_to_final_format(
    table_df: pd.DataFrame, corrections: List = None
) -> List[Dict[str, Any]]:
    """
    Convert corrected table DataFrame directly to final JSON format.
    This avoids sending large datasets back to the LLM.
    """
    corrections = corrections or []

    final_data = []

    for _, row in table_df.iterrows():
        record = {
            "Country": str(row.get("Country", "N/A")).strip(),
            "Event": str(row.get("Event", "N/A")).strip(),
            "Grade": str(row.get("Grade", "N/A")).strip(),
            "Date_Notified": str(row.get("Date_Notified", "N/A")).strip(),
            "Start_Date": str(row.get("Start_Date", "N/A")).strip(),
            "End_Date": str(row.get("End_Date", "N/A")).strip(),
            "Total_Cases": str(row.get("Total_Cases", "N/A")).strip(),
            "Confirmed_Cases": str(row.get("Confirmed_Cases", "N/A")).strip(),
            "Deaths": str(row.get("Deaths", "N/A")).strip(),
            "CFR": str(row.get("CFR", "N/A")).strip(),
        }

        # Clean up any NaN values
        for key, value in record.items():
            if value in ["nan", "None", ""]:
                record[key] = "N/A"

        final_data.append(record)

    return final_data


def format_table_with_narrative_context(
    table_df: pd.DataFrame, narrative_text: str, corrections: List = None
) -> Dict[str, str]:
    """
    Format table data and narrative corrections as separate structured inputs.
    Returns dict with table_data, narrative_corrections, and narrative_text.
    """
    corrections = corrections or []

    # Convert table to JSON with row IDs for linking
    table_records = []
    for idx, row in table_df.iterrows():
        record = {
            "table_row_id": int(idx),
            "Country": str(row.get("Country", "N/A")).strip(),
            "Event": str(row.get("Event", "N/A")).strip(),
            "Grade": str(row.get("Grade", "N/A")).strip(),
            "Date_Notified": str(row.get("Date_Notified", "N/A")).strip(),
            "Start_Date": str(row.get("Start_Date", "N/A")).strip(),
            "End_Date": str(row.get("End_Date", "N/A")).strip(),
            "Total_Cases": str(row.get("Total_Cases", "N/A")).strip(),
            "Confirmed_Cases": str(row.get("Confirmed_Cases", "N/A")).strip(),
            "Deaths": str(row.get("Deaths", "N/A")).strip(),
            "CFR": str(row.get("CFR", "N/A")).strip(),
        }
        # Clean up any pandas NaN values
        for key, value in record.items():
            if value in ["nan", "None", ""]:
                record[key] = "N/A"
        table_records.append(record)

    # Format narrative corrections with table linking
    narrative_corrections_formatted = []
    for correction in corrections:
        country = correction.get("country", "Unknown")
        event = correction.get("event", "Unknown")

        # Find the table_row_id that matches this correction
        matching_rows = table_df[
            (table_df["Country"].str.contains(country, na=False, case=False))
            & (table_df["Event"].str.contains(event, na=False, case=False))
        ]

        table_row_id = int(matching_rows.index[0]) if len(matching_rows) > 0 else None

        correction_formatted = {
            "table_row_id": table_row_id,
            "field": correction.get("field", "Unknown"),
            "old_value": correction.get("old_value", "Unknown"),
            "new_value": correction.get("new_value", "Unknown"),
            "confidence": correction.get("confidence", 0),
            "explanation": correction.get("explanation", "N/A"),
            "narrative_evidence": correction.get("explanation", "N/A")[:200] + "...",
        }
        narrative_corrections_formatted.append(correction_formatted)

    # Extract relevant narrative segments
    narrative_segments = extract_relevant_narrative_segments(narrative_text, table_df)

    return {
        "table_data": json.dumps(table_records, indent=2),
        "narrative_corrections": json.dumps(narrative_corrections_formatted, indent=2),
        "narrative_text": narrative_segments,
    }


def extract_relevant_narrative_segments(
    narrative_text: str, table_df: pd.DataFrame
) -> str:
    """
    Extract narrative text segments that are most relevant to the table data.
    This focuses the LLM on text that likely contains corrections or clarifications.
    """
    segments = []

    # Get unique countries and events from the table
    countries = table_df["Country"].unique()[:10]  # Top 10 countries
    events = table_df["Event"].unique()[:10]  # Top 10 events

    # Split narrative into sentences/paragraphs
    text_lines = narrative_text.split("\n")

    # Find segments that mention table entities
    for line in text_lines:
        line_clean = line.strip()
        if len(line_clean) < 20:  # Skip very short lines
            continue

        # Check if line mentions any countries or events
        line_lower = line_clean.lower()
        relevant = False

        for country in countries:
            if country.lower() in line_lower:
                relevant = True
                break

        if not relevant:
            for event in events:
                if event.lower() in line_lower:
                    relevant = True
                    break

        # Also include lines with numbers (likely corrections)
        if not relevant and any(char.isdigit() for char in line_clean):
            if any(
                word in line_lower
                for word in ["cases", "deaths", "reported", "confirmed"]
            ):
                relevant = True

        if relevant:
            segments.append(line_clean)

        # Limit to prevent overwhelming the LLM
        if len(segments) >= 15:
            break

    return (
        "\n".join(segments[:15]) if segments else "No specific narrative context found."
    )


def format_table_data_for_llm(table_df: pd.DataFrame, corrections: List = None) -> str:
    """Format extracted table DataFrame for LLM input with narrative corrections."""

    corrections = corrections or []

    # Use a much smaller sample to avoid overwhelming the API
    sample_size = min(10, len(table_df))  # Only show first 10 records
    sample_df = table_df.head(sample_size)

    structured_input = f"""
WHO Health Surveillance Data - Structured Table Extracted via pdfplumber

EXTRACTED TABLE SUMMARY:
- Total Records: {len(table_df)}
- Countries: {table_df['Country'].nunique()}
- Event Types: {table_df['Event'].nunique()}
- Sample showing first {sample_size} records:

STRUCTURED DATA (SAMPLE):
{sample_df.to_string(index=False)}

FULL DATA PROCESSING INSTRUCTIONS:
- Process ALL {len(table_df)} records from the complete dataset
- The above sample shows the data structure and format
- Apply the same validation to all records in the full dataset

NARRATIVE CORRECTIONS APPLIED:
- Total corrections identified: {len(corrections)}
"""

    if corrections:
        structured_input += "\n- Correction details:\n"
        for correction in corrections[:5]:  # Limit to first 5
            row_key = correction.get("row_key", ("Unknown", "Unknown"))
            field = correction.get("field", "Unknown")
            old_val = correction.get("old_value", "Unknown")
            new_val = correction.get("new_value", "Unknown")
            confidence = correction.get("confidence", 0)
            structured_input += f"  * {row_key[0]} {row_key[1]} - {field}: {old_val} → {new_val} (confidence: {confidence:.2f})\n"

    structured_input += """

Instructions: The above shows a sample of the pre-extracted data from pdfplumber.
Process the COMPLETE dataset with narrative corrections applied. Return JSON for ALL records.
"""

    return structured_input


def parse_extracted_data(response: str) -> List[Dict[str, Any]]:
    """Parse LLM response into structured data."""
    import json

    try:
        # Try to parse as JSON directly
        if response.strip().startswith("["):
            return json.loads(response.strip())

        # Look for JSON array in response
        import re

        json_match = re.search(r"\[.*\]", response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())

        # If no JSON found, return empty list
        print("⚠️ No valid JSON found in LLM response")
        return []

    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing failed: {e}")
        return []


def process_pdf_with_text_extraction(
    pdf_path: str, output_csv_path: str = None, model_name: str = None
) -> pd.DataFrame:
    """
    Complete pipeline: extract text from PDF, process with LLM, return raw DataFrame.
    Post-processing should be applied separately during experimental phase.

    Args:
        pdf_path: Path to PDF file
        output_csv_path: Optional output path
        model_name: Optional model override (e.g., "anthropic/claude-3.5-sonnet")
    """
    print("=== Starting Text-Based PDF Extraction ===")

    # Step 1: Extract text from PDF
    text_content = extract_text_from_pdf(pdf_path)

    # Step 2: Process text with LLM
    extracted_data, call_id = extract_data_from_text(
        text_content, model_name=model_name
    )

    # Step 3: Convert to DataFrame (RAW OUTPUT)
    df = pd.DataFrame(extracted_data)

    # Add source document tracking
    if len(df) > 0:
        df["SourceDocument"] = os.path.basename(pdf_path)
        print(f"📎 Added SourceDocument: {os.path.basename(pdf_path)}")

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

            # Determine model name for filename
            if model_name:
                # Use the specific model provided
                model_for_filename = model_name.replace("/", "_").replace("-", "_")
            else:
                # Use default model from config
                from src.config import Config

                config = Config.get_llm_client_config()
                model_for_filename = config["model"].replace("/", "_").replace("-", "_")

            # Use database call_id for better linking between CSV and database
            # New format: extraction_<call_id>_prompt_<version>_model_<model>.csv
            output_dir = Path(output_csv_path).parent
            tagged_path = (
                output_dir
                / f"extraction_{call_id}_prompt_{prompt_version}_model_{model_for_filename}.csv"
            )

            df.to_csv(tagged_path, index=False)
            print(f"Saved RAW LLM results to: {tagged_path}")
            print(f"💡 Linked to database record ID: {call_id}")
            print(f"💡 Tagged with prompt version: {prompt_version}")
            print(f"💡 Tagged with model: {model_for_filename}")
            print(
                "💡 Apply post-processing separately using apply_post_processing_pipeline()"
            )
    else:
        print("WARNING: No data extracted!")

    return df


def setup_prompt_version(prompt_version: str = None) -> str:
    """
    Setup prompt version, auto-importing from markdown if needed.

    Args:
        prompt_version: Specific version to use, or None for current

    Returns:
        The actual prompt version being used

    Raises:
        FileNotFoundError: If specified prompt version doesn't exist
    """
    prompt_manager = PromptManager()

    if prompt_version:
        # Check if this version exists in JSON system and has content
        try:
            prompt_data = prompt_manager.get_prompt_version(
                "health_data_extraction", prompt_version
            )
            # Check if the prompt has actual content (not empty)
            if prompt_data.get("system_prompt") or prompt_data.get(
                "user_prompt_template"
            ):
                print(f"✅ Using existing prompt version: {prompt_version}")
                # Set as current for this run
                prompt_manager.set_current_version(
                    "health_data_extraction", prompt_version
                )
                return prompt_version
            else:
                # JSON exists but is empty - force re-import
                print(
                    f"⚠️  JSON for {prompt_version} exists but is empty, re-importing from markdown..."
                )
                raise ValueError("Empty prompt content - will re-import")
        except (FileNotFoundError, ValueError):
            # Try to auto-import from markdown
            markdown_path = f"prompts/markdown/health_data_extraction/health_data_extraction_{prompt_version}.md"
            if os.path.exists(markdown_path):
                print(f"📥 Auto-importing prompt {prompt_version} from markdown...")
                imported_path = prompt_manager.create_prompt_from_markdown(
                    "health_data_extraction", markdown_path
                )
                print(f"✅ Imported prompt version: {prompt_version}")
                # Set as current for this run
                prompt_manager.set_current_version(
                    "health_data_extraction", prompt_version
                )
                return prompt_version
            else:
                # Neither JSON nor markdown exists - fail clearly
                print(f"❌ Prompt version {prompt_version} not found!")
                print(
                    f"   Checked JSON: prompts/health_data_extraction/{prompt_version}.json"
                )
                print(f"   Checked Markdown: {markdown_path}")
                raise FileNotFoundError(
                    f"Prompt version {prompt_version} not found in JSON or markdown"
                )
    else:
        # Use current prompt - this should always work
        current_prompt = prompt_manager.get_current_prompt("health_data_extraction")
        return current_prompt["version"]


def generate_call_id() -> str:
    """Generate a unique call ID for this extraction run."""
    return f"{int(time.time())}"


def extract_data_with_table_focused_preprocessing(
    pdf_path: str,
    model_name: Optional[str] = None,
    prompt_version: str = "v1.3.0",
) -> Tuple[pd.DataFrame, str]:
    """
    Table-focused extraction using WHO surveillance extractor + LLM correction.
    Modified to process ALL pages instead of hardcoded range for better coverage.

    1. Extract surveillance table using modified WHO extractor (all pages)
    2. Log preprocessing to tabular_preprocessing_logs
    3. Apply LLM corrections using prompt v1.3.0
    4. Return corrected DataFrame
    """
    call_id = generate_call_id()

    print(f"🔍 Running table-focused WHO surveillance extraction (ALL PAGES)...")

    # Step 1: Extract surveillance table from ALL pages
    from src.pre_extraction.who_surveillance_extractor import WHOSurveillanceExtractor
    from src.tabular_preprocessing_logger import TabularPreprocessingLogger

    # Create extractor and extract from ALL pages (modified to process all pages)
    extractor = WHOSurveillanceExtractor()

    # Extract surveillance data - now processes ALL pages automatically
    surveillance_df = extractor.extract_from_pdf(pdf_path, Path(pdf_path).name)

    print(f"✅ Extracted {len(surveillance_df)} surveillance records")

    # Step 2: Log preprocessing to organized system
    logger = TabularPreprocessingLogger()
    preprocessing_result = logger.log_tabular_preprocessing(
        pdf_path=pdf_path,
        preprocessing_method="table-focused",
        surveillance_df=surveillance_df,
        extraction_metadata={
            "extractor": "WHOSurveillanceExtractor",
            "records_found": len(surveillance_df),
            "pages_processed": "9-16",
        },
        execution_time_seconds=1.0,  # Placeholder - would need actual timing
        success=True,
    )

    preprocessing_id = preprocessing_result
    print(f"📊 Logged preprocessing with ID: {preprocessing_id}")

    # Step 3: Prepare for LLM correction
    if model_name and len(surveillance_df) > 0:
        print(f"🤖 Applying LLM corrections with model: {model_name}")

        # Convert DataFrame to JSON format for LLM
        records_json = surveillance_df.to_dict("records")

        # Convert OpenRouter model name to OpenAI format for direct API calls
        openai_model_name = model_name
        if model_name.startswith("openai/"):
            openai_model_name = model_name.replace("openai/", "")

        # Apply LLM corrections using prompt v1.3.0 with preprocessing_id
        corrected_df, corrections_json = apply_llm_corrections_v1_3_0(
            records_json, openai_model_name, call_id, prompt_version, preprocessing_id
        )

        print(
            f"✅ LLM corrections applied: {len(corrections_json.get('corrections', []))} changes"
        )

        # Save the corrected data using run ID for consistent naming
        output_path = save_corrected_surveillance_data(
            corrected_df, str(preprocessing_id), prompt_version, model_name
        )

        return corrected_df, call_id
    else:
        print("⚠️ No model specified or no records extracted, skipping LLM correction")

        # Save the raw surveillance data (but don't log to prompt_logs since no LLM used)
        output_path = save_raw_surveillance_data(surveillance_df, call_id)

        return surveillance_df, call_id


def apply_llm_corrections_v1_3_0(
    records_json: List[Dict],
    model_name: str,
    call_id: str,
    prompt_version: str = "v1.3.0",
    preprocessing_id: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Apply LLM corrections using the specified prompt version."""

    try:
        import openai

        from src.config import Config
        from src.prompt_manager import PromptManager

        # Load the prompt using the prompt manager (same as rest of pipeline)
        prompt_manager = PromptManager()
        prompt_data = prompt_manager.get_prompt_version(
            "health_data_extraction", prompt_version
        )
        prompt_template = prompt_data["user_prompt_template"]

        # Create the prompt with data
        prompt = f"""{prompt_template}

## Input Data
Please review and correct the following WHO surveillance records:

```json
{json.dumps(records_json, indent=2)}
```

Return only the JSON correction object as specified in the prompt."""

        # Initialize client based on model
        if "gpt" in model_name.lower():
            client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        else:
            print(f"⚠️ Model {model_name} not supported for corrections yet")
            return pd.DataFrame(records_json), {"corrections": []}

        # Make API call
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a WHO surveillance data quality expert. Return only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=4000,
        )

        # Parse response
        response_text = response.choices[0].message.content.strip()

        # Clean JSON response
        if response_text.startswith("```json"):
            response_text = (
                response_text.replace("```json", "").replace("```", "").strip()
            )
        elif response_text.startswith("```"):
            response_text = response_text.replace("```", "").strip()

        corrections_json = json.loads(response_text)

        # Log the LLM call to prompt_logs with the same run_id
        from src.prompt_logger import PromptLogger

        prompt_logger = PromptLogger()
        llm_call_id = prompt_logger.log_llm_call_with_run_id(
            run_id=preprocessing_id,  # Use same run_id as preprocessing
            prompt_metadata={
                "prompt_type": "health_data_extraction",
                "version": prompt_version,
                "correction_type": "surveillance_data_quality",
                "records_count": len(records_json),
            },
            model_name=model_name,
            model_parameters={"temperature": 0.1, "max_tokens": 4000},
            system_prompt="You are a WHO surveillance data quality expert. Return only valid JSON.",
            user_prompt=prompt,
            raw_response=response_text,
            parsed_success=True,
            records_extracted=len(corrections_json.get("corrections", [])),
            parsing_errors=None,
            execution_time_seconds=1.0,  # Placeholder
        )

        print(f"📝 Logged LLM call with same run ID: {llm_call_id}")

        # Save raw LLM response with run ID
        save_llm_corrections_json(corrections_json, str(preprocessing_id))

        # Apply corrections to DataFrame
        corrected_df = apply_corrections_to_dataframe(
            pd.DataFrame(records_json), corrections_json
        )

        return corrected_df, corrections_json

    except Exception as e:
        print(f"❌ Error in LLM corrections: {e}")
        return pd.DataFrame(records_json), {"corrections": []}


def apply_corrections_to_dataframe(
    df: pd.DataFrame, corrections_json: Dict
) -> pd.DataFrame:
    """Apply the LLM corrections to the DataFrame."""
    corrected_df = df.copy()

    for correction in corrections_json.get("corrections", []):
        record_index = correction["record_index"]
        field = correction["field"]
        new_value = correction["new_value"]

        if record_index < len(corrected_df):
            corrected_df.iloc[record_index, corrected_df.columns.get_loc(field)] = (
                new_value
            )
            print(f"✅ Applied correction: Row {record_index}, {field} → {new_value}")

    return corrected_df


def save_llm_corrections_json(corrections_json: Dict, run_id: str) -> str:
    """Save the raw LLM corrections JSON for analysis with organized structure."""
    # Save to organized outputs directory for LLM extraction metadata
    output_dir = Path("outputs/llm_extractions/metadata")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"corrections_{run_id}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corrections_json, f, indent=2, ensure_ascii=False)

    print(f"💾 LLM corrections saved: {output_path}")
    return str(output_path)


def save_corrected_surveillance_data(
    df: pd.DataFrame, run_id: str, prompt_version: str, model_name: str
) -> str:
    """Save corrected surveillance data with organized file structure."""
    model_for_filename = model_name.replace("/", "_").replace("-", "_")

    # Save to organized outputs directory
    output_dir = Path("outputs/llm_extractions/corrected")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir
        / f"corrected_{run_id}_prompt_{prompt_version}_model_{model_for_filename}.csv"
    )

    df.to_csv(output_path, index=False)
    print(f"💾 Corrected surveillance data saved: {output_path}")
    return str(output_path)


def save_raw_surveillance_data(df: pd.DataFrame, call_id: str) -> str:
    """Save raw surveillance data without corrections."""
    output_path = Config.OUTPUTS_DIR / f"surveillance_raw_{call_id}.csv"

    df.to_csv(output_path, index=False)
    print(f"💾 Raw surveillance data saved: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract health data from PDF using configurable LLM"
    )
    parser.add_argument(
        "--prompt-version",
        "-p",
        type=str,
        help="Specific prompt version to use (e.g., v1.1.1)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Model to use (e.g., 'claude-3.5-sonnet', 'gpt-4o', 'gemini-pro')",
    )
    parser.add_argument("--pdf-path", type=str, help="Path to PDF file to process")
    parser.add_argument("--output-path", type=str, help="Base output path for results")

    # Preprocessor option
    parser.add_argument(
        "--preprocessor",
        type=str,
        choices=["pdfplumber", "blank-treatment", "table-focused", "none-pdf-upload"],
        help="Use preprocessing before LLM extraction (pdfplumber: table extraction, blank-treatment: standardize blank fields, table-focused: WHO surveillance extraction + correction, none-pdf-upload: direct PDF upload to LLM without text extraction)",
    )

    args = parser.parse_args()

    # Setup prompt version (with auto-import if needed)
    prompt_version = setup_prompt_version(args.prompt_version)

    # Handle model selection
    model_name = args.model
    if model_name:
        from src.llm_client import get_model_identifier

        model_name = get_model_identifier(model_name)
        print(f"🤖 Using model: {model_name}")
    else:
        print("🤖 Using default model configuration")

    # Test the text-based extraction
    pdf_path = args.pdf_path or str(
        Path(Config.LOCAL_DIR_BASE)
        / "Cholera - General"
        / "WHO_bulletins_historical"
        / "Week_28__7_-_13_July_2025.pdf"
    )

    # Base output path - will be automatically tagged with prompt version
    base_output_path = args.output_path or str(
        Config.OUTPUTS_DIR / "text_extracted_data"
    )

    print(f"🎯 Running extraction with prompt version: {prompt_version}")

    # Generate the actual filename that will be used
    # (matching process_pdf_with_text_extraction logic)
    if model_name:
        model_for_filename = model_name.replace("/", "_").replace("-", "_")
    else:
        model_for_filename = "default"

    # Note: Actual filename will be extraction_{call_id}_prompt_{version}_model_{model}.csv
    # for pdfplumber preprocessing (modern format)
    if args.preprocessor == "pdfplumber":
        print(
            f"📁 Output will be saved with format: extraction_{{call_id}}_prompt_{prompt_version}_model_{model_for_filename}.csv"
        )
    else:
        output_name = (
            f"{base_output_path}_prompt_{prompt_version}_model_"
            f"{model_for_filename}.csv"
        )
        print(f"📁 Output will be saved as: {output_name}")

    if os.path.exists(pdf_path):
        # Use clean preprocessor flag
        if args.preprocessor == "pdfplumber":
            print("🔍 Running pdfplumber preprocessing + LLM extraction...")
            extracted_data, call_id = extract_data_with_pdfplumber_preprocessing(
                pdf_path, model_name=model_name, prompt_version=prompt_version
            )
            print(
                f"📁 Final output saved as: extraction_{call_id}_prompt_{prompt_version}_model_{model_for_filename}.csv"
            )
        elif args.preprocessor == "blank-treatment":
            print("🔧 Running blank field treatment + LLM extraction...")
            extracted_data, call_id = extract_data_with_blank_treatment(
                pdf_path, model_name=model_name
            )
            print(
                f"📁 Final output saved as: extraction_{call_id}_prompt_{prompt_version}_model_{model_for_filename}_blank_treatment.csv"
            )
        elif args.preprocessor == "table-focused":
            print(
                "🎯 Running table-focused WHO surveillance extraction + LLM correction..."
            )
            extracted_data, call_id = extract_data_with_table_focused_preprocessing(
                pdf_path, model_name=model_name, prompt_version=prompt_version
            )
            print(
                f"📁 Final output saved as: surveillance_corrected_{call_id}_prompt_{prompt_version}_model_{model_for_filename}.csv"
            )
        elif args.preprocessor == "none-pdf-upload":
            print("📤 Running direct PDF upload extraction (no text preprocessing)...")
            from src.pdf_upload_extract import extract_data_with_pdf_upload

            extracted_data, call_id = extract_data_with_pdf_upload(
                pdf_path, model_name=model_name, prompt_version=prompt_version
            )

            print(f"✅ PDF upload extraction completed: {len(extracted_data)} records")

            # Save output using the same format as other methods
            if extracted_data:
                import pandas as pd

                df = pd.DataFrame(extracted_data)

                output_path = (
                    Config.OUTPUTS_DIR
                    / f"extraction_{call_id}_prompt_{prompt_version}_model_{model_for_filename}_pdf_upload.csv"
                )
                df.to_csv(output_path, index=False)
                print(f"📁 Final output saved as: {output_path.name}")
                print(f"📊 Records extracted: {len(df)}")
            else:
                print("❌ No data extracted")
        else:
            print("📝 Running standard text extraction...")
            df = process_pdf_with_text_extraction(
                pdf_path, output_name, model_name=model_name
            )
            print(f"\n=== FINAL RESULTS ===")
            print(f"Total records extracted: {len(df)}")
    else:
        print(f"PDF file not found: {pdf_path}")
