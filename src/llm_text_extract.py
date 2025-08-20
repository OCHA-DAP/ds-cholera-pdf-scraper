"""
LLM-based extraction from PDF text using configurable LLM providers.
This module processes extracted PDF text and supports OpenAI and OpenRouter.
"""

import argparse
import json
import os
import time
from typing import Any, Dict, List

import pandas as pd
import pdfplumber

from accuracy_evaluator import evaluate_and_log_accuracy
from llm_client import LLMClient
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


def extract_data_from_text(
    text_content: str, model_name: str = None
) -> List[Dict[str, Any]]:
    """
    Use LLM to extract structured data from PDF text.
    Now supports both OpenAI and OpenRouter with integrated prompt logging.

    Args:
        text_content: PDF text content to process
        model_name: Optional model override (e.g., "anthropic/claude-3.5-sonnet")

    Returns:
        List of extracted records
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
                    raise e

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
                base_path = "/Users/zackarno/Documents/CHD/repos/ds-cholera-pdf-scraper"

                # Create predictable output path based on prompt version and model for overwriting
                prompt_version = prompt_metadata.get("version", "unknown")
                model_safe = actual_model_name.replace("/", "_").replace("-", "_")
                accuracy_output_path = (
                    f"{base_path}/logs/accuracy/{prompt_version}_{model_safe}"
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

        return extracted_data

    except Exception as e:
        execution_time = time.time() - start_time
        error_message = str(e)

        # Log failed call
        call_id = prompt_logger.log_llm_call(
            prompt_metadata=prompt_metadata,
            model_name=llm_client.model_name,
            model_parameters={"max_tokens": 16384, "temperature": 0},
            system_prompt=system_prompt,
            user_prompt=prompt_for_logging,
            raw_response=f"ERROR: {error_message}",
            parsed_success=False,
            records_extracted=0,
            parsing_errors=error_message,
            execution_time_seconds=execution_time,
            custom_metrics={"text_length_chars": len(text_content)},
        )

        print(f"Error during LLM extraction: {e}")
        print(f"🚨 This was an API-level failure (not a parsing issue)")
        raise


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
    extracted_data = extract_data_from_text(text_content, model_name=model_name)

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
                from config import Config

                config = Config.get_llm_client_config()
                model_for_filename = config["model"].replace("/", "_").replace("-", "_")

            # Add both prompt version and model to filename
            if output_csv_path.endswith(".csv"):
                base_path = output_csv_path[:-4]
                tagged_path = f"{base_path}_prompt_{prompt_version}_model_{model_for_filename}.csv"
            else:
                tagged_path = f"{output_csv_path}_prompt_{prompt_version}_model_{model_for_filename}.csv"

            df.to_csv(tagged_path, index=False)
            print(f"Saved RAW LLM results to: {tagged_path}")
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
        # Check if this version exists in JSON system
        try:
            prompt_manager.get_prompt_version("health_data_extraction", prompt_version)
            print(f"✅ Using existing prompt version: {prompt_version}")
            # Set as current for this run
            prompt_manager.set_current_version("health_data_extraction", prompt_version)
            return prompt_version
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

    args = parser.parse_args()

    # Setup prompt version (with auto-import if needed)
    prompt_version = setup_prompt_version(args.prompt_version)

    # Handle model selection
    model_name = args.model
    if model_name:
        from llm_client import get_model_identifier

        model_name = get_model_identifier(model_name)
        print(f"🤖 Using model: {model_name}")
    else:
        print("🤖 Using default model configuration")

    # Test the text-based extraction
    pdf_path = (
        args.pdf_path
        or "/Users/zackarno/Library/CloudStorage/GoogleDrive-Zachary.arno@humdata.org/Shared drives/Data Science/CERF Anticipatory Action/Cholera - General/WHO_bulletins_historical/Week_28__7_-_13_July_2025.pdf"
    )

    # Base output path - will be automatically tagged with prompt version
    base_output_path = (
        args.output_path
        or "/Users/zackarno/Documents/CHD/repos/ds-cholera-pdf-scraper/outputs/text_extracted_data"
    )

    print(f"🎯 Running extraction with prompt version: {prompt_version}")

    # Generate the actual filename that will be used
    # (matching process_pdf_with_text_extraction logic)
    if model_name:
        model_for_filename = model_name.replace("/", "_").replace("-", "_")
    else:
        model_for_filename = "default"

    output_name = (
        f"{base_output_path}_prompt_{prompt_version}_model_" f"{model_for_filename}.csv"
    )
    print(f"📁 Output will be saved as: {output_name}")

    if os.path.exists(pdf_path):
        df = process_pdf_with_text_extraction(
            pdf_path, f"{base_output_path}.csv", model_name=model_name
        )
        print("\n=== FINAL RESULTS ===")
        print(f"Total records extracted: {len(df)}")
    else:
        print(f"PDF file not found: {pdf_path}")
