"""
PDF Upload extraction using OpenAI and OpenRouter native PDF support.
Integrates with existing prompt manager and versioning system.
"""

import base64
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

from src.config import Config
from src.llm_client import LLMClient
from src.prompt_logger import PromptLogger
from src.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class OpenRouterPDFExtractor:
    """Simple extractor class for OpenRouter PDF processing."""

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        self.logger = logger

    def _extract_openrouter_pdf_upload(
        self, pdf_path: str, prompt_text: str
    ) -> List[Dict[str, Any]]:
        """Extract data from PDF using OpenRouter API with file upload."""
        url = "https://openrouter.ai/api/v1/chat/completions"

        with open(pdf_path, "rb") as pdf_file:
            files = {"file": pdf_file}

            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt_text}],
            }

            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                data=data,
                files=files,
            )

            if response.status_code != 200:
                raise Exception(
                    f"OpenRouter API error: {response.status_code} - {response.text}"
                )

            # Save raw response for debugging
            with open("/tmp/openrouter_response.txt", "w") as f:
                f.write(response.text)

            response_json = response.json()

            # Check if the response is empty or None before processing
            content = (
                response_json.get("choices", [{}])[0].get("message", {}).get("content")
            )
            if not content or content.strip() == "":
                return []

            try:
                parsed_content = json.loads(content)
                return parsed_content
            except json.JSONDecodeError as e:
                # If JSON parsing fails, try to fix truncated JSON arrays
                self.logger.warning(f"JSON parsing failed: {e}")

                # Try to fix common truncation issues
                content_fixed = content.strip()

                # If it looks like a truncated array, try to close it
                if content_fixed.startswith("[") and not content_fixed.endswith("]"):
                    # Find the last complete JSON object
                    last_brace = content_fixed.rfind("}")
                    if last_brace > 0:
                        content_fixed = content_fixed[: last_brace + 1] + "]"
                        try:
                            parsed_content = json.loads(content_fixed)
                            self.logger.info(
                                f"Successfully fixed truncated JSON, recovered {len(parsed_content)} records"
                            )
                            return parsed_content
                        except json.JSONDecodeError:
                            pass

                self.logger.error(
                    f"Failed to parse or fix OpenRouter response as JSON: {e}"
                )
                self.logger.error(f"Response content: {content[:500]}...")
                return []


def extract_data_with_pdf_upload(
    pdf_path: str, model_name: str = None, prompt_version: str = None
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Extract data using native PDF upload (OpenAI or OpenRouter).
    Integrates with existing prompt manager and logging system.

    Args:
        pdf_path: Path to PDF file
        model_name: Optional model override
        prompt_version: Optional prompt version override

    Returns:
        Tuple of (extracted records list, database call_id)
    """
    # Initialize prompt management and logging
    prompt_manager = PromptManager()
    prompt_logger = PromptLogger()

    # Initialize LLM client with intelligent provider selection
    if model_name:
        llm_client = LLMClient.create_client_for_model(model_name)
    else:
        llm_client = LLMClient()

    # Get model info for routing
    model_info = llm_client.get_model_info()
    provider = model_info["provider"]
    actual_model_name = model_info["model_name"]

    print(f"🎯 Using {provider} for PDF upload with model: {actual_model_name}")

    # Build prompt using existing prompt manager
    # For PDF upload, use the specified version (should be PDF-optimized like v1.4.2)
    system_prompt, user_prompt, prompt_metadata, prompt_for_logging = (
        prompt_manager.build_prompt(
            prompt_type="health_data_extraction", version=prompt_version
        )
    )

    start_time = time.time()

    try:
        raw_response_for_logging = None  # Keep original string for logging

        if provider == "openai":
            # OpenAI file upload method
            extracted_data, api_metadata = _extract_openai_pdf_upload(
                pdf_path, llm_client, system_prompt, user_prompt
            )
            raw_response_for_logging = extracted_data  # OpenAI returns string
        elif provider == "openrouter":
            # OpenRouter PDF upload method - use exact working version
            raw_content, api_metadata = _extract_openrouter_pdf_upload(
                pdf_path, llm_client, system_prompt, user_prompt, actual_model_name
            )
            raw_response_for_logging = raw_content  # Keep original string
            # Parse the raw content to extract data
            try:
                extracted_data = json.loads(raw_content)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed: {e}")
                # Try to fix truncated JSON arrays
                content_fixed = raw_content.strip()
                if content_fixed.startswith("[") and not content_fixed.endswith("]"):
                    last_brace = content_fixed.rfind("}")
                    if last_brace > 0:
                        content_fixed = content_fixed[: last_brace + 1] + "]"
                        try:
                            extracted_data = json.loads(content_fixed)
                            logger.info(
                                f"Fixed truncated JSON, recovered {len(extracted_data)} records"
                            )
                        except json.JSONDecodeError:
                            extracted_data = []
                    else:
                        extracted_data = []
                else:
                    extracted_data = []
        else:
            raise ValueError(f"Provider {provider} does not support PDF upload")

        execution_time = time.time() - start_time

        # Parse JSON response
        parsing_errors = None
        parsed_success = False
        records = []

        try:
            # Check if extracted_data is already parsed (from OpenRouter) or needs parsing (from OpenAI)
            if isinstance(extracted_data, (list, dict)):
                # Already parsed (OpenRouter case)
                parsed_data = extracted_data
            else:
                # String that needs parsing (OpenAI case)
                response_text = extracted_data
                if "```json" in response_text:
                    start = response_text.find("```json") + 7
                    end = response_text.rfind("```")
                    response_text = response_text[start:end].strip()
                elif "```" in response_text:
                    start = response_text.find("```") + 3
                    end = response_text.rfind("```")
                    response_text = response_text[start:end].strip()

                # Parse JSON
                parsed_data = json.loads(response_text)
            if isinstance(parsed_data, dict) and "records" in parsed_data:
                records = parsed_data["records"]
            elif isinstance(parsed_data, dict) and "data" in parsed_data:
                # Handle case where LLM returns {"data": [...]}
                records = parsed_data["data"]
            elif isinstance(parsed_data, list):
                records = parsed_data
            else:
                records = [parsed_data]

            parsed_success = True
            print(f"✅ Parsed {len(records)} records from PDF upload response")

        except json.JSONDecodeError as e:
            parsing_errors = f"JSON parsing failed: {str(e)}"
            print(f"❌ {parsing_errors}")
            records = []

        # Prepare metrics for logging
        records_extracted = len(records) if parsed_success else 0
        custom_metrics = {
            "provider": provider,
            "method": "pdf_upload",
            "response_length_chars": len(extracted_data),
            "usage_tokens": api_metadata.get("usage"),
        }

        # Log the call
        call_id = prompt_logger.log_llm_call(
            prompt_metadata=prompt_metadata,
            model_name=actual_model_name,
            model_parameters=api_metadata["model_parameters"],
            system_prompt=system_prompt,
            user_prompt=prompt_for_logging,
            raw_response=raw_response_for_logging,
            parsed_success=parsed_success,
            records_extracted=records_extracted,
            parsing_errors=parsing_errors,
            execution_time_seconds=execution_time,
            custom_metrics=custom_metrics,
        )

        return records, call_id

    except Exception as e:
        execution_time = time.time() - start_time
        error_message = str(e)

        # Log failed call
        call_id = prompt_logger.log_llm_call(
            prompt_metadata=prompt_metadata,
            model_name=actual_model_name,
            model_parameters={"method": "pdf_upload"},
            system_prompt=system_prompt,
            user_prompt=prompt_for_logging,
            raw_response=f"ERROR: {error_message}",
            parsed_success=False,
            records_extracted=0,
            parsing_errors=error_message,
            execution_time_seconds=execution_time,
            custom_metrics={"provider": provider, "method": "pdf_upload"},
        )

        print(f"❌ PDF upload extraction failed: {e}")
        raise


def _extract_openai_pdf_upload(
    pdf_path: str, llm_client: LLMClient, system_prompt: str, user_prompt: str
) -> Tuple[str, Dict[str, Any]]:
    """
    Extract using OpenAI PDF upload with proper API endpoints:
    - GPT-5: Files API + Responses API (recommended approach)
    - Other models: Chat Completions with base64 inline (existing approach)
    """
    model_name = llm_client.model_name
    is_gpt5 = "gpt-5" in model_name.lower()

    print(f"📤 Preparing PDF for OpenAI: {Path(pdf_path).name}")
    print(f"🤖 Model: {model_name} (GPT-5 mode: {is_gpt5})")

    if is_gpt5:
        return _extract_openai_gpt5_responses_api(
            pdf_path, llm_client, system_prompt, user_prompt
        )
    else:
        return _extract_openai_chat_completions(
            pdf_path, llm_client, system_prompt, user_prompt
        )


def _extract_openai_gpt5_responses_api(
    pdf_path: str, llm_client: LLMClient, system_prompt: str, user_prompt: str
) -> Tuple[str, Dict[str, Any]]:
    """
    Extract using GPT-5 with Responses API (official recommended approach).
    Uses proper input_file format with file_id.
    """
    try:
        print("📁 Uploading PDF via Files API...")

        # Step 1: Upload PDF via Files API with correct purpose
        with open(pdf_path, "rb") as pdf_file:
            file_response = llm_client.client.files.create(
                file=pdf_file, purpose="user_data"  # Correct purpose for Responses API
            )

        file_id = file_response.id
        print(f"✅ PDF uploaded: {file_id}")

        # Step 2: Call Responses API with proper input_file format
        print("🧠 Calling GPT-5 via Responses API...")

        # Combine system prompt into user message for Responses API
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"

        # Use proper Responses API format with input_file
        response = llm_client.client.responses.create(
            model=llm_client.model_name,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_file", "file_id": file_id},
                        {"type": "input_text", "text": combined_prompt},
                    ],
                }
            ],
        )

        # Extract response content
        raw_content = response.output_text
        print(f"✅ GPT-5 Responses API completed: {len(raw_content)} characters")

        # Clean up file (optional - files auto-expire but good practice)
        try:
            llm_client.client.files.delete(file_id)
            print(f"🗑️ Cleaned up file: {file_id}")
        except Exception as cleanup_error:
            print(f"⚠️ File cleanup warning: {cleanup_error}")

        # Clean markdown formatting if present
        content = raw_content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        # Return content and metadata
        api_metadata = {
            "model_parameters": {
                "method": "responses_api_with_input_file",
                "file_id": file_id,
                "file_purpose": "user_data",
            },
            "usage": (
                response.usage.model_dump()
                if hasattr(response, "usage") and response.usage
                else {}
            ),
            "model_name": llm_client.model_name,
        }

        return content, api_metadata

    except Exception as e:
        print(f"❌ GPT-5 Responses API extraction failed: {e}")
        raise


def _extract_openai_chat_completions(
    pdf_path: str, llm_client: LLMClient, system_prompt: str, user_prompt: str
) -> Tuple[str, Dict[str, Any]]:
    """
    Extract using Chat Completions API with base64 inline PDF (for GPT-4o, etc).
    """
    import base64

    try:
        # Step 1: Base64 encode PDF
        with open(pdf_path, "rb") as pdf_file:
            pdf_b64 = base64.b64encode(pdf_file.read()).decode("utf-8")
        print(f"✅ PDF encoded: {len(pdf_b64)} base64 characters")

        # Step 2: Build user content with inline PDF
        user_content = [
            {"type": "text", "text": user_prompt},
            {
                "type": "file",
                "file": {
                    "filename": Path(pdf_path).name,
                    "file_data": f"data:application/pdf;base64,{pdf_b64}",
                },
            },
        ]

        # Step 3: Call Chat Completions API
        print("🧠 Calling Chat Completions API with inline PDF...")

        request_params = {
            "model": llm_client.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": 16384,
            "temperature": 0,
        }

        response = llm_client.client.chat.completions.create(**request_params)

        # Get response content
        raw_content = response.choices[0].message.content
        print(f"✅ Chat Completions completed: {len(raw_content)} characters")

        # Clean markdown formatting
        content = raw_content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        # Return content and metadata
        api_metadata = {
            "model_parameters": {
                "temperature": 0,
                "max_tokens": 16384,
                "method": "chat_completions_inline_pdf",
            },
            "usage": response.usage.model_dump() if response.usage else {},
            "model_name": llm_client.model_name,
        }

        return content, api_metadata

    except Exception as e:
        print(f"❌ Chat Completions extraction failed: {e}")
        raise


def _extract_openrouter_pdf_upload(
    pdf_path: str,
    llm_client: LLMClient,
    system_prompt: str,
    user_prompt: str,
    model_name: str,
) -> Tuple[str, Dict[str, Any]]:
    """Extract using OpenRouter's PDF upload API with OFFICIAL format from documentation."""

    # Encode PDF to base64 per official documentation
    print(f"📤 Encoding PDF for OpenRouter: {Path(pdf_path).name}")
    with open(pdf_path, "rb") as pdf_file:
        base64_pdf = base64.b64encode(pdf_file.read()).decode("utf-8")

    print(f"✅ PDF encoded: {len(base64_pdf)} characters")

    # Use OFFICIAL message format from OpenRouter documentation
    # https://openrouter.ai/docs/features/multimodal/pdfs
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{system_prompt}\n\n{user_prompt}"},
                {
                    "type": "file",
                    "file": {
                        "filename": Path(pdf_path).name,
                        "file_data": f"data:application/pdf;base64,{base64_pdf}",
                    },
                },
            ],
        }
    ]

    # Make direct HTTP request to OpenRouter
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": Config.OPENROUTER_SITE_URL,
        "X-Title": Config.OPENROUTER_SITE_NAME,
    }

    # Official payload format
    payload = {
        "model": llm_client.model_name,
        "messages": messages,
        "max_tokens": 32000,
        "temperature": 0,
    }

    # Make API call to OpenRouter
    print("🧠 Sending PDF to OpenRouter...")

    response = requests.post(url, headers=headers, json=payload, timeout=300)

    if response.status_code != 200:
        print(f"❌ OpenRouter API error: {response.status_code}")
        print(f"Response text: {response.text[:1000]}...")
        raise Exception(
            f"OpenRouter API error: {response.status_code} - {response.text}"
        )

    # Debug the response
    print(f"Response status: {response.status_code}")
    print(f"Response headers: {dict(response.headers)}")
    print(f"Response text length: {len(response.text)}")
    print(f"Response text preview: {response.text[:500]}...")

    # Save the raw response for debugging
    with open("/tmp/openrouter_response.txt", "w") as f:
        f.write(response.text)
    print("💾 Saved raw response to /tmp/openrouter_response.txt")

    # Check if response is mostly whitespace
    cleaned_response = response.text.strip()
    if not cleaned_response:
        print("❌ Response is empty or only whitespace")
        raise Exception("OpenRouter returned empty response")

    print(f"✅ Cleaned response length: {len(cleaned_response)} characters")
    print(f"Response preview: {cleaned_response[:500]}...")

    try:
        response_data = response.json()
        response_content = response_data["choices"][0]["message"]["content"]
        print(
            f"✅ OpenRouter PDF extraction completed: {len(response_content)} characters"
        )
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing failed: {e}")
        print(f"Response ends with: {response.text[-200:]}")
        raise

    # Prepare metadata for logging
    metadata = {
        "provider": "openrouter",
        "model_name": llm_client.model_name,
        "model_parameters": {
            "max_tokens": 32000,
            "temperature": 0,
            "method": "pdf_upload",
        },
        "usage": response_data.get("usage"),
    }

    return response_content, metadata
