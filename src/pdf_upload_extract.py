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
    system_prompt, user_prompt, prompt_metadata, prompt_for_logging = (
        prompt_manager.build_prompt(
            prompt_type="health_data_extraction", 
            version=prompt_version
        )
    )

    start_time = time.time()

    try:
        if provider == "openai":
            # OpenAI file upload method
            extracted_data, api_metadata = _extract_openai_pdf_upload(
                pdf_path, llm_client, system_prompt, user_prompt
            )
        elif provider == "openrouter":
            # OpenRouter PDF upload method
            extracted_data, api_metadata = _extract_openrouter_pdf_upload(
                pdf_path, llm_client, system_prompt, user_prompt, actual_model_name
            )
        else:
            raise ValueError(f"Provider {provider} does not support PDF upload")

        execution_time = time.time() - start_time

        # Parse JSON response
        parsing_errors = None
        parsed_success = False
        records = []

        try:
            # Clean up response if it has markdown
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
            raw_response=extracted_data,
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
    Extract using OpenAI file upload API with Chat Completions.
    Uses the proven approach from src/llm_extract.py (Chat Completions + file upload).
    """
    print(f"📤 Uploading PDF to OpenAI: {Path(pdf_path).name}")
    
    # Step 1: Upload PDF file to OpenAI
    try:
        with open(pdf_path, "rb") as pdf_file:
            file_response = llm_client.client.files.create(
                file=pdf_file, 
                purpose="user_data"
            )
        
        file_id = file_response.id
        print(f"✅ PDF uploaded successfully, file ID: {file_id}")
        
    except Exception as e:
        print(f"❌ PDF upload failed: {e}")
        raise
    
    # Step 2: Use Chat Completions API with file input (proven working method)
    try:
        # Get model name
        model_name = llm_client.model_name
        
        # Use Chat Completions API with file attachment (based on working src/llm_extract.py)
        response = llm_client.client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "file", "file": {"file_id": file_id}},
                    ],
                },
            ],
            temperature=0,
            max_tokens=16000,
        )
        
        # Extract response content
        response_content = response.choices[0].message.content
        print(f"✅ OpenAI PDF extraction completed: {len(response_content)} characters")
        
        # Prepare metadata for logging
        metadata = {
            "provider": "openai",
            "model_name": model_name,
            "model_parameters": {
                "file_id": file_id,
                "api_type": "chat_completions",
                "max_tokens": 16000,
                "temperature": 0,
            },
            "usage": (
                response.usage.model_dump()
                if hasattr(response, "usage") and response.usage
                else None
            ),
        }
        
        # Clean up uploaded file
        try:
            llm_client.client.files.delete(file_id)
            print(f"🗑️ Cleaned up uploaded file: {file_id}")
        except Exception as cleanup_error:
            print(f"⚠️ Failed to cleanup file {file_id}: {cleanup_error}")
        
        return response_content, metadata
        
    except Exception as e:
        # Clean up file on error
        try:
            llm_client.client.files.delete(file_id)
        except:
            pass
        print(f"❌ OpenAI Chat Completions API failed: {e}")
        raise


def _extract_openrouter_pdf_upload(
    pdf_path: str,
    llm_client: LLMClient,
    system_prompt: str,
    user_prompt: str,
    model_name: str,
) -> Tuple[str, Dict[str, Any]]:
    """Extract using OpenRouter's PDF upload API with the WORKING format from hybrid extractor."""

    # Encode PDF to base64
    print(f"📤 Encoding PDF for OpenRouter: {Path(pdf_path).name}")
    with open(pdf_path, "rb") as pdf_file:
        base64_pdf = base64.b64encode(pdf_file.read()).decode("utf-8")

    print(f"✅ PDF encoded: {len(base64_pdf)} characters")

    # Use the WORKING message format from src/llm_extract_hybrid.py
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "file",
                    "file": {
                        "filename": Path(pdf_path).name,
                        "file_data": f"data:application/pdf;base64,{base64_pdf}"
                    }
                }
            ]
        }
    ]

    # Configure PDF processing plugins (WORKING format)
    plugins = [
        {
            "id": "file-parser",
            "pdf": {
                "engine": "pdf-text"  # Free engine for well-structured PDFs
            }
        }
    ]

    # Make direct HTTP request to OpenRouter (WORKING format)
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": Config.OPENROUTER_SITE_URL,
        "X-Title": Config.OPENROUTER_SITE_NAME,
    }

    payload = {
        "model": model_name,
        "messages": messages,
        "plugins": plugins,  # This was missing in my broken version!
        "max_tokens": 16000,
        "temperature": 0,
    }

    print(f"� Sending PDF to OpenRouter...")
    response = requests.post(url, headers=headers, json=payload, timeout=180)
    
    if response.status_code != 200:
        raise Exception(f"OpenRouter API error: {response.status_code} - {response.text}")

    response_data = response.json()
    response_content = response_data["choices"][0]["message"]["content"]

    print(f"✅ OpenRouter PDF extraction completed: {len(response_content)} characters")

    # Prepare metadata for logging
    metadata = {
        "provider": "openrouter",
        "model_name": model_name,
        "model_parameters": {
            "max_tokens": 16000,
            "temperature": 0,
            "method": "pdf_upload",
            "plugins": plugins,
        },
        "usage": response_data.get("usage"),
    }

    return response_content, metadata
