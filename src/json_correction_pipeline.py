#!/usr/bin/env python3
"""
JSON Correction Pipeline using prompt v1.3.1
Corrects extracted JSON surveillance data using LLM to fix obvious number errors.
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd

from src.config import Config
from src.llm_client import LLMClient
from src.prompt_manager import PromptManager
from src.prompt_logger import PromptLogger


def load_extracted_json(json_path: str) -> List[Dict[str, Any]]:
    """Load extracted surveillance data from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Loaded {len(data)} records from {Path(json_path).name}")
    return data


def apply_json_corrections_v1_3_1(
    extracted_data: List[Dict[str, Any]],
    model_name: str = None,
    prompt_version: str = "v1.3.1"
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
    """
    Apply LLM corrections to JSON data using prompt v1.3.1.
    
    Args:
        extracted_data: List of surveillance records with NarrativeText
        model_name: Optional model override
        prompt_version: Prompt version to use (defaults to v1.3.1)
    
    Returns:
        Tuple of (corrected_data, corrections_json, call_id)
    """
    print(f"🧠 Applying JSON corrections with prompt {prompt_version}...")
    
    # Initialize components
    prompt_manager = PromptManager()
    prompt_logger = PromptLogger()
    
    # Setup LLM client
    if model_name:
        llm_client = LLMClient.create_client_for_model(model_name)
    else:
        llm_client = LLMClient()
    
    # Import and set the prompt version
    try:
        # Check if prompt exists in JSON, if not import from markdown
        try:
            prompt_data = prompt_manager.get_prompt_version("health_data_extraction", prompt_version)
            if not (prompt_data.get("system_prompt") or prompt_data.get("user_prompt_template")):
                raise ValueError("Empty prompt content")
        except (FileNotFoundError, ValueError):
            # Auto-import from markdown
            markdown_path = f"prompts/markdown/health_data_extraction/health_data_extraction_{prompt_version}.md"
            print(f"📥 Importing prompt {prompt_version} from markdown...")
            prompt_manager.create_prompt_from_markdown("health_data_extraction", markdown_path)
        
        # Set as current version
        prompt_manager.set_current_version("health_data_extraction", prompt_version)
        
    except Exception as e:
        raise FileNotFoundError(f"Prompt {prompt_version} not found: {e}")
    
    # Build the prompt with data using template.safe_substitute to avoid formatting conflicts
    from string import Template
    
    # Get the raw prompt template
    prompt_data = prompt_manager.get_prompt_version("health_data_extraction", prompt_version)
    system_prompt = prompt_data.get("system_prompt", "")
    user_prompt_template = prompt_data.get("user_prompt_template", "")
    
    # Use Template instead of format() to safely substitute variables
    template = Template(user_prompt_template)
    user_prompt = template.safe_substitute(extracted_data=json.dumps(extracted_data, indent=2))
    
    # Get metadata for logging (extract from prompt data)
    prompt_metadata = {
        "prompt_type": "health_data_extraction",
        "version": prompt_version,
        "provider": "json_correction",
        "preprocessor": prompt_data.get("preprocessor", "json_correction")
    }
    
    # For logging, truncate the data to avoid huge logs
    json_sample = json.dumps(extracted_data[:2], indent=2) if len(extracted_data) > 2 else json.dumps(extracted_data, indent=2)
    prompt_for_logging = template.safe_substitute(extracted_data=f"[{len(extracted_data)} records - sample: {json_sample}]")
    
    start_time = time.time()
    
    try:
        # Get model info for logging
        model_info = llm_client.get_model_info()
        actual_model_name = model_info["model_name"]
        provider = model_info["provider"]
        
        print(f"🤖 Sending {len(extracted_data)} records to {provider} for correction...")
        print(f"Model: {actual_model_name}")
        print(f"Using prompt: {prompt_metadata['prompt_type']} v{prompt_metadata['version']}")
        
        # Determine token limits based on model type
        is_reasoning_model = (
            "gpt-5" in actual_model_name.lower() or 
            "grok" in actual_model_name.lower()
        )
        max_tokens = 100000 if is_reasoning_model else 16384
        
        # Make LLM call
        raw_response, api_metadata = llm_client.create_chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=0,
        )
        
        execution_time = time.time() - start_time
        
        print(f"✅ API call successful: {len(raw_response)} characters received")
        print(f"Execution time: {execution_time:.2f} seconds")
        print("🔄 Parsing correction response...")
        
        # Parse the correction response
        corrections_json, parsed_success = parse_corrections_response(raw_response)
        
        if parsed_success:
            corrections_count = len(corrections_json.get('corrections', []))
            print(f"✅ Parsed {corrections_count} corrections successfully")
        else:
            print("❌ Failed to parse corrections response")
            corrections_json = {"corrections": [], "summary": {}}
            corrections_count = 0
        
        # Log the LLM call
        call_id = prompt_logger.log_llm_call(
            prompt_metadata=prompt_metadata,
            model_name=actual_model_name,
            model_parameters=api_metadata["model_parameters"],
            system_prompt=system_prompt,
            user_prompt=prompt_for_logging,
            raw_response=raw_response,
            parsed_success=parsed_success,
            records_extracted=corrections_count,
            parsing_errors=None if parsed_success else "Failed to parse corrections JSON",
            execution_time_seconds=execution_time,
            custom_metrics={
                "input_records": len(extracted_data),
                "corrections_applied": corrections_count,
                "provider": provider,
                "usage_tokens": api_metadata.get("usage"),
                "correction_type": "json_data_quality"
            },
        )
        
        print(f"📝 Logged LLM call with ID: {call_id}")
        
        # Apply corrections to the data
        corrected_data = apply_corrections_to_json(extracted_data, corrections_json)
        
        return corrected_data, corrections_json, call_id
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_message = str(e)
        
        # Log failed call
        call_id = prompt_logger.log_llm_call(
            prompt_metadata=prompt_metadata,
            model_name=llm_client.model_name,
            model_parameters={"max_tokens": max_tokens, "temperature": 0},
            system_prompt=system_prompt,
            user_prompt=prompt_for_logging,
            raw_response=f"ERROR: {error_message}",
            parsed_success=False,
            records_extracted=0,
            parsing_errors=error_message,
            execution_time_seconds=execution_time,
            custom_metrics={"input_records": len(extracted_data)},
        )
        
        print(f"❌ JSON correction failed: {e}")
        raise


def parse_corrections_response(raw_response: str) -> Tuple[Dict[str, Any], bool]:
    """Parse the LLM corrections response."""
    try:
        # Clean up response if it has markdown
        response_text = raw_response.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        # Parse JSON
        corrections_json = json.loads(response_text)
        
        # Validate structure
        if not isinstance(corrections_json.get('corrections'), list):
            corrections_json['corrections'] = []
        if not isinstance(corrections_json.get('summary'), dict):
            corrections_json['summary'] = {}
        
        return corrections_json, True
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing failed: {e}")
        print(f"📄 Response preview: {raw_response[:500]}...")
        
        # Try to extract just the corrections array
        try:
            import re
            corrections_match = re.search(r'"corrections":\s*\[(.*?)\]', raw_response, re.DOTALL)
            if corrections_match:
                corrections_array = f'[{corrections_match.group(1)}]'
                corrections_list = json.loads(corrections_array)
                return {
                    "corrections": corrections_list,
                    "summary": {"recovery_attempted": True}
                }, True
        except:
            pass
        
        return {"corrections": [], "summary": {}}, False


def apply_corrections_to_json(
    data: List[Dict[str, Any]], 
    corrections_json: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Apply corrections to the JSON data."""
    corrected_data = [record.copy() for record in data]  # Deep copy
    
    corrections = corrections_json.get('corrections', [])
    
    for correction in corrections:
        try:
            record_index = correction['record_index']
            field = correction['field']
            new_value = correction['new_value']
            old_value = correction.get('old_value', 'unknown')
            confidence = correction.get('confidence', 0)
            explanation = correction.get('explanation', 'No explanation')
            
            # Validate index
            if 0 <= record_index < len(corrected_data):
                # Get PDF name for reference
                pdf_name = corrected_data[record_index].get('SourceFile', 'unknown')
                
                # Apply correction
                corrected_data[record_index][field] = new_value
                
                # Add PDF info to correction metadata for easier reference
                correction['pdf_name'] = pdf_name
                correction['country'] = corrected_data[record_index].get('Country', 'unknown')
                correction['event'] = corrected_data[record_index].get('Event', 'unknown')
                
                print(f"✅ Applied correction [{record_index}]: {field}")
                print(f"   PDF: {pdf_name}")
                print(f"   Country/Event: {correction['country']}/{correction['event']}")
                print(f"   Old: '{old_value}' → New: '{new_value}' (confidence: {confidence:.2f})")
                print(f"   Reason: {explanation}")
            else:
                print(f"⚠️ Invalid record index {record_index}, skipping correction")
                
        except KeyError as e:
            print(f"⚠️ Missing key in correction: {e}, skipping")
        except Exception as e:
            print(f"⚠️ Error applying correction: {e}, skipping")
    
    applied_corrections = len(corrections)
    print(f"\n📊 Corrections Summary:")
    print(f"   Total corrections applied: {applied_corrections}")
    print(f"   Records processed: {len(corrected_data)}")
    
    return corrected_data


def save_corrected_data(
    corrected_data: List[Dict[str, Any]],
    corrections_json: Dict[str, Any],
    call_id: str,
    prompt_version: str,
    model_name: str,
    output_dir: Path = None
) -> Tuple[str, str]:
    """Save corrected data and corrections metadata."""
    
    if output_dir is None:
        output_dir = Config.OUTPUTS_DIR
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create safe model name for filename
    model_for_filename = model_name.replace("/", "_").replace("-", "_")
    
    # Save corrected JSON
    corrected_json_path = (
        output_dir / f"corrected_{call_id}_prompt_{prompt_version}_model_{model_for_filename}.json"
    )
    
    with open(corrected_json_path, 'w', encoding='utf-8') as f:
        json.dump(corrected_data, f, indent=2, ensure_ascii=False)
    
    # Save corrected CSV
    corrected_csv_path = corrected_json_path.with_suffix('.csv')
    df = pd.DataFrame(corrected_data)
    df.to_csv(corrected_csv_path, index=False)
    
    # Save corrections metadata
    corrections_path = (
        output_dir / f"corrections_metadata_{call_id}_prompt_{prompt_version}_model_{model_for_filename}.json"
    )
    
    with open(corrections_path, 'w', encoding='utf-8') as f:
        json.dump(corrections_json, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved corrected data:")
    print(f"   JSON: {corrected_json_path.name}")
    print(f"   CSV: {corrected_csv_path.name}")
    print(f"   Corrections metadata: {corrections_path.name}")
    
    return str(corrected_json_path), str(corrections_path)


def main():
    """Main function for JSON correction pipeline."""
    parser = argparse.ArgumentParser(
        description="Correct extracted JSON surveillance data using LLM"
    )
    parser.add_argument(
        "--json-path",
        "-j",
        required=True,
        type=str,
        help="Path to extracted JSON file to correct"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Model to use (e.g., 'gpt-4o', 'claude-3.5-sonnet')"
    )
    parser.add_argument(
        "--prompt-version",
        "-p",
        type=str,
        default="v1.3.1",
        help="Prompt version to use (default: v1.3.1)"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Output directory (default: outputs/)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"❌ JSON file not found: {json_path}")
        return
    
    print("🔄 JSON CORRECTION PIPELINE")
    print("=" * 40)
    print(f"📁 Input file: {json_path.name}")
    print(f"🤖 Model: {args.model or 'default'}")
    print(f"📝 Prompt version: {args.prompt_version}")
    
    try:
        # Load data
        extracted_data = load_extracted_json(str(json_path))
        
        # Validate data has NarrativeText
        has_narrative = any('NarrativeText' in record for record in extracted_data[:5])
        if not has_narrative:
            print("⚠️ Warning: Records may not have NarrativeText field")
            print("   Corrections may be limited without narrative context")
        
        # Apply corrections
        corrected_data, corrections_json, call_id = apply_json_corrections_v1_3_1(
            extracted_data=extracted_data,
            model_name=args.model,
            prompt_version=args.prompt_version
        )
        
        # Save results
        output_dir = Path(args.output_dir) if args.output_dir else None
        corrected_json_path, corrections_path = save_corrected_data(
            corrected_data=corrected_data,
            corrections_json=corrections_json,
            call_id=call_id,
            prompt_version=args.prompt_version,
            model_name=args.model or "default",
            output_dir=output_dir
        )
        
        # Print summary
        corrections_count = len(corrections_json.get('corrections', []))
        summary = corrections_json.get('summary', {})
        
        print(f"\n✅ JSON CORRECTION COMPLETE")
        print("=" * 30)
        print(f"📊 Records processed: {len(corrected_data)}")
        print(f"🔧 Corrections applied: {corrections_count}")
        print(f"📝 Call ID: {call_id}")
        
        if summary:
            print(f"📋 Summary from LLM:")
            for key, value in summary.items():
                print(f"   {key}: {value}")
        
        print(f"\n📁 Output files:")
        print(f"   {Path(corrected_json_path).name}")
        print(f"   {Path(corrections_path).name}")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return


if __name__ == "__main__":
    main()