"""
JSON Recovery System for Truncated LLM Responses

Repairs truncated or malformed JSON responses from the database without
requiring new API calls. Particularly useful for recovering expensive
responses that were charged but failed parsing.
"""

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class JSONRecovery:
    """
    Attempts to recover and repair truncated or malformed JSON responses.
    """

    def __init__(self, db_path: str = None):
        """Initialize with database path."""
        if db_path is None:
            db_path = Path("logs/prompts/prompt_logs.db")
        self.db_path = db_path

    def attempt_json_recovery(self, raw_response: str) -> Tuple[bool, List[Dict], str]:
        """
        Attempt to recover JSON from a truncated response.

        Args:
            raw_response: The raw response text

        Returns:
            Tuple of (success, extracted_data, recovery_method)
        """
        recovery_methods = [
            self._clean_markdown,
            self._fix_truncated_array,
            self._fix_truncated_object,
            self._extract_complete_objects,
            self._aggressive_json_repair,
        ]

        for method in recovery_methods:
            try:
                success, data, method_name = method(raw_response)
                if success and data:
                    return True, data, method_name
            except Exception as e:
                continue

        return False, [], "no_recovery_possible"

    def _clean_markdown(self, response: str) -> Tuple[bool, List[Dict], str]:
        """Remove markdown formatting and try parsing."""
        cleaned = response.strip()

        # Remove markdown code blocks
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(cleaned)
            return True, data if isinstance(data, list) else [data], "markdown_cleanup"
        except json.JSONDecodeError:
            return False, [], "markdown_cleanup_failed"

    def _fix_truncated_array(self, response: str) -> Tuple[bool, List[Dict], str]:
        """Fix truncated JSON arrays by adding missing closing brackets."""
        cleaned = (
            self._clean_markdown(response)[1]
            if self._clean_markdown(response)[0]
            else response
        )

        # Find the JSON array start
        array_start = cleaned.find("[")
        if array_start == -1:
            return False, [], "no_array_found"

        json_content = cleaned[array_start:]

        # Count open/close brackets and braces to determine what's missing
        open_brackets = json_content.count("[")
        close_brackets = json_content.count("]")
        open_braces = json_content.count("{")
        close_braces = json_content.count("}")

        # Try to fix by adding missing closures
        fixes_to_try = []

        # If we have unmatched braces, try closing them
        if open_braces > close_braces:
            missing_braces = open_braces - close_braces
            fixes_to_try.append(json_content + "}" * missing_braces)

        # If we have unmatched brackets, try closing them
        if open_brackets > close_brackets:
            missing_brackets = open_brackets - close_brackets
            fixes_to_try.append(json_content + "]" * missing_brackets)

        # Try both braces and brackets
        if open_braces > close_braces and open_brackets > close_brackets:
            missing_braces = open_braces - close_braces
            missing_brackets = open_brackets - close_brackets
            fixes_to_try.append(
                json_content + "}" * missing_braces + "]" * missing_brackets
            )

        for fixed_json in fixes_to_try:
            try:
                data = json.loads(fixed_json)
                return (
                    True,
                    data if isinstance(data, list) else [data],
                    "truncated_array_fix",
                )
            except json.JSONDecodeError:
                continue

        return False, [], "truncated_array_fix_failed"

    def _fix_truncated_object(self, response: str) -> Tuple[bool, List[Dict], str]:
        """Fix truncated objects by removing incomplete final object."""
        cleaned = (
            self._clean_markdown(response)[1]
            if self._clean_markdown(response)[0]
            else response
        )

        array_start = cleaned.find("[")
        if array_start == -1:
            return False, [], "no_array_found"

        json_content = cleaned[array_start:]

        # Find the last complete object by looking for complete {...} patterns
        # Split by commas and try to rebuild valid JSON
        try:
            # Remove the incomplete trailing part
            last_complete_brace = json_content.rfind("}")
            if last_complete_brace == -1:
                return False, [], "no_complete_objects"

            # Find the next comma or closing bracket after the last complete brace
            after_brace = json_content[last_complete_brace + 1 :].strip()

            if after_brace.startswith(","):
                # Remove everything after the last complete object
                truncated = json_content[: last_complete_brace + 1] + "\n]"
            elif after_brace == "" or after_brace == "]":
                # Already complete
                truncated = json_content[: last_complete_brace + 1] + "]"
            else:
                # Remove incomplete trailing data
                truncated = json_content[: last_complete_brace + 1] + "\n]"

            data = json.loads(truncated)
            return (
                True,
                data if isinstance(data, list) else [data],
                "truncated_object_fix",
            )

        except json.JSONDecodeError:
            return False, [], "truncated_object_fix_failed"

    def _extract_complete_objects(self, response: str) -> Tuple[bool, List[Dict], str]:
        """Extract only complete JSON objects from the response."""
        cleaned = (
            self._clean_markdown(response)[1]
            if self._clean_markdown(response)[0]
            else response
        )

        # Use regex to find complete JSON objects
        object_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"

        objects = []
        for match in re.finditer(object_pattern, cleaned, re.DOTALL):
            try:
                obj = json.loads(match.group())
                objects.append(obj)
            except json.JSONDecodeError:
                continue

        if objects:
            return True, objects, "complete_objects_extraction"

        return False, [], "no_complete_objects_found"

    def _aggressive_json_repair(self, response: str) -> Tuple[bool, List[Dict], str]:
        """Aggressive repair by reconstructing JSON from patterns."""
        cleaned = (
            self._clean_markdown(response)[1]
            if self._clean_markdown(response)[0]
            else response
        )

        # Look for field patterns like "FieldName": "Value",
        field_pattern = r'"([^"]+)":\s*([^,\n}]+)(?:,|\s*[}\]])'

        # Try to reconstruct objects from field patterns
        objects = []
        current_object = {}

        for match in re.finditer(field_pattern, cleaned):
            field_name = match.group(1)
            field_value = match.group(2).strip().strip('"')

            # Try to convert to appropriate type
            try:
                if field_value.isdigit():
                    field_value = int(field_value)
                elif field_value.replace(".", "").isdigit():
                    field_value = float(field_value)
                elif field_value.lower() in ["true", "false"]:
                    field_value = field_value.lower() == "true"
            except:
                pass  # Keep as string

            current_object[field_name] = field_value

            # If we hit a typical end-of-record field, start a new object
            if field_name in ["PageNumber", "CFR", "Deaths"]:
                if current_object:
                    objects.append(current_object.copy())
                current_object = {}

        # Add the last object if it has content
        if current_object:
            objects.append(current_object)

        if objects:
            return True, objects, "aggressive_repair"

        return False, [], "aggressive_repair_failed"

    def recover_failed_response(self, response_id: int) -> Optional[Dict[str, Any]]:
        """
        Attempt to recover a specific failed response from the database.

        Args:
            response_id: Database ID of the failed response

        Returns:
            Recovery result with data and metadata
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT model_name, prompt_version, raw_response, parsing_errors
            FROM prompt_logs 
            WHERE id = ? AND parsed_success = 0
        """,
            (response_id,),
        )

        result = cursor.fetchone()
        conn.close()

        if not result:
            return None

        model_name, prompt_version, raw_response, original_error = result

        # Attempt recovery
        success, recovered_data, method = self.attempt_json_recovery(raw_response)

        return {
            "response_id": response_id,
            "model_name": model_name,
            "prompt_version": prompt_version,
            "original_error": original_error,
            "recovery_success": success,
            "recovery_method": method,
            "recovered_records": len(recovered_data) if success else 0,
            "recovered_data": recovered_data if success else [],
            "raw_response_length": len(raw_response),
        }

    def recover_all_failed_large_responses(self) -> List[Dict[str, Any]]:
        """
        Attempt to recover all failed responses with substantial content.

        Returns:
            List of recovery results
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id FROM prompt_logs 
            WHERE parsed_success = 0 
            AND LENGTH(raw_response) > 1000
            AND raw_response NOT LIKE 'ERROR:%'
            ORDER BY LENGTH(raw_response) DESC
        """
        )

        failed_ids = [row[0] for row in cursor.fetchall()]
        conn.close()

        results = []
        for response_id in failed_ids:
            recovery_result = self.recover_failed_response(response_id)
            if recovery_result:
                results.append(recovery_result)

        return results

    def update_database_with_recovery(
        self, response_id: int, recovered_data: List[Dict]
    ) -> bool:
        """
        Update the database with recovered data (optional - creates a new entry).

        Args:
            response_id: Original response ID
            recovered_data: Successfully recovered data

        Returns:
            Success status
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get original entry
        cursor.execute(
            """
            SELECT * FROM prompt_logs WHERE id = ?
        """,
            (response_id,),
        )

        original = cursor.fetchone()
        if not original:
            conn.close()
            return False

        try:
            # Create new entry with recovered data
            cursor.execute(
                """
                INSERT INTO prompt_logs (
                    timestamp, prompt_type, prompt_version, model_name,
                    model_parameters, system_prompt, user_prompt, raw_response,
                    parsed_success, records_extracted, parsing_errors,
                    execution_time_seconds, prompt_metadata, custom_metrics
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    original[1],  # timestamp
                    original[2],  # prompt_type
                    original[3],  # prompt_version
                    original[4] + "_RECOVERED",  # model_name (mark as recovered)
                    original[5],  # model_parameters
                    original[6],  # system_prompt
                    original[7],  # user_prompt
                    json.dumps(recovered_data, indent=2),  # raw_response (cleaned JSON)
                    1,  # parsed_success = True
                    len(recovered_data),  # records_extracted
                    f"RECOVERED from ID {response_id}",  # parsing_errors
                    original[12],  # execution_time_seconds
                    original[13],  # prompt_metadata
                    original[14],  # custom_metrics
                ),
            )

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            conn.close()
            return False
