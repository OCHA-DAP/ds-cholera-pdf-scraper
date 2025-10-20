"""
LLM-based narrative linking for table corrections.

This module uses LLM intelligence to identify and apply corrections from narrative text
to raw table data, rather than relying on rigid regex patterns.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI

from src.config import Config

logger = logging.getLogger(__name__)


class LLMNarrativeLinking:
    """
    Use LLM intelligence to link narrative corrections to table data.

    This approach is more flexible than regex patterns and can handle:
    - Complex formatting corrections (27,16 -> 27,160)
    - Contextual updates mentioned in narrative text
    - Implicit corrections that require understanding
    """

    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=config.EFFECTIVE_OPENAI_KEY)

    def apply_narrative_corrections(
        self,
        table_data: pd.DataFrame,
        narrative_text: str,
        pdf_filename: str = "unknown",
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Use LLM to identify and apply corrections from narrative text to table data.

        Args:
            table_data: Raw table extracted by preprocessor
            narrative_text: Full narrative text from PDF
            pdf_filename: For logging/provenance

        Returns:
            Tuple of (corrected_dataframe, list_of_corrections_applied)
        """
        if table_data.empty or not narrative_text.strip():
            return table_data.copy(), []

        logger.info(f"Applying LLM narrative corrections to {len(table_data)} rows")

        # Prepare the prompt for the LLM
        corrections_prompt = self._build_corrections_prompt(table_data, narrative_text)

        try:
            # Get corrections from LLM
            corrections_response = self._call_llm_for_corrections(corrections_prompt)
            corrections = self._parse_corrections_response(corrections_response)

            # Apply corrections to dataframe
            corrected_df = self._apply_corrections_to_dataframe(table_data, corrections)

            # Add provenance info
            for correction in corrections:
                correction.update(
                    {
                        "pdf_filename": pdf_filename,
                        "method": "llm_narrative_linking",
                        "narrative_snippet": (
                            narrative_text[:200] + "..."
                            if len(narrative_text) > 200
                            else narrative_text
                        ),
                    }
                )

            logger.info(f"Applied {len(corrections)} narrative corrections")
            return corrected_df, corrections

        except Exception as e:
            logger.error(f"Error in LLM narrative corrections: {e}")
            return table_data.copy(), []

    def _build_corrections_prompt(
        self, table_data: pd.DataFrame, narrative_text: str
    ) -> str:
        """Build the prompt for LLM to identify corrections."""

        # Convert table to a clear format for the LLM
        table_json = table_data.to_dict(orient="records")

        prompt = f"""You are a data quality expert analyzing health surveillance reports. You have a table extracted from a PDF and narrative text that may contain corrections or clarifications to the table data.

Your task is to identify any corrections mentioned in the narrative text that should be applied to the table data.

**EXTRACTED TABLE DATA:**
```json
{json.dumps(table_json, indent=2)}
```

**NARRATIVE TEXT:**
{narrative_text}

**INSTRUCTIONS:**
1. Carefully read the narrative text for any mentions of corrections, updates, or clarifications to numbers in the table
2. Look for patterns like:
   - "Angola has reported 27,160 cholera cases" (when table shows 27,16 - formatting error)
   - "The total should be X" or "corrected to X"
   - "Updated figures show X"
   - Any discrepancies between table numbers and narrative numbers
3. For each correction you identify, provide:
   - Which row (by Country and Event)
   - Which field needs correction
   - Old value (from table)
   - New value (from narrative)
   - Confidence level (0.0-1.0)
   - Explanation of why this is a correction

**OUTPUT FORMAT:**
Return a JSON array of corrections. If no corrections are found, return an empty array [].

Example:
```json
[
  {{
    "country": "Angola",
    "event": "Cholera",
    "field": "TotalCases",
    "old_value": "27,16",
    "new_value": "27160",
    "confidence": 0.95,
    "explanation": "Table shows malformed number 27,16 but narrative clearly states 27,160 cholera cases"
  }}
]
```

**IMPORTANT:**
- Only include corrections where you're confident the narrative contradicts or clarifies the table
- Convert all numbers to standard format (no commas for integers)
- Be conservative - if unsure, don't include the correction
- Pay special attention to formatting errors (missing digits, misplaced commas)

Return only the JSON array, no other text.
"""
        return prompt

    def _call_llm_for_corrections(self, prompt: str) -> str:
        """Call LLM to identify corrections."""
        try:
            response = self.client.chat.completions.create(
                model=self.config.OPENAI_MODEL,  # Use direct OpenAI model name
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for consistent analysis
                max_tokens=2000,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise

    def _parse_corrections_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured corrections."""
        try:
            # Extract JSON from response (in case LLM added extra text)
            response_clean = response.strip()
            if response_clean.startswith("```json"):
                response_clean = response_clean[7:]
            if response_clean.endswith("```"):
                response_clean = response_clean[:-3]

            corrections = json.loads(response_clean)

            # Validate corrections format
            if not isinstance(corrections, list):
                logger.warning(
                    "LLM response is not a list, returning empty corrections"
                )
                return []

            validated_corrections = []
            for correction in corrections:
                if self._validate_correction(correction):
                    validated_corrections.append(correction)
                else:
                    logger.warning(f"Invalid correction format: {correction}")

            return validated_corrections

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM corrections response as JSON: {e}")
            logger.debug(f"Raw response: {response}")
            return []

    def _validate_correction(self, correction: Dict[str, Any]) -> bool:
        """Validate that a correction has required fields."""
        required_fields = [
            "country",
            "event",
            "field",
            "old_value",
            "new_value",
            "confidence",
        ]
        return all(field in correction for field in required_fields)

    def _apply_corrections_to_dataframe(
        self, df: pd.DataFrame, corrections: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Apply the identified corrections to the dataframe."""
        corrected_df = df.copy()

        for correction in corrections:
            try:
                # Find the row to correct
                mask = (
                    corrected_df["Country"].str.contains(
                        correction["country"], case=False, na=False
                    )
                ) & (
                    corrected_df["Event"].str.contains(
                        correction["event"], case=False, na=False
                    )
                )

                matching_rows = corrected_df[mask]
                if len(matching_rows) == 0:
                    logger.warning(
                        f"No matching row found for correction: {correction['country']} - {correction['event']}"
                    )
                    continue
                elif len(matching_rows) > 1:
                    logger.warning(
                        f"Multiple matching rows found for correction: {correction['country']} - {correction['event']}"
                    )
                    continue

                # Apply the correction
                row_index = matching_rows.index[0]
                field = correction["field"]

                if field not in corrected_df.columns:
                    logger.warning(f"Field {field} not found in dataframe columns")
                    continue

                old_value = corrected_df.loc[row_index, field]
                new_value = correction["new_value"]

                # Verify old value matches (with some flexibility for formatting)
                if not self._values_match(str(old_value), str(correction["old_value"])):
                    logger.warning(
                        f"Old value mismatch: expected {correction['old_value']}, found {old_value}"
                    )
                    continue

                # Apply correction
                corrected_df.loc[row_index, field] = new_value

                logger.info(
                    f"Applied correction: {correction['country']} {field} {old_value} -> {new_value}"
                )

            except Exception as e:
                logger.error(f"Error applying correction {correction}: {e}")
                continue

        return corrected_df

    def _values_match(self, val1: str, val2: str) -> bool:
        """Check if two values match (with flexibility for formatting differences)."""
        # Normalize both values for comparison
        norm1 = val1.replace(",", "").replace(" ", "").strip()
        norm2 = val2.replace(",", "").replace(" ", "").strip()

        return norm1 == norm2
