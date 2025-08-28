"""
Narrative Linking Engine - Links narrative corrections to specific table rows/fields.
Implements proper RAG-based approach: retrieve relevant paragraphs -> link to rows -> extract values.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Improved regex patterns for different correction types
RevisionRE = re.compile(r"\brevis(?:ed|ion)\b.*?(\d[\d,\. ]*)", re.I)
UpdateRE = re.compile(
    r"\b(?:updat(?:ed|e)|correct(?:ed|ion))\b.*?(?:to\s+)?(\d[\d,\. ]*)", re.I
)
ShouldBeRE = re.compile(r"\bshould\s+be\s+(\d[\d,\. ]*)", re.I)
FromToRE = re.compile(r"\bfrom\s+(\d[\d,\. ]*)\s+to\s+(\d[\d,\. ]*)", re.I)
TotalReportedRE = re.compile(r"\b(?:total\s+of\s+|reported\s+)(\d[\d,\. ]*)", re.I)
# New pattern specifically for number formatting corrections
FormattingFixRE = re.compile(r"(\d{1,3}),(\d{1,2})\b.*?(\d{1,3}),(\d{3})", re.I)

# Field mapping - maps narrative mentions to table columns
FIELD_ALIASES = {
    "TotalCases": [
        "total cases",
        "cases total",
        "cumulative cases",
        "total of",
        "reported",
        "cases",
    ],
    "CasesConfirmed": ["confirmed cases", "confirmed", "laboratory confirmed"],
    "Deaths": ["deaths", "fatalities", "mortality", "died"],
    "CFR": ["cfr", "case fatality", "fatality rate", "mortality rate"],
    "Grade": ["grade", "graded", "level"],
}


class NarrativeLinkingEngine:
    """
    Proper narrative linking: retrieves relevant paragraphs and links corrections
    to specific table rows/fields with confidence scoring.
    """

    def __init__(self, field_aliases: Dict[str, List[str]] = None):
        """
        Initialize narrative linking engine.

        Args:
            field_aliases: Mapping of field names to their text aliases
        """
        self.field_aliases = field_aliases or FIELD_ALIASES
        self.initialized = True
        logger.info("NarrativeLinkingEngine initialized with proper row/field linking")

    def link_corrections_to_table(
        self, table_rows: List[Dict[str, Any]], text_blocks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Link narrative corrections to specific table rows and fields.

        Args:
            table_rows: List of table rows with country, event, and data fields
            text_blocks: List of narrative text blocks with page numbers

        Returns:
            List of correction dictionaries with row keys, fields, and provenance
        """
        all_corrections = []

        for row in table_rows:
            row_corrections = self._link_row_corrections(row, text_blocks)
            all_corrections.extend(row_corrections)

        return all_corrections

    def _link_row_corrections(
        self, table_row: Dict[str, Any], text_blocks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Find corrections for a specific table row by retrieving relevant text blocks.

        Args:
            table_row: Single table row with fields like Country, Event, TotalCases, etc.
            text_blocks: All available text blocks

        Returns:
            List of corrections for this specific row
        """
        # Build row key for linking
        row_key = self._build_row_key(table_row)

        # Retrieve relevant text blocks for this row
        relevant_blocks = self._retrieve_relevant_blocks(row_key, text_blocks)

        corrections = []
        for block in relevant_blocks:
            block_corrections = self._extract_corrections_from_block(
                block, table_row, row_key
            )
            corrections.extend(block_corrections)

        # Keep only the best correction per field
        return self._deduplicate_corrections(corrections)

    def _build_row_key(self, table_row: Dict[str, Any]) -> Tuple[str, ...]:
        """Build a unique key for this table row."""
        country = table_row.get("Country", "")
        event = table_row.get("Event", "")
        return (country, event)

    def _retrieve_relevant_blocks(
        self, row_key: Tuple[str, ...], text_blocks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Retrieve text blocks relevant to this specific row.
        Uses simple keyword matching (can be upgraded to vector search).
        """
        country, event = row_key
        relevant_blocks = []

        for block in text_blocks:
            text = block.get("text", "").lower()

            # Check if block mentions this country and event
            country_match = country.lower() in text
            event_match = event.lower() in text

            # Also look for numeric patterns that suggest data corrections
            has_numbers = bool(re.search(r"\d[\d,\. ]*", text))

            if (country_match or event_match) and has_numbers:
                # Calculate relevance score
                score = 0.0
                if country_match and event_match:
                    score = 0.9
                elif country_match or event_match:
                    score = 0.6

                block_with_score = block.copy()
                block_with_score["relevance_score"] = score
                relevant_blocks.append(block_with_score)

        # Sort by relevance score
        return sorted(
            relevant_blocks, key=lambda x: x.get("relevance_score", 0), reverse=True
        )

    def _extract_corrections_from_block(
        self, block: Dict[str, Any], table_row: Dict[str, Any], row_key: Tuple[str, ...]
    ) -> List[Dict[str, Any]]:
        """
        Extract corrections from a single text block for a specific table row.
        """
        text = block.get("text", "")
        corrections = []

        # First check for formatting fixes (malformed -> corrected numbers)
        formatting_match = FormattingFixRE.search(text)
        if formatting_match:
            malformed = f"{formatting_match.group(1)},{formatting_match.group(2)}"
            corrected = f"{formatting_match.group(3)},{formatting_match.group(4)}"

            # Check if this malformed number matches any in the table row
            for field_name, field_value in table_row.items():
                if str(field_value) == malformed:
                    corrected_num = self._normalize_number(corrected)
                    if corrected_num:
                        confidence = self._calculate_confidence(
                            block.get("relevance_score", 0.5),
                            text,
                            field_name,
                            field_value,
                            corrected_num,
                        )

                        corrections.append(
                            {
                                "row_key": row_key,
                                "field": field_name,
                                "old_value": field_value,
                                "new_value": corrected_num,
                                "correction_type": "formatting_fix",
                                "confidence": confidence
                                + 0.2,  # High confidence for exact matches
                                "provenance": {
                                    "page": block.get("page", 0),
                                    "text_snippet": text[:200],
                                    "full_match": formatting_match.group(0),
                                    "relevance_score": block.get(
                                        "relevance_score", 0.5
                                    ),
                                },
                            }
                        )

        # Try other correction patterns
        patterns = [
            ("revision", RevisionRE),
            ("update", UpdateRE),
            ("should_be", ShouldBeRE),
            ("from_to", FromToRE),
            ("total_reported", TotalReportedRE),
        ]

        for pattern_name, pattern_re in patterns:
            match = pattern_re.search(text)
            if match:
                # Determine new value
                if pattern_name == "from_to":
                    old_raw, new_raw = match.group(1), match.group(2)
                    new_value = self._normalize_number(new_raw)
                else:
                    new_raw = match.group(1)
                    new_value = self._normalize_number(new_raw)

                if new_value is None:
                    continue

                # Infer which field this correction applies to
                field = self._infer_field(text)
                if not field:
                    # Try to guess based on the context and existing table values
                    field = self._guess_field_from_context(text, table_row, new_value)

                if field:
                    old_value = table_row.get(field)
                    confidence = self._calculate_confidence(
                        block.get("relevance_score", 0.5),
                        text,
                        field,
                        old_value,
                        new_value,
                    )

                    corrections.append(
                        {
                            "row_key": row_key,
                            "field": field,
                            "old_value": old_value,
                            "new_value": new_value,
                            "correction_type": pattern_name,
                            "confidence": confidence,
                            "provenance": {
                                "page": block.get("page", 0),
                                "text_snippet": text[:200],
                                "full_match": match.group(0),
                                "relevance_score": block.get("relevance_score", 0.5),
                            },
                        }
                    )

        return corrections

    def _normalize_number(self, raw_number: str) -> Optional[float]:
        """Convert raw number string to normalized float."""
        if not raw_number:
            return None

        # Remove commas and extra spaces
        cleaned = raw_number.replace(",", "").replace(" ", "").strip()

        try:
            return float(cleaned)
        except ValueError:
            return None

    def _infer_field(self, text: str) -> Optional[str]:
        """
        Infer which table field a correction applies to based on text context.
        """
        text_lower = text.lower()

        # Check each field's aliases
        for field, aliases in self.field_aliases.items():
            if any(alias in text_lower for alias in aliases):
                return field

        return None

    def _guess_field_from_context(
        self, text: str, table_row: Dict[str, Any], new_value: float
    ) -> Optional[str]:
        """
        Guess the field when direct alias matching fails.
        Uses context clues and value comparison.
        """
        text_lower = text.lower()

        # Look for death-related keywords
        if any(word in text_lower for word in ["death", "died", "fatal", "mortality"]):
            return "Deaths"

        # Look for case-related keywords
        if any(word in text_lower for word in ["case", "infection", "patient"]):
            # Check if it's close to existing TotalCases or CasesConfirmed
            total_cases = self._normalize_number(str(table_row.get("TotalCases", "")))
            confirmed_cases = self._normalize_number(
                str(table_row.get("CasesConfirmed", ""))
            )

            if total_cases and abs(new_value - total_cases) < total_cases * 0.1:
                return "TotalCases"
            elif (
                confirmed_cases
                and abs(new_value - confirmed_cases) < confirmed_cases * 0.1
            ):
                return "CasesConfirmed"
            else:
                return "TotalCases"  # Default to total cases

        return None

    def _calculate_confidence(
        self,
        relevance_score: float,
        text: str,
        field: str,
        old_value: Any,
        new_value: float,
    ) -> float:
        """
        Calculate confidence score for a correction.
        """
        base_confidence = relevance_score

        # Bonus for explicit field mention
        if field.lower() in text.lower():
            base_confidence += 0.15

        # Bonus for reasonable value changes
        if old_value is not None:
            try:
                old_num = self._normalize_number(str(old_value))
                if old_num is not None and old_num > 0:
                    # Small relative change increases confidence
                    relative_change = abs(new_value - old_num) / old_num
                    if relative_change < 0.5:  # Less than 50% change
                        base_confidence += 0.1
            except (ValueError, TypeError):
                pass

        return max(0.0, min(1.0, base_confidence))

    def _deduplicate_corrections(
        self, corrections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Keep only the best correction per field (highest confidence).
        """
        best_by_field = {}

        for correction in sorted(
            corrections, key=lambda x: x["confidence"], reverse=True
        ):
            field = correction["field"]
            if field not in best_by_field:
                best_by_field[field] = correction

        return list(best_by_field.values())

    # Legacy method for backward compatibility
    def link_narrative(self, text_blocks: List[Dict]) -> Dict[str, Any]:
        """
        Legacy method for backward compatibility.
        Note: This doesn't do proper row/field linking.
        """
        logger.warning("Using legacy link_narrative method - no row/field linking")

        corrections = []
        for block in text_blocks:
            # Simple pattern detection without row linking
            for pattern_name, pattern_re in [
                ("revision", RevisionRE),
                ("update", UpdateRE),
                ("should_be", ShouldBeRE),
            ]:
                match = pattern_re.search(block.get("text", ""))
                if match:
                    corrections.append(
                        {
                            "new_value": match.group(1),
                            "correction_type": pattern_name,
                            "provenance": {
                                "page": block.get("page", 0),
                                "text": block.get("text", ""),
                            },
                            "text": block.get("text", ""),
                        }
                    )

        return {
            "corrections": corrections,
            "total_corrections": len(corrections),
            "method": "regex_legacy",
        }
        """
        Detect corrections in narrative text using multiple patterns.

        Args:
            block: Text block dictionary

        Returns:
            Correction dictionary or None
        """
        text = block.get("text", "")

        # Pattern 1: Traditional revision language
        revision_pattern = re.compile(r"revis(ed|ion).*?(\d[\d,\.]*)", re.I)
        match = revision_pattern.search(text)
        if match:
            return {
                "new_value": match.group(2),
                "correction_type": "revision",
                "provenance": {
                    "page": block.get("page", 0),
                    "text": text,
                },
                "text": text,
            }

        # Pattern 2: Number formatting corrections (malformed -> corrected)
        # Look for patterns like "27,16" followed by "27,160" in same context
        formatting_pattern = re.compile(
            r"(\d{1,3}),(\d{1,2})\b.*?(\d{1,3})[,\s](\d{3})", re.I
        )
        match = formatting_pattern.search(text)
        if match:
            malformed = f"{match.group(1)},{match.group(2)}"
            corrected = (
                f"{match.group(3)},{match.group(4)}"
                if "," in text[match.start(3) : match.end(4)]
                else f"{match.group(3)} {match.group(4)}"
            )

            # Only consider it a correction if the numbers are related (same base)
            if match.group(1) == match.group(3):
                return {
                    "new_value": corrected.replace(
                        " ", ","
                    ),  # Normalize to comma format
                    "old_value": malformed,
                    "correction_type": "formatting",
                    "provenance": {
                        "page": block.get("page", 0),
                        "text": text,
                    },
                    "text": text,
                }

        # Pattern 3: Update/correction language
        update_pattern = re.compile(
            r"(?:updat(?:ed|e)|correct(?:ed|ion))\s+(?:to\s+)?(\d[\d,\.\s]*)", re.I
        )
        match = update_pattern.search(text)
        if match:
            return {
                "new_value": match.group(1).replace(" ", ","),  # Normalize spacing
                "correction_type": "update",
                "provenance": {
                    "page": block.get("page", 0),
                    "text": text,
                },
                "text": text,
            }

        # Pattern 4: "should be" corrections
        should_be_pattern = re.compile(r"should\s+be\s+(\d[\d,\.\s]*)", re.I)
        match = should_be_pattern.search(text)
        if match:
            return {
                "new_value": match.group(1).replace(" ", ","),
                "correction_type": "should_be",
                "provenance": {
                    "page": block.get("page", 0),
                    "text": text,
                },
                "text": text,
            }

        return None
