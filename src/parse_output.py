"""
Parse LLM output into structured pandas DataFrames.

This module handles parsing LLM responses and converting them into
pandas DataFrames that match the baseline schema.
"""

import logging
from typing import Dict, List, Any
import pandas as pd
import json
from datetime import datetime


logger = logging.getLogger(__name__)


class OutputParser:
    """Parses LLM extraction output into structured DataFrames."""

    def __init__(self, baseline_schema: Dict[str, str] = None):
        """
        Initialize the output parser.

        Args:
            baseline_schema: Schema definition for baseline comparison
        """
        self.baseline_schema = baseline_schema or self._get_default_schema()

    def _get_default_schema(self) -> Dict[str, str]:
        """
        Get the default schema for cholera data.

        Returns:
            Schema dictionary with column names and types
        """
        return {
            "reporting_date": "datetime64[ns]",
            "country": "string",
            "admin1": "string",
            "admin2": "string",
            "suspected_cases": "Int64",
            "confirmed_cases": "Int64",
            "deaths": "Int64",
            "case_fatality_rate": "float64",
            "population_at_risk": "Int64",
            "reporting_period_start": "datetime64[ns]",
            "reporting_period_end": "datetime64[ns]",
            "source_file": "string",
            "extraction_timestamp": "datetime64[ns]",
        }

    def validate_extracted_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean extracted data.

        Args:
            data: Raw extracted data dictionary

        Returns:
            Cleaned and validated data dictionary
        """
        logger.debug(f"Validating extracted data: {data}")

        cleaned_data = {}

        for field, expected_type in self.baseline_schema.items():
            value = data.get(field)

            try:
                if value is None or value == "":
                    cleaned_data[field] = None
                elif "datetime" in expected_type:
                    cleaned_data[field] = self._parse_date(value)
                elif "Int64" in expected_type:
                    cleaned_data[field] = self._parse_integer(value)
                elif "float64" in expected_type:
                    cleaned_data[field] = self._parse_float(value)
                else:
                    cleaned_data[field] = str(value) if value is not None else None

            except Exception as e:
                logger.warning(f"Failed to parse field {field}={value}: {e}")
                cleaned_data[field] = None

        return cleaned_data

    def _parse_date(self, value: Any) -> pd.Timestamp:
        """Parse various date formats into pandas Timestamp."""
        if value is None:
            return None

        if isinstance(value, str):
            try:
                return pd.to_datetime(value)
            except Exception:
                logger.warning(f"Could not parse date: {value}")
                return None

        return pd.to_datetime(value)

    def _parse_integer(self, value: Any) -> int:
        """Parse integer values, handling various formats."""
        if value is None or value == "":
            return None

        if isinstance(value, str):
            # Remove commas, spaces, and other formatting
            clean_value = value.replace(",", "").replace(" ", "").strip()
            try:
                return int(float(clean_value))  # Handle decimal strings
            except ValueError:
                return None

        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    def _parse_float(self, value: Any) -> float:
        """Parse float values, handling various formats."""
        if value is None or value == "":
            return None

        if isinstance(value, str):
            clean_value = value.replace(",", "").replace(" ", "").strip()
            if clean_value.endswith("%"):
                clean_value = clean_value[:-1]
                try:
                    return float(clean_value)  # Keep as percentage value
                except ValueError:
                    return None
            try:
                return float(clean_value)
            except ValueError:
                return None

        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def parse_single_extraction(self, extraction_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Parse a single LLM extraction into a DataFrame row.

        Args:
            extraction_data: Dictionary from LLM extraction

        Returns:
            Single-row DataFrame
        """
        logger.debug("Parsing single extraction into DataFrame")

        # Validate and clean the data
        cleaned_data = self.validate_extracted_data(extraction_data)

        # Create DataFrame
        df = pd.DataFrame([cleaned_data])

        # Apply schema types
        for column, dtype in self.baseline_schema.items():
            if column in df.columns:
                try:
                    df[column] = df[column].astype(dtype)
                except Exception as e:
                    logger.warning(f"Could not convert {column} to {dtype}: {e}")

        return df

    def parse_multiple_extractions(
        self, extractions_list: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Parse multiple LLM extractions into a combined DataFrame.

        Args:
            extractions_list: List of extraction dictionaries

        Returns:
            Combined DataFrame with all extractions
        """
        logger.info(f"Parsing {len(extractions_list)} extractions into DataFrame")

        if not extractions_list:
            # Return empty DataFrame with correct schema
            empty_df = pd.DataFrame(columns=list(self.baseline_schema.keys()))
            for column, dtype in self.baseline_schema.items():
                empty_df[column] = empty_df[column].astype(dtype)
            return empty_df

        parsed_dfs = []

        for i, extraction_data in enumerate(extractions_list):
            try:
                df_row = self.parse_single_extraction(extraction_data)
                parsed_dfs.append(df_row)

            except Exception as e:
                logger.error(f"Failed to parse extraction {i}: {e}")
                continue

        if not parsed_dfs:
            logger.warning("No successful extractions to combine")
            return pd.DataFrame(columns=list(self.baseline_schema.keys()))

        # Combine all DataFrames
        combined_df = pd.concat(parsed_dfs, ignore_index=True)

        logger.info(f"Successfully parsed {len(combined_df)} records")
        return combined_df

    def save_to_csv(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save DataFrame to CSV file.

        Args:
            df: DataFrame to save
            output_path: Path for output CSV file
        """
        logger.info(f"Saving DataFrame to {output_path}")

        try:
            df.to_csv(output_path, index=False)
            logger.info(f"Successfully saved {len(df)} records to {output_path}")

        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
            raise

    def load_baseline_for_comparison(self, baseline_path: str) -> pd.DataFrame:
        """
        Load baseline DataFrame for comparison.

        Args:
            baseline_path: Path to baseline CSV file

        Returns:
            Baseline DataFrame
        """
        logger.info(f"Loading baseline data from {baseline_path}")

        try:
            baseline_df = pd.read_csv(baseline_path)

            # Apply schema if needed
            for column, dtype in self.baseline_schema.items():
                if column in baseline_df.columns:
                    try:
                        baseline_df[column] = baseline_df[column].astype(dtype)
                    except Exception as e:
                        logger.warning(f"Could not convert baseline {column}: {e}")

            logger.info(f"Loaded baseline with {len(baseline_df)} records")
            return baseline_df

        except Exception as e:
            logger.error(f"Failed to load baseline data: {e}")
            raise


def main():
    """Main execution function for testing."""
    logging.basicConfig(level=logging.INFO)

    # Example usage
    parser = OutputParser()

    # Sample extraction data
    sample_data = {
        "reporting_date": "2024-01-15",
        "country": "Haiti",
        "admin1": "Ouest",
        "suspected_cases": 150,
        "confirmed_cases": 75,
        "deaths": 5,
        "case_fatality_rate": 3.3,
        "source_file": "haiti_report_2024_01.pdf",
    }

    df = parser.parse_single_extraction(sample_data)
    print(f"Parsed DataFrame:\n{df}")


if __name__ == "__main__":
    main()
