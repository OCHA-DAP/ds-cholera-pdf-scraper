"""
Test module for output parsing functionality.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from parse_output import OutputParser


class TestOutputParser:
    """Test cases for OutputParser class."""

    @pytest.fixture
    def parser(self):
        """Create test parser instance."""
        return OutputParser()

    @pytest.fixture
    def sample_data(self):
        """Sample extraction data for testing."""
        return {
            "reporting_date": "2024-01-15",
            "country": "Haiti",
            "admin1": "Ouest",
            "suspected_cases": "150",
            "confirmed_cases": 75,
            "deaths": 5,
            "case_fatality_rate": "3.3%",
            "source_file": "test.pdf",
        }

    def test_init(self, parser):
        """Test parser initialization."""
        assert parser.baseline_schema is not None
        assert "reporting_date" in parser.baseline_schema
        assert "country" in parser.baseline_schema

    def test_parse_date(self, parser):
        """Test date parsing."""
        assert parser._parse_date("2024-01-15") == pd.Timestamp("2024-01-15")
        assert parser._parse_date(None) is None
        assert parser._parse_date("invalid") is None

    def test_parse_integer(self, parser):
        """Test integer parsing."""
        assert parser._parse_integer("150") == 150
        assert parser._parse_integer("1,500") == 1500
        assert parser._parse_integer(None) is None
        assert parser._parse_integer("invalid") is None

    def test_parse_float(self, parser):
        """Test float parsing."""
        assert parser._parse_float("3.3") == 3.3
        assert parser._parse_float("3.3%") == 3.3
        assert parser._parse_float("1,500.50") == 1500.50
        assert parser._parse_float(None) is None

    def test_validate_extracted_data(self, parser, sample_data):
        """Test data validation."""
        cleaned = parser.validate_extracted_data(sample_data)

        assert isinstance(cleaned["reporting_date"], pd.Timestamp)
        assert cleaned["country"] == "Haiti"
        assert cleaned["suspected_cases"] == 150
        assert cleaned["case_fatality_rate"] == 3.3

    def test_parse_single_extraction(self, parser, sample_data):
        """Test single extraction parsing."""
        df = parser.parse_single_extraction(sample_data)

        assert len(df) == 1
        assert df["country"].iloc[0] == "Haiti"
        assert df["suspected_cases"].iloc[0] == 150

    def test_parse_multiple_extractions(self, parser, sample_data):
        """Test multiple extractions parsing."""
        extractions = [sample_data, sample_data.copy()]
        extractions[1]["country"] = "Somalia"

        df = parser.parse_multiple_extractions(extractions)

        assert len(df) == 2
        assert "Haiti" in df["country"].values
        assert "Somalia" in df["country"].values

    def test_parse_empty_extractions(self, parser):
        """Test parsing empty extraction list."""
        df = parser.parse_multiple_extractions([])

        assert len(df) == 0
        assert list(df.columns) == list(parser.baseline_schema.keys())
