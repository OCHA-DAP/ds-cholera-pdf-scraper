"""
Test module for LLM extraction functionality.
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from llm_extract import LLMExtractor


class TestLLMExtractor:
    """Test cases for LLMExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create test extractor instance."""
        return LLMExtractor(api_key="test-key", model="gpt-4o", temperature=0.1)

    def test_init(self, extractor):
        """Test extractor initialization."""
        assert extractor.model == "gpt-4o"
        assert extractor.temperature == 0.1
        assert extractor.client is not None

    @patch("openai.OpenAI")
    def test_extract_data_from_text(self, mock_openai, extractor):
        """Test data extraction from text."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices[
            0
        ].message.content = """{
            "reporting_date": "2024-01-15",
            "country": "Haiti",
            "suspected_cases": 150,
            "confirmed_cases": 75,
            "deaths": 5
        }"""

        mock_openai.return_value.chat.completions.create.return_value = mock_response
        extractor.client = mock_openai.return_value

        # Test extraction
        result = extractor.extract_data_from_text("Sample PDF text")

        assert result["country"] == "Haiti"
        assert result["suspected_cases"] == 150
        assert result["confirmed_cases"] == 75

    def test_create_extraction_prompt(self, extractor):
        """Test prompt creation."""
        pdf_text = "Sample cholera report text"
        prompt = extractor.create_extraction_prompt(pdf_text)

        assert "cholera surveillance reports" in prompt
        assert "JSON" in prompt
        assert pdf_text in prompt

    @patch.object(LLMExtractor, "extract_text_from_pdf")
    @patch.object(LLMExtractor, "extract_data_from_text")
    def test_process_pdf_file(self, mock_extract_data, mock_extract_text, extractor):
        """Test end-to-end PDF processing."""
        # Mock dependencies
        mock_extract_text.return_value = "Sample PDF text"
        mock_extract_data.return_value = {"country": "Haiti", "suspected_cases": 100}

        # Test processing
        pdf_path = Path("test.pdf")
        result = extractor.process_pdf_file(pdf_path)

        assert result["country"] == "Haiti"
        assert result["suspected_cases"] == 100
        assert "source_file" in result
        assert "extraction_timestamp" in result

    def test_process_multiple_pdfs(self, extractor):
        """Test processing multiple PDFs."""
        with patch.object(extractor, "process_pdf_file") as mock_process:
            mock_process.return_value = {"country": "Haiti", "cases": 100}

            pdf_paths = [Path("test1.pdf"), Path("test2.pdf")]
            results = extractor.process_multiple_pdfs(pdf_paths)

            assert len(results) == 2
            assert mock_process.call_count == 2
