"""
Test module for data comparison functionality.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.compare import DataComparator


class TestDataComparator:
    """Test cases for DataComparator class."""

    @pytest.fixture
    def comparator(self):
        """Create test comparator instance."""
        return DataComparator(tolerance=0.1)

    @pytest.fixture
    def sample_llm_data(self):
        """Sample LLM data for testing."""
        return pd.DataFrame(
            {
                "reporting_date": ["2024-01-15", "2024-01-16"],
                "country": ["Haiti", "Somalia"],
                "admin1": ["Ouest", "Banadir"],
                "suspected_cases": [150, 200],
                "confirmed_cases": [75, 100],
                "deaths": [5, 8],
            }
        )

    @pytest.fixture
    def sample_baseline_data(self):
        """Sample baseline data for testing."""
        return pd.DataFrame(
            {
                "reporting_date": ["2024-01-15", "2024-01-16"],
                "country": ["Haiti", "Somalia"],
                "admin1": ["Ouest", "Banadir"],
                "suspected_cases": [145, 190],  # Slightly different
                "confirmed_cases": [70, 95],  # Slightly different
                "deaths": [5, 8],  # Same
            }
        )

    def test_init(self, comparator):
        """Test comparator initialization."""
        assert comparator.tolerance == 0.1
        assert isinstance(comparator.comparison_results, dict)

    def test_align_dataframes(self, comparator, sample_llm_data, sample_baseline_data):
        """Test DataFrame alignment."""
        aligned_llm, aligned_baseline = comparator.align_dataframes(
            sample_llm_data, sample_baseline_data
        )

        assert len(aligned_llm) == len(aligned_baseline)
        assert "_comparison_key" in aligned_llm.columns
        assert "_comparison_key" in aligned_baseline.columns

    def test_compare_numerical_columns(
        self, comparator, sample_llm_data, sample_baseline_data
    ):
        """Test numerical column comparison."""
        results = comparator.compare_numerical_columns(
            sample_llm_data,
            sample_baseline_data,
            ["suspected_cases", "confirmed_cases", "deaths"],
        )

        assert "suspected_cases" in results
        assert "mae" in results["suspected_cases"]
        assert "rmse" in results["suspected_cases"]
        assert "correlation" in results["suspected_cases"]
        assert "within_tolerance_pct" in results["suspected_cases"]

    def test_compare_categorical_columns(
        self, comparator, sample_llm_data, sample_baseline_data
    ):
        """Test categorical column comparison."""
        results = comparator.compare_categorical_columns(
            sample_llm_data, sample_baseline_data, ["country", "admin1"]
        )

        assert "country" in results
        assert "exact_match_pct" in results["country"]
        assert results["country"]["exact_match_pct"] == 100.0  # Perfect match

    def test_numerical_accuracy_calculation(self, comparator):
        """Test numerical accuracy calculations."""
        llm_df = pd.DataFrame({"values": [100, 200, 300]})
        baseline_df = pd.DataFrame({"values": [95, 210, 290]})  # Within 10% tolerance

        results = comparator.compare_numerical_columns(llm_df, baseline_df, ["values"])

        # All values should be within 10% tolerance
        assert results["values"]["within_tolerance_pct"] == 100.0

    def test_categorical_accuracy_calculation(self, comparator):
        """Test categorical accuracy calculations."""
        llm_df = pd.DataFrame({"category": ["A", "B", "C", "D"]})
        baseline_df = pd.DataFrame({"category": ["A", "B", "X", "D"]})  # 3/4 match

        results = comparator.compare_categorical_columns(
            llm_df, baseline_df, ["category"]
        )

        assert results["category"]["exact_match_pct"] == 75.0
