"""
Test module for data comparison functionality.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.compare import DataComparator, perform_discrepancy_analysis


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


def _records(rows):
    """Build an extraction DataFrame from (country, event, date, cases, deaths) rows."""
    return pd.DataFrame(
        rows,
        columns=["Country", "Event", "DateNotified", "TotalCases", "Deaths"],
    ).assign(CasesConfirmed=0, CFR=0.0, Grade="Grade 3")


class TestPerformDiscrepancyAnalysis:
    """Record matching between an LLM and a rule-based extraction."""

    def test_repeated_country_event_rows_are_paired_one_to_one(self):
        # A bulletin lists two crises for Chad under the same event name, and
        # the extractions return them in opposite order. Each row has an exact
        # counterpart, so nothing should be reported.
        llm = _records(
            [
                ("Chad", "Complex Humanitarian crisis", "15-Apr-23", 1574102, 0),
                ("Chad", "Complex Humanitarian crisis", "11-Feb-22", 6100000, 0),
            ]
        )
        rule_based = llm.iloc[::-1].reset_index(drop=True)

        discrepancies, common, llm_only, rule_based_only = perform_discrepancy_analysis(
            llm, rule_based
        )

        assert len(discrepancies) == 0
        assert len(common) == 2
        assert len(llm_only) == 0 and len(rule_based_only) == 0

    def test_real_difference_in_a_repeated_row_is_still_reported(self):
        # Pairing must not hide a genuine disagreement: the second Chad row
        # differs in Deaths only, and that one difference is what comes back.
        llm = _records(
            [
                ("Chad", "Complex Humanitarian crisis", "15-Apr-23", 1574102, 0),
                ("Chad", "Complex Humanitarian crisis", "11-Feb-22", 6100000, 5),
            ]
        )
        rule_based = llm.assign(Deaths=[0, 9])

        discrepancies, _, _, _ = perform_discrepancy_analysis(llm, rule_based)

        assert len(discrepancies) == 1
        row = discrepancies.iloc[0]
        assert (row["llm_Deaths"], row["rule_based_Deaths"]) == (5, 9)
        assert not row["TotalCases_discrepancy"]

    def test_extra_repeated_row_is_reported_as_only_in_that_extraction(self):
        # The rule-based extraction emits the Zambia row twice. The surplus row
        # has no LLM partner, so it is a rule-based-only record.
        llm = _records([("Zambia", "Cholera", "30-Dec-24", 1094, 18)])
        rule_based = pd.concat([llm, llm], ignore_index=True)

        discrepancies, common, llm_only, rule_based_only = perform_discrepancy_analysis(
            llm, rule_based
        )

        assert len(discrepancies) == 0
        assert len(common) == 1
        assert len(llm_only) == 0
        assert len(rule_based_only) == 1

    def test_spelling_variants_match_as_the_same_record(self):
        # Accents, capitalisation, punctuation and spacing differ between the
        # extractions; these are the same records, not unique ones.
        llm = _records(
            [
                ("Côte d'Ivoire", "Yellow fever", "15-Feb-26", 14, 2),
                ("Zambia", "Poliomyelitis (cVDPV2)", "18-Feb-26", 0, 0),
            ]
        )
        rule_based = _records(
            [
                ("Cote d'Ivoire", "Yellow Fever", "15-Feb-26", 14, 2),
                ("Zambia", "Poliomyelitis (c VDPV2)", "18-Feb-26", 0, 0),
            ]
        )

        discrepancies, common, llm_only, rule_based_only = perform_discrepancy_analysis(
            llm, rule_based
        )

        assert len(discrepancies) == 0
        assert len(common) == 2
        assert len(llm_only) == 0 and len(rule_based_only) == 0

    def test_different_events_stay_unmatched(self):
        # Normalising names must not merge records that are genuinely different.
        llm = _records([("Niger", "Diphtheria", "15-Jan-26", 755, 24)])
        rule_based = _records([("Nigeria", "Diphtheria", "30-Jan-26", 16997, 596)])

        _, common, llm_only, rule_based_only = perform_discrepancy_analysis(
            llm, rule_based
        )

        assert len(common) == 0
        assert len(llm_only) == 1 and len(rule_based_only) == 1
