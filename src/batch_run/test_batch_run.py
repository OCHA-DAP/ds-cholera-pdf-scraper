"""
Test script for batch_run module.

Run with: python -m src.batch_run.test_batch_run
"""

import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from src.batch_run import (
    load_llm_data,
    load_rule_based_data,
    analyze_llm_vs_rule_based,
    categorize_discrepancies,
    create_summary_statistics,
)


def test_data_loading():
    """Test data loading functions."""
    print("="*80)
    print("TEST 1: Data Loading")
    print("="*80)

    # Load LLM data
    print("\n1. Loading LLM data...")
    try:
        llm_df, llm_list = load_llm_data("outputs/batch_run")
        print(f"   ✓ Loaded {len(llm_df)} LLM records from {len(llm_list)} files")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    # Load rule-based data
    print("\n2. Loading rule-based data...")
    try:
        rule_based_df = load_rule_based_data("data/final_data_for_powerbi_with_kpi.csv")
        print(f"   ✓ Loaded {len(rule_based_df)} rule-based records")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    return llm_df, rule_based_df


def test_analysis(llm_df, rule_based_df, use_corrections=False):
    """Test analysis functions."""
    print("\n" + "="*80)
    print(f"TEST 2: Analysis (corrections={'ON' if use_corrections else 'OFF'})")
    print("="*80)

    try:
        results, combined_discrepancies = analyze_llm_vs_rule_based(
            llm_df,
            rule_based_df,
            correct_gap_fill_errors=use_corrections,
            verbose=False  # Suppress detailed output for test
        )

        print(f"\n   ✓ Analyzed {len(results)} weeks")

        if combined_discrepancies is not None:
            print(f"   ✓ Found {len(combined_discrepancies)} discrepancies")
        else:
            print(f"   ✓ No discrepancies found")

        return results, combined_discrepancies

    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_categorization(combined_discrepancies):
    """Test categorization functions."""
    print("\n" + "="*80)
    print("TEST 3: Discrepancy Categorization")
    print("="*80)

    if combined_discrepancies is None or len(combined_discrepancies) == 0:
        print("\n   ⚠️  No discrepancies to categorize")
        return None

    try:
        categorized = categorize_discrepancies(combined_discrepancies)
        print(f"\n   ✓ Categorized {len(categorized)} discrepancies")

        # Show category breakdown
        print("\n   Category breakdown:")
        for category, count in categorized['Category'].value_counts().items():
            pct = count / len(categorized) * 100
            print(f"      {category}: {count} ({pct:.1f}%)")

        return categorized

    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_summary_stats(results):
    """Test summary statistics creation."""
    print("\n" + "="*80)
    print("TEST 4: Summary Statistics")
    print("="*80)

    if not results:
        print("\n   ⚠️  No results to summarize")
        return None

    try:
        summary = create_summary_statistics(results)
        print(f"\n   ✓ Created summary for {len(summary)} weeks")
        print("\n   Summary preview:")
        print(summary[['Week', 'Year', 'LLMRecords', 'RuleBasedRecords', 'ValueDiscrepancies']].head())

        return summary

    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("LLM vs RULE-BASED MODULE TEST SUITE")
    print("="*80)

    # Test 1: Data loading
    data = test_data_loading()
    if not data:
        print("\n❌ Data loading failed. Cannot proceed with tests.")
        return False

    llm_df, rule_based_df = data

    # Test 2a: Analysis without corrections
    results_no_corr, disc_no_corr = test_analysis(llm_df, rule_based_df, use_corrections=False)

    # Test 2b: Analysis with corrections
    results_with_corr, disc_with_corr = test_analysis(llm_df, rule_based_df, use_corrections=True)

    # Test 3: Categorization
    if disc_with_corr is not None:
        categorized = test_categorization(disc_with_corr)

    # Test 4: Summary statistics
    if results_with_corr:
        summary = test_summary_stats(results_with_corr)

    # Final summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print("\n✅ All tests completed successfully!")
    print(f"\nComparison (corrections OFF vs ON):")

    if disc_no_corr is not None and disc_with_corr is not None:
        print(f"   Discrepancies without corrections: {len(disc_no_corr)}")
        print(f"   Discrepancies with corrections:    {len(disc_with_corr)}")
        print(f"   Difference: {len(disc_with_corr) - len(disc_no_corr):+d}")

    print("\n" + "="*80)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
