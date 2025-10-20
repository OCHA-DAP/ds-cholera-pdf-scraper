# Batch Run Analysis Module

A clean, modular interface for comparing batch LLM extraction results against baseline rule-based scraper data.

## Overview

This module provides functions to:
- Load batch extraction results from multiple CSV files
- Load and standardize baseline comparison data
- Perform discrepancy analysis between batch and baseline
- Categorize discrepancies by type
- Generate summary statistics

## Quick Start

```python
from src.batch_run import (
    load_batch_data,
    load_baseline_data,
    analyze_batch_vs_baseline,
    categorize_discrepancies,
    create_summary_statistics
)

# Load data
batch_df, batch_list = load_batch_data("outputs/batch_run")
baseline_df = load_baseline_data("data/final_data_for_powerbi_with_kpi.csv")

# Run analysis (with optional gap-filling corrections)
results, combined_discrepancies = analyze_batch_vs_baseline(
    batch_df,
    baseline_df,
    correct_gap_fill_errors=True,  # Optional: enable experimental corrections
    verbose=True
)

# Create summary statistics
summary = create_summary_statistics(results)

# Categorize discrepancies
categorized = categorize_discrepancies(combined_discrepancies)
```

## Module Structure

```
src/batch_run/
├── __init__.py          # Main interface - import from here
├── loader.py            # Data loading functions
├── analyzer.py          # Analysis and comparison functions
├── test_batch_run.py    # Test suite
└── README.md            # This file
```

## Functions

### Data Loading (`loader.py`)

#### `load_batch_data(batch_dir, add_metadata=True)`
Load all batch extraction CSV files from a directory.

**Args:**
- `batch_dir`: Directory containing batch CSV files (default: "outputs/batch_run")
- `add_metadata`: Add WeekNumber, Year, SourceFile columns (default: True)

**Returns:**
- `(combined_df, list_of_dfs)`: All data combined + list of individual DataFrames

**Example:**
```python
batch_df, batch_list = load_batch_data("outputs/batch_run")
print(f"Loaded {len(batch_df)} records from {len(batch_list)} files")
```

#### `load_baseline_data(baseline_path, standardize=True)`
Load baseline rule-based scraper data.

**Args:**
- `baseline_path`: Path to baseline CSV (default: "data/final_data_for_powerbi_with_kpi.csv")
- `standardize`: Standardize column names (default: True)

**Returns:**
- `baseline_df`: DataFrame with baseline data

**Example:**
```python
baseline_df = load_baseline_data()
```

#### `parse_week_year_from_filename(filename)`
Extract week number and year from filename.

**Example:**
```python
week, year = parse_week_year_from_filename("Week_24__09_to_15_June_2025.csv")
# Returns: (24, 2025)
```

### Analysis (`analyzer.py`)

#### `analyze_batch_vs_baseline(batch_df, baseline_df, correct_gap_fill_errors=False, verbose=True)`
Main analysis function - compare all batch runs with baseline.

**Args:**
- `batch_df`: Batch extraction data (from `load_batch_data`)
- `baseline_df`: Baseline data (from `load_baseline_data`)
- `correct_gap_fill_errors`: Enable experimental gap-filling corrections (default: False)
- `verbose`: Print progress information (default: True)

**Returns:**
- `(results_list, combined_discrepancies_df)`: Analysis results + all discrepancies

**Example:**
```python
# Without corrections (default)
results, discrepancies = analyze_batch_vs_baseline(batch_df, baseline_df)

# With experimental corrections
results, discrepancies = analyze_batch_vs_baseline(
    batch_df,
    baseline_df,
    correct_gap_fill_errors=True
)
```

#### `analyze_single_week(batch_df, baseline_df, week, year, correct_gap_fill_errors=False)`
Analyze a specific week/year combination.

**Args:**
- `batch_df`: Batch data
- `baseline_df`: Baseline data
- `week`: Week number
- `year`: Year
- `correct_gap_fill_errors`: Apply corrections (default: False)

**Returns:**
- Dictionary with analysis results for that week

**Example:**
```python
result = analyze_single_week(batch_df, baseline_df, week=24, year=2025)
print(f"Batch: {result['batch_count']}, Baseline: {result['baseline_count']}")
```

#### `categorize_discrepancies(discrepancies_df)`
Categorize discrepancies by type.

**Categories:**
- Zero vs Non-Zero
- Comma/Thousands Issue
- Large Difference (>100)
- Medium Difference (21-100)
- Small Difference (≤20)

**Args:**
- `discrepancies_df`: DataFrame with discrepancies

**Returns:**
- DataFrame with added 'Category' and 'SubCategory' columns

**Example:**
```python
categorized = categorize_discrepancies(combined_discrepancies)
print(categorized['Category'].value_counts())
```

#### `create_summary_statistics(results_list)`
Create week-by-week summary table.

**Args:**
- `results_list`: List of analysis results from `analyze_batch_vs_baseline`

**Returns:**
- DataFrame with summary statistics by week

**Example:**
```python
summary = create_summary_statistics(results)
print(summary[['Week', 'Year', 'BatchRecords', 'ValueDiscrepancies']])
```

## Gap-Filling Corrections

The `correct_gap_fill_errors` parameter enables experimental post-processing corrections:

**What it does:**
1. **Pattern 1**: `CasesConfirmed == TotalCases` for non-protracted events → Set to 0
2. **Pattern 2**: Deaths inconsistent with CFR → Recalculate from CFR

**When to use:**
- ✅ Experimental analysis (like Chapter 03)
- ✅ When you want to test correction effectiveness
- ❌ Production pipelines (not yet validated)
- ❌ When preserving raw LLM output is important

**Example comparison:**
```python
# Without corrections
results_raw, disc_raw = analyze_batch_vs_baseline(batch_df, baseline_df)
print(f"Discrepancies: {len(disc_raw)}")

# With corrections
results_corrected, disc_corrected = analyze_batch_vs_baseline(
    batch_df, baseline_df, correct_gap_fill_errors=True
)
print(f"Discrepancies: {len(disc_corrected)}")
```

## Testing

Run the test suite:

```bash
python -m src.batch_run.test_batch_run
```

This will:
1. Load batch and baseline data
2. Run analysis with corrections OFF and ON
3. Test categorization and summary statistics
4. Compare results

## Integration with Notebooks

**Chapter 03 (Batch Run Analysis)** uses this module:

```python
from src.batch_run import (
    load_batch_data,
    load_baseline_data,
    analyze_batch_vs_baseline,
    categorize_discrepancies,
    create_summary_statistics
)

# Load raw data
batch_df, batch_list = load_batch_data("outputs/batch_run")
baseline_df = load_baseline_data("data/final_data_for_powerbi_with_kpi.csv")

# Analyze with corrections
results, combined_discrepancies = analyze_batch_vs_baseline(
    batch_df,
    baseline_df,
    correct_gap_fill_errors=True,  # Enable for Chapter 03 analysis
    verbose=True
)

# Create summary
summary_by_week = create_summary_statistics(results)
disc_cat = categorize_discrepancies(combined_discrepancies)
```

## Migration from tmp/ Scripts

This module replaces:
- `tmp/batch_run_discrepancy_analysis.py` → Use `analyze_batch_vs_baseline()`
- Manual CSV reading → Use `load_batch_data()` and `load_baseline_data()`
- Hardcoded analysis → Use configurable functions with `correct_gap_fill_errors` parameter

**Old way (tmp/ scripts):**
```python
# Hardcoded in script, corrections always applied
results = main()
```

**New way (batch_run module):**
```python
# Flexible, corrections optional
results, discrepancies = analyze_batch_vs_baseline(
    batch_df,
    baseline_df,
    correct_gap_fill_errors=True  # Explicit choice
)
```

## Next Steps

After confirming everything works:
1. Run Chapter 03 notebook to verify it works with new module
2. Delete `tmp/batch_run_discrepancy_analysis.py` (no longer needed)
3. Delete pre-computed CSV files in `tmp/` (can be regenerated from raw data)

---

**Benefits of Refactored Module:**
- ✅ Cleaner interface - import functions, not scripts
- ✅ Notebooks read raw data - more transparent
- ✅ Gap-filling corrections are explicit and optional
- ✅ Reusable across different analyses
- ✅ Easy to test and maintain
