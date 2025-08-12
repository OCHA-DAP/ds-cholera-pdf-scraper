# Accuracy Metrics Logging System

This system calculates detailed accuracy metrics from LLM extraction results and logs them to the prompt logging database for tracking and analysis.

## Quick Start

### 🚀 Simple Commands
```bash
# Process all prompt versions
./backfill-accuracy all

# Process current prompt version only  
./backfill-accuracy current

# See what would be processed (dry run)
./backfill-accuracy dry-run

# Get help
./backfill-accuracy help
```

### 🎯 Advanced Commands
```bash
# Process specific version
python scripts/backfill_accuracy_metrics.py --version v1.1.0

# Force overwrite existing metrics
python scripts/backfill_accuracy_metrics.py --force

# Custom directories
python scripts/backfill_accuracy_metrics.py --outputs results --data baseline_data
```

## Components

### 1. `AccuracyMetricsCalculator` (`src/accuracy_metrics.py`)
- Calculates field-level accuracy metrics from discrepancy analysis
- Provides overall accuracy, coverage, and problematic field identification
- Generates human-readable summary reports

### 2. Enhanced `PromptLogger` (`src/prompt_logger.py`)
- Added `update_log_with_accuracy_metrics()` method
- Added `get_latest_log_for_prompt_version()` method
- Stores accuracy metrics in the `custom_metrics` field of existing logs

### 3. **Backfill Tool** (`scripts/backfill_accuracy_metrics.py`)
- Auto-discovers prompt-versioned extraction files
- Calculates accuracy metrics for any/all prompt versions
- Updates corresponding log entries in database
- Supports dry-run, force overwrite, and filtering options

### 4. **Simple Wrapper** (`./backfill-accuracy`)
- Easy terminal commands: `all`, `current`, `dry-run`, `help`
- Perfect for quick accuracy calculations

## Usage

### From QMD/Notebook (Interactive)
```python
# After running discrepancy analysis in QMD, add accuracy metrics
from accuracy_metrics import calculate_accuracy_from_qmd_results
from prompt_logger import PromptLogger

# Calculate metrics
accuracy_metrics = calculate_accuracy_from_qmd_results(
    discrepancies_df=discrepancies_df,
    llm_common=llm_common,
    llm_only_df=llm_only_df,
    baseline_only_df=baseline_only_df,
    prompt_version=prompt_version
)

# Log to database
logger = PromptLogger()
latest_log = logger.get_latest_log_for_prompt_version(prompt_version)
if latest_log:
    logger.update_log_with_accuracy_metrics(str(latest_log['id']), accuracy_metrics)
```

### From Command Line (Standalone)
```bash
# For current prompt version
./backfill-accuracy current

# For all versions
./backfill-accuracy all

# See what would be processed
./backfill-accuracy dry-run

# Advanced usage with specific version
python scripts/backfill_accuracy_metrics.py --version v1.1.0

# With custom directories
python scripts/backfill_accuracy_metrics.py --outputs results --data baseline_data
```

### From Script (Programmatic)
```python
from scripts.calculate_accuracy_metrics import perform_discrepancy_analysis
from accuracy_metrics import AccuracyMetricsCalculator

# Load your data...
llm_data = pd.read_csv("extraction_results.csv")
baseline_data = pd.read_csv("baseline_data.csv")

# Perform analysis
discrepancies_df, llm_common, llm_only_df, baseline_only_df = perform_discrepancy_analysis(
    llm_data, baseline_data
)

# Calculate metrics
calculator = AccuracyMetricsCalculator()
metrics = calculator.calculate_metrics_from_qmd_variables(...)
```

## Metrics Calculated

### Overall Metrics
- `overall_accuracy_percent`: Percentage of records with no discrepancies
- `overall_discrepancy_rate_percent`: Percentage of records with any discrepancy
- `coverage_rate_percent`: Percentage of baseline records found by LLM
- `total_compared_records`: Number of records in both LLM and baseline
- `llm_only_records`: Records found only in LLM output
- `baseline_only_records`: Records found only in baseline

### Field-Level Metrics (for each field: TotalCases, CasesConfirmed, Deaths, CFR, Grade)
- `{field}_accuracy_percent`: Accuracy for this specific field
- `{field}_discrepancy_count`: Number of discrepant values for this field
- `{field}_discrepancy_rate_percent`: Percentage error rate for this field

### Additional Analysis
- `problematic_fields`: List of fields with >10% error rate, sorted by severity

## Database Storage

Accuracy metrics are stored in the `custom_metrics` JSON field of prompt logs:

```json
{
  "accuracy_metrics": {
    "prompt_version": "v1.1.0",
    "overall_accuracy_percent": 85.2,
    "coverage_rate_percent": 92.3,
    "field_accuracy_metrics": {
      "TotalCases_accuracy_percent": 88.1,
      "CasesConfirmed_accuracy_percent": 82.3,
      ...
    },
    "problematic_fields": [
      {"field": "CasesConfirmed", "discrepancy_rate": 17.7},
      ...
    ]
  }
}
```

## Integration with Existing Workflow

1. **During LLM Extraction**: Regular prompt logging continues as normal
2. **After Analysis**: Run accuracy calculation (manual or automated)
3. **Metrics Storage**: Accuracy metrics are attached to the existing log entry
4. **Reporting**: Query database for accuracy trends across prompt versions

## Retroactive Analysis

To add accuracy metrics to past runs:

```bash
# Calculate for all available prompt versions
for version in v1.0.0 v1.1.0 v2.0.0; do
  python scripts/backfill_accuracy_metrics.py --version $version
done
```

This system provides comprehensive accuracy tracking while keeping the LLM extraction process fast and simple, with post-processing accuracy analysis as a separate, modular step.
