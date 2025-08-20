# Prompt Engineering Guide

This guide shows you how to create and test new prompt versions for the cholera data extraction system.

## Quick Start: Creating a New Prompt

The simplest workflow is:

1. **Create a new markdown prompt file**:
   ```bash
   cp prompts/markdown/health_data_extraction/v1.1.2.md \
      prompts/markdown/health_data_extraction/v1.1.3.md
   ```

2. **Edit your prompt** in the new file:
   ```bash
   vim prompts/markdown/health_data_extraction/v1.2.0.md
   # Update the version number in the YAML frontmatter
   # Make your prompt improvements in the markdown content
   ```

3. **Run extraction with your new prompt**:
   ```bash
   python src/llm_text_extract.py --prompt-version v1.2.0
   ```

That's it! The system automatically:
- Converts your markdown to JSON format
- Stores the JSON in `prompts/health_data_extraction/v1.2.0.json`
- Logs all extractions with the prompt version in SQLite
- Tracks which files were processed with which prompt version

## How File Tracking Works

### Automatic JSON Generation
When you run extraction with a new prompt version, the system:
1. Checks if `prompts/health_data_extraction/v1.2.0.json` exists
2. If not, automatically imports from `prompts/markdown/health_data_extraction/v1.2.0.md`
3. Creates the JSON file for runtime use
4. Uses the JSON file for all LLM API calls

### Extraction Logging
Every extraction is logged in SQLite (`outputs/extractions.db`) with:
- **prompt_version**: Which prompt was used (e.g., "v1.2.0")
- **source_file**: Which PDF was processed
- **extraction_timestamp**: When it was processed
- **full_response**: Complete LLM response
- **extracted_data**: Parsed structured data

### Output File Organization
Extraction results are saved by prompt version:
```
outputs/
├── v1.1.0/
│   ├── baseline_comparison.csv
│   └── extracted_data.csv
├── v1.1.2/
│   ├── baseline_comparison.csv
│   └── extracted_data.csv
└── v1.2.0/                    # Your new version
    ├── baseline_comparison.csv
    └── extracted_data.csv
```

## Testing Your New Prompt

```bash
# Test specific prompt version
python src/llm_text_extract.py --prompt-version v1.2.0

# Calculate accuracy for new version
./backfill-accuracy --prompt-version v1.2.0

# Compare against previous versions
python -c "from src.reporting import get_analysis_summary_by_prompt_version; print(get_analysis_summary_by_prompt_version('v1.2.0'))"
```

## Key Lessons Learned

### 1. Field Integrity Rules
Prevent LLM from substituting values between fields when source data is missing. If a field isn't clearly stated, use null rather than copying from another field.

### 2. Error Correction vs Data Integrity
- **Good**: LLM correcting obvious PDF parsing errors (`27,16` → `27,160`)
- **Bad**: LLM making assumptions about missing data

### 3. Table Structure Awareness
Handle inconsistent column order and merged cells across different PDF formats.

## Performance Analysis

Current problematic fields (v1.1.2):
- **TotalCases**: 5 errors (95.2% accuracy)
- **CasesConfirmed**: 5 errors (95.2% accuracy) 
- **Deaths**: 4 errors (96.2% accuracy)

Common patterns: Field substitution (fixed in v1.1.2), number formatting improvements, better null handling.

## Advanced Features

### Batch Testing
```bash
# Test new prompt against all baseline records
python src/llm_text_extract.py --prompt-version v1.2.0 --batch-mode
./backfill-accuracy --prompt-version v1.2.0 --detailed
```

### Version Comparison
```bash
python -c "
from src.reporting import get_discrepancies_by_prompt_version
old = get_discrepancies_by_prompt_version('v1.1.2')
new = get_discrepancies_by_prompt_version('v1.2.0')
print(f'Improvement: {new.accuracy - old.accuracy:.2f} percentage points')
"
```

###  Reporting
Open `exploration/discrepancy_insights.qmd` for executive-ready comparison tables.

## Troubleshooting

**Prompt Import Failures**: Check YAML frontmatter syntax and file exists

**Accuracy Regression**: Compare field-level metrics and review specific discrepancy records

**Field Substitution Errors**: Add explicit examples in your prompt of correct vs incorrect behavior

### Debugging Commands
```bash
# Verbose extraction with debug info
python src/llm_text_extract.py --prompt-version v1.2.0 --debug

# Detailed accuracy breakdown
python src/accuracy_metrics.py --prompt-version v1.2.0 --field-details
```

---

For questions about prompt engineering or to propose improvements, see the main project documentation or contact the development team.

## Appendix: Version History & Accuracy Notes

### Current Versions

| Version | Accuracy | Key Improvements | Status |
|---------|----------|------------------|--------|
| v1.1.0  | 87.5%    | Initial structured extraction | Baseline |
| v1.1.1  | 90.0%    | Improved field detection | Improved |
| v1.1.2  | 91.09%   | Field integrity rules | Current |

### Important Notes on Accuracy Metrics

**Take these accuracy percentages with a grain of salt.** The "accuracy" is calculated by comparing LLM extractions against baseline PDF scraper data, but:

1. **LLM often corrects errors**: In many cases, the LLM is actually correct and the baseline PDF scraper data is wrong. For example:
   - LLM correctly reads "27,160" while baseline has "27,16" (PDF parsing error)
   - LLM adds missing commas: "357900" → "357,900"

2. **Limited test data**: Currently we're only comparing against 1 PDF worth of data, which is not statistically meaningful for assessing true accuracy.

3. **Different error types**: Some "errors" are actually improvements where the LLM demonstrates better understanding of the source material than the automated baseline scraper.

These metrics are useful for tracking relative improvements between prompt versions, but should not be interpreted as absolute measures of extraction quality. A more comprehensive evaluation would require manual validation against multiple PDFs and ground truth data.
