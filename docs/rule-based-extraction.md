# Rule-Based PDF Table Extraction

## Overview

This project now supports **two extraction methods** for WHO cholera surveillance PDFs:

1. **LLM-Based Extraction** (existing): Uses OpenAI vision models to extract table data
2. **Rule-Based Extraction** (new): Uses Tabula for deterministic table detection

## Rule-Based Extraction Architecture

### Components

#### 1. `src/rule_based_extract.py`
Core extraction module using Tabula-Java for table detection.

**Key Functions:**
- `extract_table_from_pdf()`: Main API for extracting outbreak data
- `find_table_pages()`: Locates table section using text search
- `extract_raw_table()`: Uses Tabula with lattice mode
- `clean_data()`: Standardizes numeric fields and adds metadata
- `calculate_kpis()`: Computes week-over-week case changes

**Example Usage:**
```python
from src.rule_based_extract import extract_table_from_pdf

df = extract_table_from_pdf(
    pdf_path="OEW42-2025.pdf",
    week=42,
    year=2025
)
print(f"Extracted {len(df)} outbreak records")
```

#### 2. `scripts/run_rule_based_extraction_gha.py`
GitHub Actions wrapper script that:
- Downloads PDFs from Azure Blob Storage
- Runs rule-based extraction
- Uploads results and logs to blob

#### 3. `.github/workflows/rule-based-extract.yml`
GitHub Actions workflow that:
- Installs Java (required for Tabula)
- Runs extraction on latest or specific week
- Creates detailed job summaries

## Comparison: Rule-Based vs LLM-Based

| Aspect | Rule-Based (Tabula) | LLM-Based (GPT-5) |
|--------|---------------------|-------------------|
| **Speed** | ⚡ Fast (~5-10 seconds) | 🐢 Slower (~30-60 seconds) |
| **Cost** | 💰 Free | 💸 Per-API-call cost |
| **Accuracy** | 📊 Good for well-structured tables | 🎯 Better for complex/varied formats |
| **Robustness** | ⚠️ Brittle (fails if PDF format changes) | ✅ Flexible (adapts to format variations) |
| **Dependencies** | ☕ Requires Java + Tabula | 🔑 Requires OpenAI API key |
| **Maintenance** | 🔧 Requires updates if WHO changes PDF format | 🤖 Self-adapting with prompt updates |

## When to Use Each Method

### Use Rule-Based Extraction When:
- ✅ PDF format is consistent and stable
- ✅ Tables have clear borders (lattice structure)
- ✅ Cost optimization is important
- ✅ Speed is critical
- ✅ Deterministic results are required

### Use LLM-Based Extraction When:
- ✅ PDF format varies or changes frequently
- ✅ Tables have complex layouts or merged cells
- ✅ Manual corrections are embedded in PDFs
- ✅ Flexibility is more important than speed
- ✅ You need to extract unstructured notes/context

## Running Extractions

### GitHub Actions (Automated)

#### Latest Week:
```yaml
# Trigger manually with defaults
workflow_dispatch: rule-based-extract.yml
```

#### Specific Week:
```yaml
# Trigger manually with inputs
workflow_dispatch: rule-based-extract.yml
  week_number: "42"
  year: "2025"
```

### Local Development

#### Install Dependencies:
```bash
# Install Java (required for Tabula)
# macOS
brew install openjdk

# Ubuntu/Debian
sudo apt-get install default-jre

# Install Python dependencies
uv sync
```

#### Run Extraction:
```python
from src.rule_based_extract import extract_table_from_pdf

# Extract from local PDF
df = extract_table_from_pdf(
    pdf_path="path/to/OEW42-2025.pdf",
    week=42,
    year=2025
)

# Save results
df.to_csv("extracted_data.csv", index=False)
print(f"Extracted {len(df)} records")
```

## Blob Storage Structure

### Inputs (Raw PDFs):
```
ds-cholera-pdf-scraper/
  raw/
    monitoring/
      OEW42-2025.pdf
      OEW43-2025.pdf
      ...
```

### Outputs (Extracted Data):
```
ds-cholera-pdf-scraper/
  processed/
    rule_based_extractions/
      OEW42-2025_rule-based_1729468934.csv
      OEW43-2025_rule-based_1729555334.csv
      ...
    llm_extractions/
      OEW42-2025_gpt-5_1729468934.csv
      OEW43-2025_gpt-5_1729555334.csv
      ...
    logs/
      rule_based_extraction_log.jsonl
      prompt_logs/
        run_1729468934.parquet
        ...
```

## Execution Logs

Rule-based extractions are logged in JSONL format:

```json
{
  "week": 42,
  "year": 2025,
  "pdf_name": "OEW42-2025.pdf",
  "run_date": "2025-10-20T14:32:14",
  "status": "success",
  "extraction_method": "rule-based-tabula",
  "records_extracted": 45,
  "execution_time_seconds": 8.23,
  "runner": "github-actions",
  "blob_uploaded": true,
  "csv_blob_path": "ds-cholera-pdf-scraper/processed/rule_based_extractions/OEW42-2025_rule-based_1729468934.csv"
}
```

## Comparing Extraction Methods

You can compare results from both methods using blob storage queries:

```python
import pandas as pd
from azure.storage.blob import BlobServiceClient

# Download both extractions
df_rule = pd.read_csv("processed/rule_based_extractions/OEW42-2025_rule-based_*.csv")
df_llm = pd.read_csv("processed/llm_extractions/OEW42-2025_gpt-5_*.csv")

# Compare record counts
print(f"Rule-based: {len(df_rule)} records")
print(f"LLM-based: {len(df_llm)} records")

# Compare specific fields
comparison = pd.merge(
    df_rule[['Country', 'Event', 'Total cases']],
    df_llm[['Country', 'Event', 'Total cases']],
    on=['Country', 'Event'],
    suffixes=('_rule', '_llm')
)

# Find discrepancies
discrepancies = comparison[comparison['Total cases_rule'] != comparison['Total cases_llm']]
print(f"Discrepancies: {len(discrepancies)} records")
```

## Troubleshooting

### Java Not Found
```bash
# Install Java
sudo apt-get install default-jre

# Verify installation
java -version
```

### Tabula Extraction Fails
- **Check PDF structure**: Tabula requires tables with clear borders
- **Inspect page range**: Verify "All events currently being monitored" text exists
- **Try LLM extraction**: May be more robust for complex formats

### No Data Extracted
- **Check logs**: Look for PyPDF2 or Tabula errors
- **Verify PDF validity**: Ensure PDF is not corrupted
- **Check page numbers**: Table detection may have failed

## Future Improvements

1. **Hybrid Approach**: Use rule-based as primary, LLM as fallback
2. **Quality Scoring**: Automatically assess extraction quality
3. **Diff Analysis**: Automated comparison between methods
4. **OCR Fallback**: Handle scanned/image-based PDFs
5. **Historical Validation**: Cross-check against previous weeks

## Credits

Rule-based extraction logic adapted from Kenny's original implementation with enhancements for cloud-native deployment and blob storage integration.
