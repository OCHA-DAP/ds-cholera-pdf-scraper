# CLI Tools Reference

Complete reference for all command-line tools in the Cholera PDF Data Extraction Pipeline.

## Accuracy Metrics Tools

### `./backfill-accuracy` (Simple Wrapper)

Easy-to-use wrapper for accuracy metrics calculation.

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

### `scripts/backfill_accuracy_metrics.py` (Full Tool)

Advanced tool with complete options for accuracy metrics calculation.

```bash
python scripts/backfill_accuracy_metrics.py [OPTIONS]
```

**Options:**
- `--version VERSION` - Process specific prompt version (e.g., v1.1.0)
- `--outputs DIR` - Directory containing extraction files (default: outputs)
- `--data DIR` - Directory containing baseline data (default: data)  
- `--dry-run` - Show what would be processed without making changes
- `--force` - Overwrite existing accuracy metrics

**Examples:**
```bash
# Process all available versions
python scripts/backfill_accuracy_metrics.py

# Process specific version
python scripts/backfill_accuracy_metrics.py --version v1.1.0

# Dry run to see what would be processed
python scripts/backfill_accuracy_metrics.py --dry-run

# Force overwrite existing metrics
python scripts/backfill_accuracy_metrics.py --force

# Custom directories
python scripts/backfill_accuracy_metrics.py --outputs results --data baseline
```

## Prompt Management Tools

### `prompt_cli.py`

Command-line interface for prompt management and versioning.

```bash
python prompt_cli.py [COMMAND] [OPTIONS]
```

**Commands:**
- `list` - List all available prompts and versions
- `create` - Create new prompt from template
- `export` - Export prompt to markdown format
- `import` - Import prompt from markdown file
- `set-current` - Set current active prompt version

**Examples:**
```bash
# List all prompts
python prompt_cli.py list

# Export current prompt to markdown
python prompt_cli.py export health_data_extraction --version v1.1.0

# Import prompt from markdown
python prompt_cli.py import prompts/markdown/my_prompt.md

# Set current prompt version
python prompt_cli.py set-current health_data_extraction v1.1.0
```

## Data Extraction Tools

### `llm_text_extract.py`

Main LLM extraction pipeline for processing PDFs.

```bash
python src/llm_text_extract.py [OPTIONS]
```

**Options:**
- `--input FILE` - Input PDF file to process
- `--output DIR` - Output directory for results
- `--prompt-version VERSION` - Specific prompt version to use
- `--model MODEL` - OpenAI model to use (default: gpt-4o)

**Examples:**
```bash
# Process single PDF with current prompt
python src/llm_text_extract.py --input data/cholera_report.pdf

# Process with specific prompt version
python src/llm_text_extract.py --input data/report.pdf --prompt-version v1.0.0

# Custom output directory
python src/llm_text_extract.py --input data/report.pdf --output results/
```

### `download_historical_pdfs.py`

Download and upload historical PDFs to blob storage.

```bash
python scripts/download_historical_pdfs.py [OPTIONS]
```

**Options:**
- `--download-only` - Only download, don't upload to blob
- `--upload-only` - Only upload existing files to blob  
- `--resume` - Resume interrupted download/upload
- `--dry-run` - Show what would be processed

## Utility Tools

### File Discovery

```bash
# Find all prompt-versioned extraction files
ls outputs/text_extracted_data_prompt_v*.csv

# Count extraction records
wc -l outputs/text_extracted_data_prompt_v1.1.0.csv

# Check prompt logs database
python -c "
import sys; sys.path.append('src')
from prompt_logger import PromptLogger
logger = PromptLogger()
# Query logs...
"
```

### Database Queries

```bash
# Check accuracy metrics for a version
python -c "
import sys; sys.path.append('src')
from prompt_logger import PromptLogger
import json

logger = PromptLogger()
log = logger.get_latest_log_for_prompt_version('v1.1.0')
if log and log.get('custom_metrics'):
    metrics = json.loads(log['custom_metrics'])
    if 'accuracy_metrics' in metrics:
        acc = metrics['accuracy_metrics']
        print(f'Accuracy: {acc[\"overall_accuracy_percent\"]}%')
        print(f'Coverage: {acc[\"coverage_rate_percent\"]}%')
"
```

## Exit Codes

All tools use standard exit codes:
- `0` - Success
- `1` - General error
- `2` - Command-line argument error
- `3` - File not found error
- `4` - Database error

## Environment Variables

Tools respect these environment variables:

- `OPENAI_API_KEY` - OpenAI API key for LLM calls
- `AZURE_STORAGE_CONNECTION_STRING` - Azure blob storage connection
- `CHOLERA_OUTPUT_DIR` - Default output directory (overrides --output)
- `CHOLERA_DATA_DIR` - Default data directory (overrides --data)
