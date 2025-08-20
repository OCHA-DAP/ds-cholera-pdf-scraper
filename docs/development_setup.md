# Development Setup Guide

Complete guide for setting up the Cholera PDF Data Extraction Pipeline development environment.

## Prerequisites

### System Requirements
- macOS, Linux, or Windows with WSL2
- Python 3.11.4 (managed via pyenv recommended)
- Git
- Azure CLI (for blob storage operations)

### Account Requirements
- OpenAI API account with GPT-4 access
- Azure Storage account (for blob operations)

## Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd ds-cholera-pdf-scraper
```

### 2. Python Environment Setup

**Option A: Using pyenv (Recommended)**
```bash
# Install pyenv if not already installed
curl https://pyenv.run | bash

# Install Python 3.11.4
pyenv install 3.11.4
pyenv local 3.11.4
```

**Option B: Using system Python**
Ensure Python 3.11.4 is available via `python3` or `python`.

### 3. Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 4. Install Dependencies
```bash
# Install production dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Install ocha-stratus for blob operations
pip install ocha-stratus
```

### 5. Environment Configuration

Create `.env` file in project root:
```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Azure Storage Configuration  
AZURE_STORAGE_CONNECTION_STRING=your_azure_connection_string

# Optional: Custom directories
CHOLERA_OUTPUT_DIR=outputs
CHOLERA_DATA_DIR=data
```

**Important:** Add `.env` to `.gitignore` to avoid committing secrets.

## Project Structure Setup

### Directory Creation
The following directories should be created:
```bash
mkdir -p data downloads logs outputs test_downloads
mkdir -p prompts/markdown prompts/health_data_extraction
mkdir -p docs
```

### Initial Data Setup
```bash
# Place baseline extraction CSV in data/ directory
cp path/to/baseline_extraction.csv data/

# Download historical PDFs (optional for development)
python scripts/download_historical_pdfs.py --download-only
```

## Development Tools Configuration

### Code Quality Tools

**Black (Code Formatting)**
```bash
# Format all Python files
black . --line-length 88

# Check formatting without changes
black . --check --line-length 88
```

**flake8 (Linting)**
```bash
# Run linter
flake8 src/ tests/ scripts/

# Configuration in setup.cfg or pyproject.toml
```

**mypy (Type Checking)**
```bash
# Run type checking
mypy src/

# Configuration in mypy.ini or pyproject.toml
```

### VS Code Setup (Optional)

Create `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "88"],
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true
    }
}
```

## Testing Setup

### Run Test Suite
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_llm_extract.py

# Run with verbose output
pytest -v
```

### Test Data Setup
```bash
# Create test data directories
mkdir -p test_downloads tests/fixtures

# Add sample PDFs for testing (not in git)
cp sample_pdfs/* test_downloads/
```

## Database Initialization

### SQLite Database Setup
```bash
# Initialize prompt logging database
python -c "
import sys; sys.path.append('src')
from prompt_logger import PromptLogger
logger = PromptLogger()
print('Database initialized successfully')
"
```

## Verification

### Quick Verification Tests
```bash
# Test OpenAI API connection
python -c "
import openai
openai.api_key = 'your_key_here'
print('OpenAI API:', openai.Model.list()['data'][0]['id'])
"

# Test ocha-stratus blob operations
python -c "
import ocha_stratus as stratus
print('Stratus loaded successfully')
"

# Test prompt system
python -c "
import sys; sys.path.append('src')
from prompt_manager import PromptManager
pm = PromptManager()
print('Prompt system working')
"
```

### Run Development Pipeline
```bash
# Test full pipeline with sample data
python src/main.py --test-mode

# Run accuracy backfill
python backfill_accuracy_metrics.py --dry-run
```

## Common Development Commands

### Daily Development Workflow
```bash
# 1. Activate environment
source venv/bin/activate

# 2. Pull latest changes
git pull origin main

# 3. Install any new dependencies
pip install -r requirements.txt -r requirements-dev.txt

# 4. Run tests
pytest

# 5. Format code
black .

# 6. Check linting
flake8 src/ tests/ scripts/

# 7. Run type checking
mypy src/
```

### Data Processing Commands
```bash
# Process new PDF
python src/llm_text_extract.py --input data/new_report.pdf

# Update accuracy metrics
./backfill-accuracy current

# Compare results
python exploration/discrepancy_insights.qmd
```

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure src/ is in Python path
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
```

**Permission Errors**
```bash
# Make scripts executable
chmod +x backfill-accuracy
chmod +x scripts/*.py
```

**API Key Issues**
```bash
# Verify environment variables loaded
python -c "import os; print('API Key:', os.getenv('OPENAI_API_KEY', 'NOT SET'))"
```

**Database Lock Issues**
```bash
# Remove SQLite lock file if needed
rm -f logs/prompts.db-lock
```

### Debug Mode
Enable debug logging by setting environment variable:
```bash
export CHOLERA_DEBUG=1
python src/main.py
```

## Next Steps

After successful setup:
1. Review [CLI Reference](cli_reference.md) for available commands
2. Check [Accuracy Logging System](accuracy_logging_system.md) for metrics workflow
3. Read project README for usage examples
4. Explore `exploration/` directory for analysis notebooks
