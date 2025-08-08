# Cholera PDF Data Extraction Pipeline

This repository contains a machine learning pipeline for extracting structured data from WHO cholera outbreak reports using OpenAI's LLM models. The project aims to replicate and improve upon baseline DataFrame extractions from historical cholera PDFs.

## Project Overview

The cholera PDF scraper is designed to:

1. **Download historical PDFs** from WHO sources and store them in blob storage ✅ **COMPLETED**
2. **Extract structured data** using OpenAI LLMs to parse PDF content into structured tables
3. **Compare and validate** LLM outputs against baseline extractions
4. **Support production workflows** for weekly ingestion of new reports

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   WHO Sources   │───▶│  Local Storage  │───▶│  Azure Blob     │
│   (271 PDFs)    │    │  (Google Drive) │    │  (projects/     │
└─────────────────┘    └─────────────────┘    │   ds-cholera-   │
                                              │   pdf-scraper/) │
                                              └─────────────────┘
                                                       │
                                              ┌─────────────────┐
                                              │   OpenAI LLM    │
                                              │   Processing    │
                                              │   (TODO)        │
                                              └─────────────────┘
                                                       │
                                              ┌─────────────────┐
                                              │  Structured     │
                                              │  DataFrame      │
                                              │  (TODO)         │
                                              └─────────────────┘
```

## Project Structure

```
ds-cholera-pdf-scraper/
├── scripts/
│   └── download_historical_pdfs.py    # PDF download and upload script ✅
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py                      # Configuration settings ✅
│   ├── llm_extract.py                 # OpenAI API integration (TODO)
│   ├── parse_output.py                # LLM response parsing (TODO)
│   └── compare.py                     # Baseline comparison (TODO)
├── tests/
│   ├── __init__.py
│   └── test_main.py
├── requirements.txt                   # Python dependencies
├── requirements-dev.txt               # Development dependencies
├── pyproject.toml                     # Project configuration
└── README.md
```

## ✅ Completed: Historical PDF Download and Storage

We have successfully implemented a comprehensive PDF download and storage system that handles all 271 historical cholera reports from WHO sources.

### Key Features Implemented:

#### 1. **Intelligent PDF Discovery**
- **CSV-based metadata**: Reads comprehensive PDF information from GitHub-hosted CSV
- **URL validation**: Filters and validates 271 PDF URLs from WHO sources
- **Metadata enrichment**: Maps each PDF to standardized filename using CSV `FileName` column

#### 2. **Robust Download System**
- **VPN-compatible access**: Uses `apps.who.int` endpoints (avoiding rate-limited `iris.who.int`)
- **Retry strategy**: Automatic retry with exponential backoff for failed downloads
- **Rate limiting**: Respectful delays to avoid overwhelming WHO servers
- **Existing file detection**: Skips already-downloaded files for efficiency
- **Progress tracking**: Detailed logging of download progress and failures

#### 3. **Local File Management**
- **Staging directory**: Downloads to Google Drive shared folder for team access
- **Filename standardization**: Uses CSV metadata for consistent naming conventions
- **Cleanup functionality**: Removes invalid/incorrectly named files
- **Directory organization**: Maintains clean, organized local storage

#### 4. **Azure Blob Storage Integration**
- **ocha-stratus integration**: Uses organizational blob storage library with proper write permissions
- **Structured organization**: Uploads to `projects/ds-cholera-pdf-scraper/raw/pdfs/`
- **Batch processing**: Efficient upload of all downloaded files
- **Overwrite support**: Handles reprocessing scenarios safely

### Usage Examples:

```bash
# Download all PDFs and upload to blob storage (full pipeline)
python scripts/download_historical_pdfs.py

# Download only (skip blob upload)
python scripts/download_historical_pdfs.py --download-only

# Upload existing local PDFs to blob storage
python scripts/download_historical_pdfs.py --upload-only

# Clean up local directory (remove invalid files)
python scripts/download_historical_pdfs.py --cleanup
```

### Technical Implementation Details:

#### PDF Discovery Process:
```python
# Data source
CSV_URL = "https://github.com/CBPFGMS/pfbi-data/raw/main/who_download_log.csv"

# Key columns used:
# - LinktoDocument: PDF URL
# - FileName: Standardized filename for consistent naming
# - Additional metadata for future LLM processing
```

#### Download Configuration:
- **Retry strategy**: 3 attempts with exponential backoff
- **Rate limiting**: 0.5s delay between requests, 2s every 10 files
- **Timeout handling**: 30-second timeout with proper error handling
- **VPN compatibility**: Direct server access without problematic redirects

#### Blob Storage Structure:
```
projects/
└── ds-cholera-pdf-scraper/
    └── raw/
        └── pdfs/
            ├── OEW01-271203012021.pdf
            ├── Week_52__25_-_31_December_2023.pdf
            ├── Week_26__24_to_30_June_2024.pdf
            └── ... (253 total files)
```

### Results Achieved:

- **✅ 253 PDFs successfully downloaded** from WHO sources
- **✅ 253 PDFs successfully uploaded** to Azure blob storage  
- **✅ 100% success rate** for blob uploads using corrected `stratus` permissions
- **✅ Consistent file naming** using CSV metadata
- **✅ Clean local directory** with standardized organization
- **✅ Comprehensive error handling** and logging throughout

### Configuration

The system uses environment-based configuration:

```bash
# Required environment variables
STAGE=dev                              # dev, staging, prod
BLOB_CONTAINER=projects                # Azure blob container
BLOB_PROJ_DIR=ds-cholera-pdf-scraper  # Project folder in blob
LOCAL_DIR_BASE=/path/to/google/drive   # Google Drive shared folder

# For future LLM integration
OPENAI_API_KEY=your_openai_key_here    # OpenAI API access
OPENAI_MODEL=gpt-4o                    # Model selection
OPENAI_TEMPERATURE=0.1                 # Temperature for consistent extraction
```

## Next Steps (Roadmap)

### Phase 2: LLM-Based Data Extraction (Next Priority)
- [ ] **Implement `src/llm_extract.py`**: OpenAI API integration for PDF text extraction
- [ ] **Design extraction prompts**: Create prompts for structured data extraction from cholera reports
- [ ] **Handle PDF preprocessing**: Text extraction, cleaning, and preparation for LLM processing
- [ ] **Implement batch processing**: Process all 253 uploaded PDFs efficiently

### Phase 3: Output Processing and Validation
- [ ] **Implement `src/parse_output.py`**: Parse LLM responses into structured DataFrames
- [ ] **Schema validation**: Ensure outputs match baseline schema requirements
- [ ] **Implement `src/compare.py`**: Compare LLM extractions against baseline data
- [ ] **Quality metrics**: Develop accuracy and completeness metrics

### Phase 4: Production Pipeline
- [ ] **Create `scripts/weekly_ingest.py`**: Automated weekly processing of new reports
- [ ] **Set up monitoring**: Error detection and alerting for production pipeline
- [ ] **Database integration**: Store processed data in production database
- [ ] **API endpoints**: Expose processed data via REST API

## Target Data Schema

The baseline schema for extracted cholera data:

```python
BASELINE_SCHEMA = {
    "reporting_date": "datetime64[ns]",
    "country": "string", 
    "admin1": "string",
    "admin2": "string",
    "suspected_cases": "Int64",
    "confirmed_cases": "Int64", 
    "deaths": "Int64",
    "case_fatality_rate": "float64",
    "population_at_risk": "Int64",
    "reporting_period_start": "datetime64[ns]",
    "reporting_period_end": "datetime64[ns]",
    "source_file": "string",
    "extraction_timestamp": "datetime64[ns]"
}
```

## Installation and Setup

### Prerequisites
- Python 3.11.4 (via pyenv)
- Access to Azure blob storage via `ocha-stratus`
- OpenAI API key (for future LLM integration)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd ds-cholera-pdf-scraper

# Set up Python environment
pyenv local 3.11.4
pyenv virtualenv 3.11.4 ds-cholera-pdf-scraper
pyenv local ds-cholera-pdf-scraper

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development

# Configure environment
cp .env.example .env
# Edit .env with your configuration
```

### Core Dependencies
- **pandas**: Data manipulation and CSV processing
- **requests**: HTTP client for PDF downloads with retry logic
- **ocha-stratus**: Azure blob storage integration with organizational authentication
- **azure.storage.blob**: Direct Azure blob operations
- **openai**: OpenAI API client (for future LLM integration)

### Development Dependencies
- **pytest**: Testing framework
- **black**: Code formatting (88 character limit)
- **flake8**: Linting and style checking
- **mypy**: Type checking

## Contributing

Please follow the established coding standards:
- Use Black for code formatting (88 character line limit)
- Run flake8 for linting before commits
- Add type hints and use mypy for type checking
- Write pytest tests for new functionality
- **Never fix linting with LLM AI** - always leave to dedicated tools

## License

[Add your license information here]
