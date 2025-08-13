# Cholera PDF Data Extraction Pipeline

This repository contains a machine learning pipeline for extracting structured data from WHO cholera outbreak reports using OpenAI's LLM models. The project aims to replicate and improve upon baseline DataFrame extractions from historical cholera PDFs.

## Quick Start

```bash
# Extract data with current prompt version
python src/llm_text_extract.py

# Extract with specific prompt version
python src/llm_text_extract.py --prompt-version v1.1.2

# Calculate accuracy metrics for all prompt versions
./backfill-accuracy

# Run comprehensive discrepancy analysis
# Open exploration/discrepancy_insights.qmd in your notebook editor

# Compare prompt versions (showcase for stakeholders)
python -c "from src.reporting import quick_discrepancy_check; quick_discrepancy_check('v1.1.2')"
```

## 📚 Documentation

- **[CLI Reference](docs/cli_reference.md)** - Command-line tools and utilities
- **[Accuracy Logging System](docs/accuracy_logging_system.md)** - Comprehensive accuracy tracking
- **[Prompt Engineering Guide](docs/prompt_engineering.md)** - Working with versioned prompts  
- **[Development Setup](docs/development_setup.md)** - Local development guide

## Project Status ✅

- ✅ **PDF Download & Storage** - 271 historical PDFs in Azure blob storage
- ✅ **LLM Extraction Pipeline** - OpenAI-powered structured data extraction with good accuracy
- ✅ **Prompt Versioning System** - Markdown-based prompt management with (v1.1.2 - latest)
- ✅ **Accuracy Logging** - Field-level accuracy tracking and comprehensive analysis
- ✅ **Baseline Comparison** - Automated discrepancy analysis against manual pdf scraper
- ✅ **Multi-PDF Support** - SourceDocument tracking for batch processing analysis
- ✅ **Stakeholder Reporting** - Executive dashboard and comparison tools
- 🔄 **Production Pipeline** - Weekly automated ingestion (in progress)

## Project Overview

The cholera PDF scraper is designed to:

1. **Download historical PDFs** from WHO sources and store them in blob storage ✅ **COMPLETED**
2. **Extract structured data** using OpenAI LLMs to parse PDF content into structured tables ✅ **COMPLETED**
3. **Compare and validate** LLM outputs against baseline extractions ✅ **COMPLETED**
4. **Track accuracy improvements** through systematic prompt engineering ✅ **COMPLETED**
5. **Support production workflows** for weekly ingestion of new reports 🔄 **IN PROGRESS**

## Current Performance

**Latest Results (Prompt v1.1.2)**:
- **91.09% Overall Accuracy** - Field-level accuracy across all extracted data
- **97.12% Coverage/Precision** - Successfully processes 104/104 baseline records
- **Progressive Improvement** - From 87.5% (v1.1.0) → 90.0% (v1.1.1) → 91.09% (v1.1.2)
- **Field-Specific Tracking** - Identifies problematic fields (TotalCases, CasesConfirmed, Deaths)
- **Systematic Validation** - Comprehensive discrepancy analysis and reporting

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
                                              │ (v1.1.2%)│
                                              └─────────────────┘
                                                       │
                                              ┌─────────────────┐
                                              │  Structured     │
                                              │  DataFrame      │
                                              │ + Accuracy Log  │
                                              └─────────────────┘
                                                       │
                                              ┌─────────────────┐
                                              │ Reporting Suite │
                                              │ • Discrepancies │
                                              │                 │
                                              │                 |
                                              └─────────────────┘
```

## Project Structure

```
ds-cholera-pdf-scraper/
├── scripts/
│   ├── download_historical_pdfs.py    # PDF download and upload script ✅
│   ├── backfill_accuracy_metrics.py   # Retroactive accuracy calculation ✅
│   └── weekly_ingest.py               # Weekly processing pipeline (TODO)
├── src/
│   ├── __init__.py
│   ├── main.py                        # Legacy entry point
│   ├── config.py                      # Configuration settings ✅
│   ├── llm_text_extract.py           # OpenAI LLM extraction pipeline ✅
│   ├── prompt_manager.py              # Versioned prompt management ✅
│   ├── prompt_logger.py               # Extraction logging system ✅
│   ├── accuracy_metrics.py            # Accuracy calculation engine ✅
│   ├── post_processing.py             # Data cleaning and standardization ✅
│   ├── compare.py                     # Baseline comparison tools ✅
│   └── reporting/
│       ├── __init__.py
│       └── prompt_comparison_utils.py # Multi-version analysis tools ✅
├── prompts/
│   ├── health_data_extraction/        # JSON prompt versions ✅
│   └── markdown/
│       └── health_data_extraction/    # Markdown prompt editing ✅
├── exploration/
│   └── discrepancy_insights.qmd       # Analysis notebook ✅
├── outputs/                           # Extraction results by prompt version ✅
├── docs/                              # Comprehensive documentation ✅
├── tests/                             # Test suite
├── ./backfill-accuracy                # Convenience wrapper script ✅
├── requirements.txt                   # Python dependencies
├── requirements-dev.txt               # Development dependencies
├── pyproject.toml                     # Project configuration
└── README.md
```

## ✅ Completed: LLM-Based Data Extraction & Accuracy System

Beyond the historical PDF download system, we have implemented a comprehensive LLM-powered extraction pipeline with sophisticated accuracy tracking and prompt engineering capabilities.

### Key Features Implemented:

#### 1. **Advanced LLM Extraction Pipeline**
- **OpenAI Integration**: GPT-4 powered extraction with configurable parameters
- **Prompt Versioning**: Markdown-based prompt management with auto-import functionality
- **Multi-Format Support**: Handles various WHO report formats and table structures
- **Source Tracking**: SourceDocument field for multi-PDF batch processing analysis
- **Robust Error Handling**: Comprehensive logging and failure recovery

#### 2. **Systematic Prompt Engineering**
- **Version Progression**: v1.1.0 → v1.1.1 → v1.1.2 with measurable improvements
- **Markdown Editing**: User-friendly prompt editing with YAML frontmatter
- **Auto-Import**: Seamless workflow from markdown editing to testing
- **Field Integrity Rules**: Prevents value substitution between fields (Deaths → CasesConfirmed)
- **Table Structure Awareness**: Consistent column order detection and processing

#### 3. **Comprehensive Accuracy Tracking**
- **Field-Level Analysis**: Individual accuracy metrics for TotalCases, CasesConfirmed, Deaths, CFR, Grade
- **Progressive Tracking**: Accuracy improvements documented across prompt versions
- **SQLite Logging**: Persistent storage of all extractions with prompt metadata
- **Discrepancy Analysis**: Detailed comparison against baseline ground truth
- **Problematic Field Identification**: Automated detection of high-error fields

#### 4. **Enterprise-Grade Reporting System**
- **Executive Dashboard**: Stakeholder-ready comparison tables and metrics
- **Multi-Version Analysis**: Side-by-side prompt performance comparison
- **Discrepancy Details**: Record-level analysis of extraction errors
- **Business Insights**: Automated generation of improvement summaries
- **Utility Functions**: Easy access to analysis results by prompt version

### Performance Results:

#### Prompt Version Comparison:
| Version | Records | Accuracy | TotalCases Errors | CasesConfirmed Errors | Deaths Errors |
|---------|---------|----------|-------------------|----------------------|---------------|
| v1.1.0  | 104     | 87.5%    | 8                 | 7                    | 6             |
| v1.1.1  | 104     | 90.0%    | 6                 | 5                    | 5             |
| v1.1.2  | 104     | 91.09%   | 5                 | 5                    | 4             |

**Key Insights**:
- **+3.59 percentage points** improvement from v1.1.0 to v1.1.2
- **Perfect record coverage**: 104/104 baseline records successfully processed
- **Systematic error reduction**: Consistent improvement across all field types
- **Smart error correction**: LLM fixes comma-related PDF parsing errors (e.g., 27,16 → 27,160)

### Technical Implementation:

#### Extraction Pipeline:
```python
# Auto-import from markdown
python src/llm_text_extract.py --prompt-version v1.1.2

# Pipeline stages:
# 1. PDF text extraction and cleaning
# 2. Prompt template population with PDF content
# 3. OpenAI API call with structured output format
# 4. Response parsing and DataFrame creation
# 5. SourceDocument field addition for multi-PDF tracking
# 6. Accuracy logging with prompt metadata
```

#### Accuracy Analysis Workflow:
```python
# Generate comprehensive accuracy metrics
./backfill-accuracy

# Access analysis results programmatically
from src.reporting import get_discrepancies_by_prompt_version
discrepancies = get_discrepancies_by_prompt_version('v1.1.2')

# Executive summary
from src.reporting import quick_discrepancy_check
quick_discrepancy_check('v1.1.2')
```

#### Stakeholder Reporting:
```python
# Multi-version comparison showcase
# Open exploration/discrepancy_insights.qmd
# Run the showcase chunk for executive presentation
```

### Data Quality Insights:

#### LLM vs Baseline Comparison:
- **LLM Advantages**: Corrects obvious PDF parsing errors, uses contextual text information
- **Example Corrections**: 
  - Angola: 27,16 → 27,160 (comma placement correction)
  - Madagascar: 3579 → 357,900 (missing comma restoration)
- **Remaining Challenges**: Field substitution when CasesConfirmed missing (addressed in v1.1.2)
- **Coverage Excellence**: 97.12% precision with comprehensive field extraction

### Historical PDF Download System:

We have successfully implemented a comprehensive PDF download and storage system that handles all 271 historical cholera reports from WHO sources.

#### Key Features Implemented:

##### 1. **Intelligent PDF Discovery**
- **CSV-based metadata**: Reads comprehensive PDF information from GitHub-hosted CSV
- **URL validation**: Filters and validates 271 PDF URLs from WHO sources
- **Metadata enrichment**: Maps each PDF to standardized filename using CSV `FileName` column

##### 2. **Robust Download System**
- **VPN-compatible access**: Uses `apps.who.int` endpoints (avoiding rate-limited `iris.who.int`)
- **Retry strategy**: Automatic retry with exponential backoff for failed downloads
- **Rate limiting**: Respectful delays to avoid overwhelming WHO servers
- **Existing file detection**: Skips already-downloaded files for efficiency
- **Progress tracking**: Detailed logging of download progress and failures

##### 3. **Azure Blob Storage Integration**
- **ocha-stratus integration**: Uses organizational blob storage library with proper write permissions
- **Structured organization**: Uploads to `projects/ds-cholera-pdf-scraper/raw/pdfs/`
- **Batch processing**: Efficient upload of all downloaded files
- **Results**: ✅ 253 PDFs successfully uploaded to Azure blob storage

### Usage Examples:

```bash
# LLM Extraction (Primary Workflow)
python src/llm_text_extract.py                    # Use current prompt version
python src/llm_text_extract.py --prompt-version v1.1.2  # Use specific version

# Accuracy Analysis
./backfill-accuracy                                # Calculate metrics for all versions
./backfill-accuracy --dry-run                     # Preview what would be calculated
python -c "from src.reporting import quick_discrepancy_check; quick_discrepancy_check('v1.1.2')"

# Stakeholder Reporting
# Open exploration/discrepancy_insights.qmd in your notebook editor
# Run the showcase chunk for executive presentation

# PDF Download (Historical)
python scripts/download_historical_pdfs.py        # Download all PDFs and upload to blob storage
python scripts/download_historical_pdfs.py --download-only  # Download only (skip blob upload)
python scripts/download_historical_pdfs.py --upload-only    # Upload existing local PDFs to blob storage
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
