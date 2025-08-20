# Development Setup

This document explains how to set up the project for development without using `sys.path` modifications.

## Installation for Development

1. **Install the package in development mode:**
   ```bash
   pip install -e .
   ```
   
   This installs the package in "editable" mode, allowing you to make changes to the source code and have them immediately available without reinstalling.

2. **Install development dependencies:**
   ```bash
   pip install -r requirements-dev.txt
   ```

## Running Tests

With the proper setup, tests can be run without any `sys.path` modifications:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_llm_extract.py

# Run with verbose output
pytest -v
```

The `pyproject.toml` file contains pytest configuration that automatically sets up the Python path correctly.

## Running Scripts

Scripts can now be run directly:

```bash
# Run CLI tools
python prompt_cli.py logs --limit 5

# Run weekly ingestion
python scripts/weekly_ingest.py

# Run extraction
python -m src.llm_text_extract --pdf-path example.pdf
```

## Package Structure

The project uses a proper Python package structure:

```
ds-cholera-pdf-scraper/
├── src/                    # Main package
│   ├── __init__.py
│   ├── config.py
│   ├── llm_extract.py
│   └── ...
├── tests/                  # Test package  
│   ├── __init__.py
│   ├── test_llm_extract.py
│   └── ...
├── scripts/                # Utility scripts
├── setup.py               # Package configuration
└── pyproject.toml         # Tool configuration
```

## Best Practices

### For Test Files
- Use `from src.module import Class` instead of `sys.path.append()`
- pytest automatically handles the PYTHONPATH configuration

### For Scripts
- Use `from src.module import Class` for imports
- Install the package in development mode (`pip install -e .`)

### For CLI Tools
- Import from the `src` package using absolute imports
- No need for sys.path modifications

## Migration Notes

We've removed `sys.path.append()` calls from:
- `tests/test_*.py` files
- `prompt_cli.py`
- `scripts/weekly_ingest.py`
- Various other utility scripts

All imports now use proper absolute imports from the `src` package.
