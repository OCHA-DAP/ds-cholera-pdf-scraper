# Development Setup

This document explains how to set up the project for development using modern Python tooling.

## Installation for Development

We use **uv** for fast, reliable package management:

1. **Install uv (if not already installed):**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install project with development dependencies:**
   ```bash
   # Install all dependencies (production + dev)
   uv sync --group dev

   # This installs:
   # - Production deps from pyproject.toml dependencies
   # - Dev deps: pytest, black, ruff
   # - Creates .venv/ automatically
   ```

   The package is installed in "editable" mode automatically, allowing you to make changes to the source code and have them immediately available.

**Why uv?**
- 10-100x faster than pip
- Deterministic builds via uv.lock
- Automatic virtual environment management
- No need for separate requirements.txt files

## Running Tests

With uv, tests can be run without any `sys.path` modifications:

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_llm_extract.py

# Run with verbose output
uv run pytest -v
```

The `pyproject.toml` file contains pytest configuration that automatically sets up the Python path correctly.

## Running Scripts

Scripts can be run using `uv run`:

```bash
# Run CLI tools
uv run python prompt_cli.py logs --limit 5

# Run weekly ingestion
uv run python scripts/weekly_ingest.py

# Run extraction
uv run python -m src.llm_text_extract --pdf-path example.pdf

# Or activate the virtual environment
source .venv/bin/activate  # Unix
# .venv\Scripts\activate  # Windows
python scripts/weekly_ingest.py
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
