<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Cholera PDF Scraper Project Instructions

This is a Python project for scraping and processing cholera-related PDF documents.

## Project Setup
- Python version: 3.11.4 (managed with pyenv)
- Virtual environment: Managed with pyenv-virtualenv
- Package management: pip with requirements.txt files
- Code formatting: Black (88 character line length)
- Linting: flake8
- Type checking: mypy
- Testing: pytest

## Development Guidelines
- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Write comprehensive docstrings for all modules, classes, and functions
- Include unit tests for all new functionality
- Use meaningful variable and function names
- Keep functions small and focused on a single responsibility

## Project Structure
- `src/`: Main source code directory
- `tests/`: Unit tests
- `requirements.txt`: Production dependencies
- `requirements-dev.txt`: Development dependencies
- `setup.py`: Package configuration
- `pyproject.toml`: Tool configuration (Black, mypy, etc.)

## When writing code:
- Import modules in this order: standard library, third-party, local imports
- Use f-strings for string formatting
- Handle exceptions appropriately with specific exception types
- Use pathlib for file path operations
- Follow the existing code style and patterns
