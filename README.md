# Cholera PDF Scraper

A Python project for scraping and processing cholera-related PDF documents.

## Prerequisites

- [pyenv](https://github.com/pyenv/pyenv) for Python version management
- [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv) for virtual environment management

## Setup Instructions

### 1. Install Python 3.11.4 with pyenv

```bash
# Install Python 3.11.4 if not already installed
pyenv install 3.11.4

# Set Python 3.11.4 as the local version for this project
pyenv local 3.11.4
```

### 2. Create and Activate Virtual Environment

```bash
# Create a virtual environment for this project
pyenv virtualenv 3.11.4 cholera-pdf-scraper

# Activate the virtual environment
pyenv activate cholera-pdf-scraper

# Or set it as the local environment for auto-activation
pyenv local cholera-pdf-scraper
```

### 3. Install Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt

# Install development dependencies (optional, for development)
pip install -r requirements-dev.txt

# Or install the package in development mode
pip install -e .
```

### 4. Verify Installation

```bash
# Run the main script
python src/main.py

# Run tests
pytest

# Check code formatting
black --check .

# Run linter
flake8 .

# Run type checker
mypy .
```

## Project Structure

```
ds-cholera-pdf-scraper/
├── .github/
│   └── copilot-instructions.md    # Copilot custom instructions
├── src/
│   ├── __init__.py                # Package initialization
│   └── main.py                    # Main application entry point
├── tests/
│   ├── __init__.py
│   └── test_main.py               # Unit tests
├── .gitignore                     # Git ignore rules
├── .python-version                # pyenv Python version
├── requirements.txt               # Production dependencies
├── requirements-dev.txt           # Development dependencies
├── setup.py                       # Package configuration
├── pyproject.toml                 # Tool configuration
└── README.md                      # This file
```

## Development Workflow

### Code Quality Tools

- **Black**: Code formatter (88 character line length)
- **flake8**: Linter for style and error checking
- **mypy**: Static type checker
- **pytest**: Testing framework

### Running Quality Checks

```bash
# Format code
black .

# Check linting
flake8 .

# Type checking
mypy .

# Run tests with coverage
pytest --cov=src
```

### VS Code Integration

This project includes VS Code tasks for common operations:
- Build/Install: Installs dependencies and the package
- Test: Runs the test suite
- Format: Formats code with Black
- Lint: Runs flake8 linting

## Usage

After setup, you can run the main application:

```bash
python src/main.py
```

Or if installed as a package:

```bash
cholera-scraper
```

## Contributing

1. Ensure you have the development dependencies installed
2. Run all quality checks before committing
3. Add tests for new functionality
4. Update documentation as needed

## License

This project is licensed under the MIT License.
