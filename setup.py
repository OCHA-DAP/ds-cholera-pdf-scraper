"""
DEPRECATED: This setup.py is kept for backward compatibility.
All project configuration has moved to pyproject.toml.

For new installations, use:
    uv sync

For development:
    uv sync --dev

For legacy pip installations:
    pip install -e .
"""

import warnings
from setuptools import setup, find_packages

warnings.warn(
    "setup.py is deprecated. Use 'uv sync' or 'pip install -e .' with pyproject.toml",
    DeprecationWarning,
    stacklevel=2
)

# Minimal setup - all configuration is in pyproject.toml
setup(
    name="cholera-pdf-scraper",
    version="0.2.0",
    packages=find_packages(),
    python_requires=">=3.12",
)
