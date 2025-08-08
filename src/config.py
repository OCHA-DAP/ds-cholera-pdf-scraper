"""
Configuration settings for the cholera PDF scraper project.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # dotenv not available, continue with os.getenv defaults
    pass


class Config:
    """Configuration class for the project."""

    # API Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
    OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))

    # Storage Configuration
    STAGE = os.getenv("STAGE", "dev")  # dev, staging, prod
    BLOB_CONTAINER = os.getenv("BLOB_CONTAINER", "projects")
    BLOB_PROJ_DIR = os.getenv("BLOB_PROJ_DIR", "ds-cholera-pdf-scraper")

    # Local download paths
    LOCAL_DIR_BASE = os.getenv("LOCAL_DIR_BASE", str(Path.home()))

    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    DOWNLOADS_DIR = PROJECT_ROOT / "downloads"
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    LOGS_DIR = PROJECT_ROOT / "logs"

    # Historical PDFs download location
    HISTORICAL_PDFS_DIR = (
        Path(LOCAL_DIR_BASE) / "Cholera - General" / "WHO_bulletins_historical"
    )  # PDF Source Configuration
    BASE_PDF_URL = os.getenv(
        "BASE_PDF_URL", "https://example.com/cholera-reports"  # Replace with actual URL
    )

    # Processing Configuration
    NUMERICAL_TOLERANCE = float(os.getenv("NUMERICAL_TOLERANCE", "0.05"))  # 5%
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))

    # Schema Configuration
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
        "extraction_timestamp": "datetime64[ns]",
    }

    # Key columns for alignment and deduplication
    KEY_COLUMNS = ["reporting_date", "country", "admin1", "source_file"]

    @classmethod
    def create_directories(cls):
        """Create necessary directories."""
        for directory in [
            cls.DATA_DIR,
            cls.DOWNLOADS_DIR,
            cls.OUTPUTS_DIR,
            cls.LOGS_DIR,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """
        Validate configuration settings.

        Returns:
            Dictionary with validation results
        """
        issues = []

        if not cls.OPENAI_API_KEY:
            issues.append("OPENAI_API_KEY not set")

        if cls.STAGE not in ["dev", "staging", "prod"]:
            issues.append(f"Invalid STAGE: {cls.STAGE}")

        if cls.NUMERICAL_TOLERANCE < 0 or cls.NUMERICAL_TOLERANCE > 1:
            issues.append(f"Invalid NUMERICAL_TOLERANCE: {cls.NUMERICAL_TOLERANCE}")

        return {"valid": len(issues) == 0, "issues": issues}

    @classmethod
    def get_blob_paths(cls) -> Dict[str, str]:
        """
        Get blob storage paths for different data types.

        Returns:
            Dictionary with blob paths
        """
        return {
            "historical_pdfs": f"historical_pdfs/",
            "weekly_pdfs": f"weekly_pdfs/",
            "historical_data": f"cholera_historical_data.csv",
            "baseline_data": f"cholera_baseline_data.csv",
            "comparison_reports": f"comparison_reports/",
        }


# Create directories on import
Config.create_directories()
