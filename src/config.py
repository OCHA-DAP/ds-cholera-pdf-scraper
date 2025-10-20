"""
Configuration settings for the cholera PDF scraper project.
"""

import os
from pathlib import Path
from typing import Any, Dict

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
    OPENAI_ORG_API_KEY = os.getenv(
        "DSCI_AZ_OPENAI_API_KEY_WHO_CHOLERA", ""
    )  # Organizational key
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

    # Determine which OpenAI key to use (prefer organizational key)
    EFFECTIVE_OPENAI_KEY = OPENAI_ORG_API_KEY if OPENAI_ORG_API_KEY else OPENAI_API_KEY

    # Model Configuration
    USE_OPENROUTER = os.getenv("USE_OPENROUTER", "false").lower() == "true"
    MODEL_NAME = os.getenv(
        "MODEL_NAME", "openai/gpt-4o"
    )  # Default model for OpenRouter
    OPENAI_MODEL = os.getenv(
        "OPENAI_MODEL", "gpt-4o"
    )  # Legacy fallback for direct OpenAI
    MODEL_TEMPERATURE = float(
        os.getenv("MODEL_TEMPERATURE", "0.1")
    )  # OpenRouter Configuration
    OPENROUTER_SITE_URL = os.getenv(
        "OPENROUTER_SITE_URL", "https://github.com/OCHA-DAP/ds-cholera-pdf-scraper"
    )
    OPENROUTER_SITE_NAME = os.getenv("OPENROUTER_SITE_NAME", "OCHA Cholera PDF Scraper")

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
    )

    # Processing Configuration
    NUMERICAL_TOLERANCE = float(os.getenv("NUMERICAL_TOLERANCE", "0.05"))  # 5%

    # Logging Configuration
    LOG_BACKEND = os.getenv("LOG_BACKEND", "sqlite")  # sqlite, duckdb, or jsonl
    BLOB_LOGS_ENABLED = os.getenv("BLOB_LOGS_ENABLED", "false").lower() == "true"

    # Surveillance Preprocessing Configuration
    ENABLE_SURVEILLANCE_PREPROCESSING = (
        os.getenv("ENABLE_SURVEILLANCE_PREPROCESSING", "false").lower() == "true"
    )
    SURVEILLANCE_MODE = os.getenv(
        "SURVEILLANCE_MODE", "hybrid"
    )  # hybrid|surveillance-only|llm-only

    # Surveillance table extraction config
    SURVEILLANCE_CONFIDENCE_THRESHOLD = float(
        os.getenv("SURVEILLANCE_CONFIDENCE_THRESHOLD", "0.95")
    )
    SURVEILLANCE_PAGES_START = int(
        os.getenv("SURVEILLANCE_PAGES_START", "9")
    )  # WHO bulletin table start page
    SURVEILLANCE_PAGES_END = int(
        os.getenv("SURVEILLANCE_PAGES_END", "15")
    )  # WHO bulletin table end page

    # Integration modes
    FALLBACK_TO_RAW_TEXT = os.getenv("FALLBACK_TO_RAW_TEXT", "true").lower() == "true"
    LOG_SURVEILLANCE_METADATA = (
        os.getenv("LOG_SURVEILLANCE_METADATA", "true").lower() == "true"
    )

    # Legacy Preprocessing Configuration (deprecated in favor of surveillance approach)
    ENABLE_PREPROCESSING = os.getenv("ENABLE_PREPROCESSING", "false").lower() == "true"
    PREPROCESSING_MODE = os.getenv(
        "PREPROCESSING_MODE", "hybrid"
    )  # hybrid|preprocess-only|llm-only

    # Table detection config
    TABLE_DETECTION_ENABLED = (
        os.getenv("TABLE_DETECTION_ENABLED", "true").lower() == "true"
    )
    TABLE_CONFIDENCE_THRESHOLD = float(os.getenv("TABLE_CONFIDENCE_THRESHOLD", "0.8"))
    TATR_MODEL_PATH = os.getenv("TATR_MODEL_PATH", "models/tatr")

    # Narrative linking config
    NARRATIVE_LINKING_ENABLED = (
        os.getenv("NARRATIVE_LINKING_ENABLED", "false").lower() == "true"
    )
    VECTOR_STORE_ENABLED = os.getenv("VECTOR_STORE_ENABLED", "false").lower() == "true"
    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

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

        # Check API keys based on usage
        if cls.USE_OPENROUTER:
            if not cls.OPENROUTER_API_KEY:
                issues.append("OPENROUTER_API_KEY not set (USE_OPENROUTER=true)")
        else:
            if not cls.OPENAI_API_KEY:
                issues.append("OPENAI_API_KEY not set")

        if cls.STAGE not in ["dev", "staging", "prod"]:
            issues.append(f"Invalid STAGE: {cls.STAGE}")

        if cls.NUMERICAL_TOLERANCE < 0 or cls.NUMERICAL_TOLERANCE > 1:
            tolerance_val = cls.NUMERICAL_TOLERANCE
            issues.append(f"Invalid NUMERICAL_TOLERANCE: {tolerance_val}")

        return {"valid": len(issues) == 0, "issues": issues}

    @classmethod
    def is_openai_model(cls, model_name: str) -> bool:
        """
        Check if a model name is an OpenAI model.

        Args:
            model_name: Model identifier (e.g., "openai/gpt-4o" or "gpt-4o")

        Returns:
            True if it's an OpenAI model
        """
        model_lower = model_name.lower()

        # Check for explicit OpenAI prefix
        if model_lower.startswith("openai/"):
            return True

        # Check for known OpenAI model patterns
        openai_patterns = [
            "gpt-",
            "o1-",
            "text-davinci",
            "text-curie",
            "text-babbage",
            "text-ada",
            "code-davinci",
            "code-cushman",
            "davinci",
            "curie",
            "babbage",
            "ada",
        ]

        return any(pattern in model_lower for pattern in openai_patterns)

    @classmethod
    def get_llm_client_config_for_model(cls, model_name: str = None) -> Dict[str, Any]:
        """
        Get LLM client configuration automatically selecting provider based on model.
        OpenAI models use organizational OpenAI API, others use OpenRouter.

        Args:
            model_name: Specific model to configure for (overrides default)

        Returns:
            Dictionary with client configuration
        """
        target_model = model_name if model_name else cls.MODEL_NAME

        if cls.is_openai_model(target_model):
            # Use organizational OpenAI API for OpenAI models
            clean_model = target_model.replace(
                "openai/", ""
            )  # Remove prefix if present
            return {
                "provider": "openai",
                "api_key": cls.EFFECTIVE_OPENAI_KEY,
                "model": clean_model,
                "temperature": cls.MODEL_TEMPERATURE,
            }
        else:
            # Use OpenRouter for non-OpenAI models
            return {
                "provider": "openrouter",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": cls.OPENROUTER_API_KEY,
                "model": target_model,
                "temperature": cls.MODEL_TEMPERATURE,
                "extra_headers": {
                    "HTTP-Referer": cls.OPENROUTER_SITE_URL,
                    "X-Title": cls.OPENROUTER_SITE_NAME,
                },
            }

    @classmethod
    def get_llm_client_config(cls) -> Dict[str, Any]:
        """
        Get LLM client configuration based on provider choice.

        Returns:
            Dictionary with client configuration
        """
        if cls.USE_OPENROUTER:
            return {
                "provider": "openrouter",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": cls.OPENROUTER_API_KEY,
                "model": cls.MODEL_NAME,
                "temperature": cls.MODEL_TEMPERATURE,
                "extra_headers": {
                    "HTTP-Referer": cls.OPENROUTER_SITE_URL,
                    "X-Title": cls.OPENROUTER_SITE_NAME,
                },
            }
        else:
            return {
                "provider": "openai",
                "api_key": cls.EFFECTIVE_OPENAI_KEY,  # Use organizational key
                "model": cls.OPENAI_MODEL,
                "temperature": cls.MODEL_TEMPERATURE,
            }

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
            # DuckDB/Parquet logs (for cloud querying)
            "parquet_logs": f"processed/logs/",
            "prompt_logs": f"processed/logs/prompt_logs/",
            "preprocessing_logs": f"processed/logs/tabular_preprocessing_logs/",
            # Extraction outputs
            "extractions": f"raw/monitoring/extractions/",
            "llm_extractions": f"processed/llm_extractions/",
        }

    @classmethod
    def get_duckdb_logs_dir(cls) -> Path:
        """
        Get local directory for DuckDB parquet logs.

        Returns:
            Path to DuckDB logs directory
        """
        logs_dir = cls.DATA_DIR / "duckdb_logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir


# Create directories on import
Config.create_directories()
