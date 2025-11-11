"""
Cloud logging module for DuckDB-based parquet logging on Azure Blob Storage.

This module provides cloud-native logging infrastructure that enables:
- Writing logs as Parquet files for efficient columnar storage
- Querying logs directly from Azure Blob without downloading (HTTP range requests)
- Seamless integration with GitHub Actions workflows

Main components:
- DuckDBLogger: Writes logs to local Parquet files for upload
- DuckDBCloudQuery: Queries Parquet files directly from Azure Blob
"""

from src.cloud_logging.duckdb_logger import DuckDBLogger
from src.cloud_logging.duckdb_cloud_query import DuckDBCloudQuery

__all__ = [
    'DuckDBLogger',
    'DuckDBCloudQuery',
]
