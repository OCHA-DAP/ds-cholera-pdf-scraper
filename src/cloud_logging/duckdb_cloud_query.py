"""
DuckDB Cloud Query Helper for Azure Blob Storage.

Enables querying Parquet files directly from Azure Blob Storage without downloading.
Uses HTTP range requests to read only the data needed.
"""

import os
from typing import Optional, List, Dict, Any
import duckdb
import pandas as pd

from src.config import Config


class DuckDBCloudQuery:
    """
    Helper class for querying Parquet files on Azure Blob Storage with DuckDB.

    DuckDB can query Parquet files directly from Azure Blob using HTTP range requests,
    reading only the columns and row groups needed - no full download required!
    """

    def __init__(self, stage: str = None, container: str = None, proj_dir: str = None):
        """
        Initialize DuckDB cloud query helper.

        Args:
            stage: Azure stage (dev/prod), defaults to Config.STAGE
            container: Azure container name, defaults to Config.BLOB_CONTAINER
            proj_dir: Project directory in blob, defaults to Config.BLOB_PROJ_DIR
        """
        self.stage = stage or Config.STAGE
        self.container = container or Config.BLOB_CONTAINER
        self.proj_dir = proj_dir or Config.BLOB_PROJ_DIR

        # Azure storage account name (inferred from stage)
        self.account_name = f"imb0chd0{self.stage}"

        # Get SAS token from environment
        sas_token_key = f"DSCI_AZ_BLOB_{self.stage.upper()}_SAS_WRITE"
        self.sas_token = os.getenv(sas_token_key)

        if not self.sas_token:
            # Try read-only token as fallback
            sas_token_key = f"DSCI_AZ_BLOB_{self.stage.upper()}_SAS"
            self.sas_token = os.getenv(sas_token_key)

        # Initialize DuckDB connection
        self.con = duckdb.connect()

        # Install and load Azure extension
        self.con.execute("INSTALL azure;")
        self.con.execute("LOAD azure;")

        # Configure Azure authentication
        self._configure_azure_auth()

    def _configure_azure_auth(self):
        """Configure DuckDB Azure authentication using SAS token."""
        if not self.sas_token:
            print("⚠️  No Azure SAS token found. Cloud querying may not work.")
            print(f"   Set DSCI_AZ_BLOB_{self.stage.upper()}_SAS_WRITE environment variable.")
            return

        # Create Azure secret in DuckDB
        # DuckDB supports SAS tokens via CREATE SECRET
        try:
            # First, drop any existing azure secret
            self.con.execute("DROP SECRET IF EXISTS azure_secret;")

            # Create new secret with SAS token
            # Note: SAS token should start with '?' or 'sv=' depending on format
            sas_token = self.sas_token
            if not sas_token.startswith('?') and not sas_token.startswith('sv='):
                sas_token = '?' + sas_token

            self.con.execute(f"""
                CREATE SECRET azure_secret (
                    TYPE AZURE,
                    ACCOUNT_NAME '{self.account_name}',
                    CONNECTION_STRING 'BlobEndpoint=https://{self.account_name}.blob.core.windows.net;SharedAccessSignature={sas_token.lstrip("?")}'
                );
            """)

            print(f"✅ DuckDB Azure authentication configured for {self.account_name}")

        except Exception as e:
            print(f"⚠️  Failed to configure Azure authentication: {e}")
            print("   Cloud querying may not work without proper authentication.")

    def get_blob_url(self, blob_path: str) -> str:
        """
        Get full Azure Blob URL for a path.

        Args:
            blob_path: Relative path in blob (e.g., "logs/prompt_logs/historical.parquet")

        Returns:
            Full Azure Blob URL
        """
        # Remove leading slash if present
        blob_path = blob_path.lstrip('/')

        return f"az://{self.account_name}.blob.core.windows.net/{self.container}/{self.proj_dir}/{blob_path}"

    def query_prompt_logs(
        self,
        columns: Optional[List[str]] = None,
        where: Optional[str] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Query prompt_logs from blob storage.

        Args:
            columns: List of columns to select (None = all columns)
            where: WHERE clause (e.g., "timestamp > '2025-01-01'")
            order_by: ORDER BY clause (e.g., "timestamp DESC")
            limit: LIMIT clause

        Returns:
            pandas DataFrame with results

        Example:
            >>> query = DuckDBCloudQuery()
            >>> df = query.query_prompt_logs(
            ...     columns=['id', 'timestamp', 'model_name', 'records_extracted'],
            ...     where="timestamp > '2025-10-01'",
            ...     order_by="timestamp DESC",
            ...     limit=10
            ... )
        """
        # Build SELECT clause
        if columns:
            select_clause = ", ".join(columns)
        else:
            select_clause = "*"

        # Build blob URL pattern (matches all parquet files in directory)
        blob_url = self.get_blob_url("processed/logs/prompt_logs/*.parquet")

        # Build query
        query = f"SELECT {select_clause} FROM read_parquet('{blob_url}')"

        if where:
            query += f" WHERE {where}"

        if order_by:
            query += f" ORDER BY {order_by}"

        if limit:
            query += f" LIMIT {limit}"

        # Execute and return as DataFrame
        try:
            return self.con.execute(query).df()
        except Exception as e:
            print(f"❌ Query failed: {e}")
            print(f"   Query: {query}")
            print(f"   Blob URL: {blob_url}")
            raise

    def query_preprocessing_logs(
        self,
        columns: Optional[List[str]] = None,
        where: Optional[str] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Query tabular_preprocessing_logs from blob storage.

        Args:
            columns: List of columns to select (None = all columns)
            where: WHERE clause
            order_by: ORDER BY clause
            limit: LIMIT clause

        Returns:
            pandas DataFrame with results
        """
        # Build SELECT clause
        if columns:
            select_clause = ", ".join(columns)
        else:
            select_clause = "*"

        # Build blob URL pattern
        blob_url = self.get_blob_url("processed/logs/tabular_preprocessing_logs/*.parquet")

        # Build query
        query = f"SELECT {select_clause} FROM read_parquet('{blob_url}')"

        if where:
            query += f" WHERE {where}"

        if order_by:
            query += f" ORDER BY {order_by}"

        if limit:
            query += f" LIMIT {limit}"

        # Execute and return as DataFrame
        try:
            return self.con.execute(query).df()
        except Exception as e:
            print(f"❌ Query failed: {e}")
            print(f"   Query: {query}")
            print(f"   Blob URL: {blob_url}")
            raise

    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute arbitrary SQL query against blob storage.

        Args:
            query: SQL query (use read_parquet with blob URLs)

        Returns:
            pandas DataFrame with results

        Example:
            >>> query = DuckDBCloudQuery()
            >>> df = query.execute_query(\"\"\"
            ...     SELECT
            ...         model_name,
            ...         COUNT(*) as runs,
            ...         AVG(records_extracted) as avg_records
            ...     FROM read_parquet('az://...')
            ...     GROUP BY model_name
            ... \"\"\")
        """
        try:
            return self.con.execute(query).df()
        except Exception as e:
            print(f"❌ Query failed: {e}")
            print(f"   Query: {query}")
            raise

    def get_latest_runs(self, n: int = 10) -> pd.DataFrame:
        """
        Get the latest N extraction runs.

        Args:
            n: Number of runs to retrieve

        Returns:
            DataFrame with latest runs
        """
        return self.query_prompt_logs(
            columns=['id', 'timestamp', 'prompt_version', 'model_name', 'records_extracted', 'execution_time_seconds'],
            order_by='timestamp DESC',
            limit=n
        )

    def get_model_statistics(self) -> pd.DataFrame:
        """
        Get aggregated statistics by model.

        Returns:
            DataFrame with model statistics
        """
        blob_url = self.get_blob_url("processed/logs/prompt_logs/*.parquet")

        query = f"""
            SELECT
                model_name,
                COUNT(*) as total_runs,
                SUM(records_extracted) as total_records,
                AVG(records_extracted) as avg_records_per_run,
                AVG(execution_time_seconds) as avg_execution_time,
                MIN(timestamp) as first_run,
                MAX(timestamp) as last_run
            FROM read_parquet('{blob_url}')
            GROUP BY model_name
            ORDER BY total_runs DESC
        """

        return self.execute_query(query)

    def close(self):
        """Close DuckDB connection."""
        self.con.close()


# Example usage
if __name__ == "__main__":
    import sys

    # Check if SAS token is available
    stage = os.getenv('STAGE', 'dev')
    sas_token = os.getenv(f'DSCI_AZ_BLOB_{stage.upper()}_SAS_WRITE')

    if not sas_token:
        print("❌ No SAS token found in environment")
        print(f"   Set DSCI_AZ_BLOB_{stage.upper()}_SAS_WRITE in your .env file")
        sys.exit(1)

    print("Testing DuckDB Cloud Query...")
    print()

    try:
        # Initialize cloud query
        query = DuckDBCloudQuery()

        # Test 1: Get latest runs
        print("📊 Latest 5 extraction runs:")
        df = query.get_latest_runs(n=5)
        print(df.to_string(index=False))
        print()

        # Test 2: Get model statistics
        print("📊 Model statistics:")
        stats = query.get_model_statistics()
        print(stats.to_string(index=False))
        print()

        print("✅ DuckDB cloud querying works!")

        query.close()

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
