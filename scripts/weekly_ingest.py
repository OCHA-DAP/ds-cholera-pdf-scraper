#!/usr/bin/env python3
"""
Weekly ingest pipeline for cholera PDF processing.

This script orchestrates the weekly download, LLM processing,
and database update workflow.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import ocha_stratus as stratus
import pandas as pd

from src.compare import DataComparator

# Import project modules
from src.llm_extract import LLMExtractor
from src.parse_output import OutputParser

logger = logging.getLogger(__name__)


class WeeklyIngestPipeline:
    """Orchestrates weekly cholera PDF ingestion and processing."""

    def __init__(
        self, openai_api_key: str, stage: str = "dev", output_dir: Path = None
    ):
        """
        Initialize the weekly ingest pipeline.

        Args:
            openai_api_key: OpenAI API key for LLM extraction
            stage: Environment stage (dev/prod)
            output_dir: Directory for outputs
        """
        self.openai_api_key = openai_api_key
        self.stage = stage
        self.output_dir = output_dir or Path("./outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.llm_extractor = LLMExtractor(openai_api_key)
        self.output_parser = OutputParser()
        self.comparator = DataComparator()

        # Paths
        self.download_dir = Path("./downloads/weekly")
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def discover_new_pdfs(self, since_date: datetime = None) -> List[str]:
        """
        Discover new PDFs published since the last run.

        Args:
            since_date: Only get PDFs newer than this date

        Returns:
            List of new PDF URLs
        """
        if since_date is None:
            since_date = datetime.now() - timedelta(days=7)  # Last week

        logger.info(f"Discovering new PDFs since {since_date}")

        # TODO: Implement logic to discover new PDFs
        # This will depend on the specific cholera reporting website
        new_pdf_urls = []

        # Placeholder implementation
        # You would typically:
        # 1. Check the website for new publications
        # 2. Compare against previously processed files
        # 3. Return only new URLs

        logger.info(f"Found {len(new_pdf_urls)} new PDFs")
        return new_pdf_urls

    def download_pdfs(self, pdf_urls: List[str]) -> List[Path]:
        """
        Download new PDFs to local directory.

        Args:
            pdf_urls: List of PDF URLs to download

        Returns:
            List of local PDF file paths
        """
        logger.info(f"Downloading {len(pdf_urls)} PDFs")

        downloaded_files = []

        for i, pdf_url in enumerate(pdf_urls, 1):
            try:
                logger.info(f"Downloading PDF {i}/{len(pdf_urls)}: {pdf_url}")

                # TODO: Implement actual download logic
                # Similar to the historical downloader
                filename = f"weekly_{datetime.now().strftime('%Y%m%d')}_{i}.pdf"
                local_path = self.download_dir / filename

                # Placeholder - actual download would happen here
                # downloaded_files.append(local_path)

            except Exception as e:
                logger.error(f"Failed to download {pdf_url}: {e}")
                continue

        logger.info(f"Successfully downloaded {len(downloaded_files)} PDFs")
        return downloaded_files

    def upload_pdfs_to_blob(self, pdf_paths: List[Path]) -> None:
        """
        Upload new PDFs to blob storage.

        Args:
            pdf_paths: List of local PDF file paths
        """
        logger.info(f"Uploading {len(pdf_paths)} PDFs to blob storage")

        for pdf_path in pdf_paths:
            try:
                blob_name = f"weekly_pdfs/{pdf_path.name}"
                logger.info(f"Uploading {pdf_path} to blob as {blob_name}")

                # TODO: Implement blob upload using ocha_stratus
                # stratus.upload_file_to_blob(pdf_path, blob_name, stage=self.stage)

            except Exception as e:
                logger.error(f"Failed to upload {pdf_path}: {e}")
                continue

    def process_pdfs_with_llm(self, pdf_paths: List[Path]) -> pd.DataFrame:
        """
        Process PDFs using LLM extraction.

        Args:
            pdf_paths: List of PDF file paths

        Returns:
            DataFrame with extracted data
        """
        logger.info(f"Processing {len(pdf_paths)} PDFs with LLM")

        # Extract data from all PDFs
        extraction_results = self.llm_extractor.process_multiple_pdfs(pdf_paths)

        # Parse results into DataFrame
        extracted_df = self.output_parser.parse_multiple_extractions(extraction_results)

        logger.info(f"Extracted data from {len(extracted_df)} records")
        return extracted_df

    def load_historical_data(self) -> pd.DataFrame:
        """
        Load historical data from blob storage.

        Returns:
            Historical data DataFrame
        """
        logger.info("Loading historical data from blob storage")

        try:
            # TODO: Load from blob using ocha_stratus
            # historical_df = stratus.load_csv_from_blob(
            #     "cholera_historical_data.csv",
            #     stage=self.stage
            # )

            # Placeholder - return empty DataFrame for now
            historical_df = pd.DataFrame()

            logger.info(f"Loaded {len(historical_df)} historical records")
            return historical_df

        except Exception as e:
            logger.warning(f"Could not load historical data: {e}")
            return pd.DataFrame()

    def merge_with_historical_data(
        self, new_data: pd.DataFrame, historical_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge new data with historical data.

        Args:
            new_data: Newly extracted DataFrame
            historical_data: Historical data DataFrame

        Returns:
            Combined DataFrame
        """
        logger.info("Merging new data with historical data")

        if historical_data.empty:
            logger.info("No historical data found, using only new data")
            return new_data

        # Combine data, removing duplicates
        combined_df = pd.concat([historical_data, new_data], ignore_index=True)

        # Remove duplicates based on key columns
        key_columns = ["reporting_date", "country", "admin1", "source_file"]
        available_keys = [col for col in key_columns if col in combined_df.columns]

        if available_keys:
            combined_df = combined_df.drop_duplicates(
                subset=available_keys, keep="last"
            )

        logger.info(f"Combined data: {len(combined_df)} total records")
        return combined_df

    def save_updated_data(self, combined_df: pd.DataFrame) -> None:
        """
        Save updated data to blob storage.

        Args:
            combined_df: Combined DataFrame to save
        """
        logger.info("Saving updated data to blob storage")

        try:
            # Save locally first
            local_output = self.output_dir / "cholera_updated_data.csv"
            combined_df.to_csv(local_output, index=False)

            # TODO: Upload to blob using ocha_stratus
            # stratus.upload_csv_to_blob(
            #     combined_df,
            #     "cholera_historical_data.csv",
            #     stage=self.stage
            # )

            logger.info(f"Successfully saved {len(combined_df)} records")

        except Exception as e:
            logger.error(f"Failed to save updated data: {e}")
            raise

    def run_weekly_pipeline(self) -> Dict[str, Any]:
        """
        Execute the complete weekly pipeline.

        Returns:
            Pipeline execution summary
        """
        logger.info("Starting weekly cholera PDF ingest pipeline")
        start_time = datetime.now()

        try:
            # 1. Discover new PDFs
            new_pdf_urls = self.discover_new_pdfs()

            if not new_pdf_urls:
                logger.info("No new PDFs found. Pipeline complete.")
                return {
                    "status": "success",
                    "new_pdfs": 0,
                    "message": "No new PDFs to process",
                }

            # 2. Download PDFs
            pdf_paths = self.download_pdfs(new_pdf_urls)

            # 3. Upload to blob storage
            self.upload_pdfs_to_blob(pdf_paths)

            # 4. Process with LLM
            new_data = self.process_pdfs_with_llm(pdf_paths)

            # 5. Load historical data
            historical_data = self.load_historical_data()

            # 6. Merge data
            combined_data = self.merge_with_historical_data(new_data, historical_data)

            # 7. Save updated data
            self.save_updated_data(combined_data)

            # 8. Generate summary
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            summary = {
                "status": "success",
                "execution_time_seconds": execution_time,
                "new_pdfs_processed": len(pdf_paths),
                "new_records_extracted": len(new_data),
                "total_records": len(combined_data),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
            }

            logger.info(f"Weekly pipeline completed successfully: {summary}")
            return summary

        except Exception as e:
            logger.error(f"Weekly pipeline failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "execution_time_seconds": (datetime.now() - start_time).total_seconds(),
            }


def main():
    """Main execution function."""
    logging.basicConfig(level=logging.INFO)

    # TODO: Get API key from environment or config
    api_key = "your-openai-api-key"  # Replace with actual API key

    pipeline = WeeklyIngestPipeline(openai_api_key=api_key, stage="dev")

    summary = pipeline.run_weekly_pipeline()
    print(f"Pipeline execution summary: {summary}")


if __name__ == "__main__":
    main()
