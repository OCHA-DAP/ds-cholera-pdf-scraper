#!/usr/bin/env python3
"""
Download all historical PDFs and upload them to blob storage.

This script downloads all existing cholera PDFs from the source and uploads
them to blob storage using ocha_stratus for processing by the LLM pipeline.

Two-phase approach:
1. Download all PDFs to local directory (Google Drive shared folder)
2. Batch upload to blob storage after verification
"""

import logging
import time
from pathlib import Path
from typing import List, Optional

import ocha_stratus as stratus
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options
from urllib3.util.retry import Retry

# Use absolute imports as per copilot instructions
from src.config import Config

logger = logging.getLogger(__name__)


class HistoricalPDFDownloader:
    """Downloads and manages historical cholera PDFs."""

    def __init__(self, csv_url: str, stage: str = "dev"):
        self.csv_url = csv_url
        self.stage = stage
        self.local_download_dir = Config.HISTORICAL_PDFS_DIR
        self.blob_container = Config.BLOB_CONTAINER
        self.blob_proj_dir = Config.BLOB_PROJ_DIR

        # Create local download directory
        self.local_download_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Local download directory: {self.local_download_dir}")

        # Configure requests session with retry strategy
        self.session = requests.Session()

        # Add browser-like headers to handle iris.who.int URLs
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }
        )

        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=2,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def get_pdf_metadata(self) -> pd.DataFrame:
        """
        Get metadata for all PDFs from the CSV file.

        Returns:
            DataFrame with PDF metadata
        """
        logger.info(f"Loading PDF metadata from CSV: {self.csv_url}")

        try:
            # Read the CSV file
            df = pd.read_csv(self.csv_url)

            # Filter to only rows with valid PDF URLs
            if "LinktoDocument" in df.columns:
                df = df[df["LinktoDocument"].notna()]
                df = df[df["LinktoDocument"].str.lower().str.contains(".pdf")]

            logger.info(f"Loaded metadata for {len(df)} PDFs")
            return df

        except Exception as e:
            logger.error(f"Failed to load PDF metadata: {e}")
            return pd.DataFrame()

    def discover_pdf_urls(self) -> List[str]:
        """
        Discover all available PDF URLs from the CSV file.

        Returns:
            List of PDF URLs to download
        """
        logger.info(f"Loading PDF URLs from CSV: {self.csv_url}")

        try:
            # Read the CSV file from GitHub
            df = pd.read_csv(self.csv_url)

            # Extract URLs from the LinktoDocument column
            if "LinktoDocument" not in df.columns:
                logger.error("LinktoDocument column not found in CSV")
                return []

            # Get all non-null PDF URLs
            pdf_urls = df["LinktoDocument"].dropna().tolist()

            # Filter to only include PDF URLs
            pdf_urls = [url for url in pdf_urls if ".pdf" in url.lower()]

            logger.info(f"Found {len(pdf_urls)} PDF URLs in CSV")
            return pdf_urls

        except Exception as e:
            logger.error(f"Failed to load CSV file: {e}")
            return []

    def get_filename_from_metadata(
        self, pdf_url: str, metadata_df: pd.DataFrame
    ) -> str:
        """
        Generate a meaningful filename based on CSV metadata.

        Args:
            pdf_url: URL of the PDF
            metadata_df: DataFrame with PDF metadata

        Returns:
            Generated filename
        """
        try:
            # Find the row for this PDF URL
            row = metadata_df[metadata_df["LinktoDocument"] == pdf_url]

            if len(row) > 0:
                row = row.iloc[0]

                # Try to create filename from available columns
                # Priority: FileName column (exact match from CSV)
                if "FileName" in row and pd.notna(row["FileName"]):
                    return row["FileName"]
                elif "WeekNumber" in row and "Year" in row:
                    week = row["WeekNumber"]
                    year = row["Year"]
                    return f"OEW{week:02d}-{year}.pdf"

        except Exception as e:
            logger.warning(f"Could not generate filename from metadata: {e}")

        # Fallback to URL-based filename
        filename = pdf_url.split("/")[-1]
        if not filename.endswith(".pdf"):
            filename += ".pdf"
        return filename

    def validate_pdf_file(self, file_path: Path, min_size_kb: int = 10) -> bool:
        """
        Validate that a downloaded PDF file is not corrupted.

        Args:
            file_path: Path to the PDF file
            min_size_kb: Minimum file size in KB to consider valid

        Returns:
            True if file appears valid, False if corrupted
        """
        if not file_path.exists():
            return False

        # Check file size (corrupted files are typically 255-257 bytes)
        file_size = file_path.stat().st_size
        min_size_bytes = min_size_kb * 1024

        if file_size < min_size_bytes:
            logger.warning(
                f"File {file_path.name} is too small ({file_size} bytes), "
                f"likely corrupted"
            )
            return False

        # Basic PDF header check
        try:
            with open(file_path, "rb") as f:
                header = f.read(4)
                if not header.startswith(b"%PDF"):
                    logger.warning(
                        f"File {file_path.name} does not have valid PDF header"
                    )
                    return False
        except Exception as e:
            logger.warning(f"Error reading file {file_path.name}: {e}")
            return False

        return True

    def _resolve_iris_url_with_selenium(self, iris_url: str) -> Optional[str]:
        """
        Resolve iris.who.int URLs using browser automation.

        This is necessary because iris.who.int uses JavaScript-based redirects
        that cannot be handled by simple HTTP requests.

        Args:
            iris_url: The iris.who.int bitstream URL

        Returns:
            Direct download URL or None if resolution fails
        """
        if "iris.who.int/bitstream/handle/" not in iris_url:
            return None

        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")

        user_agent = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/118.0.0.0 Safari/537.36"
        )
        chrome_options.add_argument(f"--user-agent={user_agent}")

        driver = None
        try:
            logger.info(f"Attempting iris.who.int resolution for: {iris_url}")
            driver = webdriver.Chrome(options=chrome_options)

            # Navigate to the URL
            driver.get(iris_url)

            # Wait for JavaScript redirects
            time.sleep(3)

            final_url = driver.current_url

            if final_url != iris_url and "bitstreams/" in final_url:
                logger.info(f"Successfully resolved iris URL: {final_url}")
                return final_url
            else:
                logger.warning(f"No valid redirect found for {iris_url}")
                return None

        except WebDriverException as e:
            logger.error(f"WebDriver error resolving {iris_url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error resolving iris URL {iris_url}: {e}")
            return None
        finally:
            if driver:
                try:
                    driver.quit()
                except Exception:
                    pass  # Ignore cleanup errors

    def download_pdf(
        self, pdf_url: str, filename: Optional[str] = None, max_retries: int = 3
    ) -> Path:
        """
        Download a single PDF file with corruption detection and retry logic.

        Args:
            pdf_url: URL of the PDF to download
            filename: Optional custom filename, otherwise inferred from URL
            max_retries: Maximum number of retry attempts for corrupted files

        Returns:
            Path to the downloaded file
        """
        if not filename:
            filename = pdf_url.split("/")[-1]
            if not filename.endswith(".pdf"):
                filename += ".pdf"

        local_path = self.local_download_dir / filename
        download_url = pdf_url

        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"Retry {attempt}/{max_retries} for {filename}")

                logger.info(f"Downloading {download_url} to {local_path}")

                # Add a small delay to avoid overwhelming the server
                time.sleep(0.5)

                # Follow redirects to get the actual PDF content
                response = self.session.get(
                    download_url, stream=True, timeout=30, allow_redirects=True
                )

                response.raise_for_status()

                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Validate the downloaded file
                if self.validate_pdf_file(local_path):
                    logger.info(f"Successfully downloaded {filename}")
                    return local_path
                else:
                    logger.warning(
                        f"Downloaded file {filename} appears corrupted, "
                        f"attempt {attempt + 1}/{max_retries + 1}"
                    )
                    if local_path.exists():
                        local_path.unlink()  # Remove corrupted file

                    # Try iris.who.int resolution as fallback if this is the first attempt
                    if attempt == 0 and "iris.who.int/bitstream/handle/" in pdf_url:

                        logger.info(
                            f"Attempting iris.who.int resolution fallback for corrupted {filename}"
                        )
                        resolved_url = self._resolve_iris_url_with_selenium(pdf_url)

                        if resolved_url:
                            logger.info(
                                f"Got resolved URL, retrying download: {resolved_url}"
                            )
                            download_url = (
                                resolved_url  # Use resolved URL for remaining attempts
                            )
                            time.sleep(1)
                            continue
                        else:
                            logger.warning(
                                f"iris.who.int resolution failed for {filename}"
                            )

                    if attempt < max_retries:
                        # Wait before retry
                        time.sleep(2.0 * (attempt + 1))
                        continue
                    else:
                        raise ValueError(
                            f"Failed to download valid file after "
                            f"{max_retries + 1} attempts"
                        )

            except requests.RequestException as e:
                logger.error(f"Failed to download {pdf_url}: {e}")

                # Try iris.who.int resolution as fallback if this is the first attempt
                if attempt == 0 and "iris.who.int/bitstream/handle/" in pdf_url:

                    logger.info(
                        f"Attempting iris.who.int resolution fallback for {filename}"
                    )
                    resolved_url = self._resolve_iris_url_with_selenium(pdf_url)

                    if resolved_url:
                        logger.info(
                            f"Got resolved URL, retrying download: {resolved_url}"
                        )
                        download_url = (
                            resolved_url  # Use resolved URL for remaining attempts
                        )
                        time.sleep(1)
                        continue
                    else:
                        logger.warning(f"iris.who.int resolution failed for {filename}")

                if attempt < max_retries:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                else:
                    raise

    def upload_file_to_blob(
        self, local_path: Path, blob_name: Optional[str] = None
    ) -> None:
        """
        Upload a single PDF file to blob storage.

        Args:
            local_path: Local path to the PDF file
            blob_name: Optional blob name. If None, uses filename
        """
        if not local_path.exists():
            logger.error(f"File not found: {local_path}")
            return

        if blob_name is None:
            blob_name = local_path.name

        # Create the blob path: {BLOB_PROJ_DIR}/raw/pdfs/{filename}
        blob_path = f"{self.blob_proj_dir}/raw/pdfs/{blob_name}"

        try:
            logger.info(f"Uploading {local_path.name} to {blob_path}")

            # Use stratus to get the container client with proper credentials
            container_client = stratus.get_container_client(
                container_name=self.blob_container, stage=self.stage, write=True
            )

            # Upload the file
            with open(local_path, "rb") as data:
                container_client.upload_blob(name=blob_path, data=data, overwrite=True)

            logger.info(f"Successfully uploaded {blob_name}")

        except Exception as e:
            logger.error(f"Failed to upload {blob_name}: {str(e)}")
            raise

    def cleanup_corrupted_files(self) -> List[Path]:
        """
        Remove corrupted PDF files from the download directory.

        Returns:
            List of removed file paths
        """
        logger.info("Cleaning up corrupted files")

        pdf_files = list(self.local_download_dir.glob("*.pdf"))
        corrupted_files = []

        for pdf_file in pdf_files:
            if not self.validate_pdf_file(pdf_file):
                logger.info(f"Removing corrupted file: {pdf_file.name}")
                try:
                    pdf_file.unlink()
                    corrupted_files.append(pdf_file)
                except Exception as e:
                    logger.error(f"Failed to remove {pdf_file.name}: {e}")

        logger.info(f"Removed {len(corrupted_files)} corrupted files")
        return corrupted_files

    def download_all_pdfs(self) -> List[Path]:
        """
        Download all historical PDFs to local directory.

        Returns:
            List of successfully downloaded file paths
        """
        logger.info("Starting historical PDF download process")

        # Clean up any existing corrupted files first
        self.cleanup_corrupted_files()

        # Get metadata first for better filename generation
        metadata_df = self.get_pdf_metadata()
        pdf_urls = self.discover_pdf_urls()

        logger.info(f"Found {len(pdf_urls)} PDFs to download")

        downloaded_files = []
        failed_downloads = []

        for i, pdf_url in enumerate(pdf_urls, 1):
            try:
                logger.info(f"Downloading PDF {i}/{len(pdf_urls)}: {pdf_url}")

                # Generate filename using metadata
                filename = self.get_filename_from_metadata(pdf_url, metadata_df)

                # Check if file already exists
                local_path = self.local_download_dir / filename
                if local_path.exists():
                    logger.info(f"File already exists, skipping: {filename}")
                    downloaded_files.append(local_path)
                    continue

                # Download PDF
                local_path = self.download_pdf(pdf_url, filename)
                downloaded_files.append(local_path)

                # Add delay between downloads to be respectful to the server
                if i % 10 == 0:  # Longer pause every 10 downloads
                    logger.info(f"Processed {i} files, taking a longer break...")
                    time.sleep(2.0)

            except requests.exceptions.HTTPError as e:
                if "429" in str(e):
                    logger.warning(f"Rate limited on {pdf_url}, waiting 10 seconds...")
                    time.sleep(10)
                    failed_downloads.append(pdf_url)
                else:
                    logger.error(f"HTTP error downloading {pdf_url}: {e}")
                    failed_downloads.append(pdf_url)
                continue
            except Exception as e:
                logger.error(f"Failed to download {pdf_url}: {e}")
                failed_downloads.append(pdf_url)
                continue

        logger.info(
            f"Download completed: {len(downloaded_files)}/{len(pdf_urls)} files"
        )

        if failed_downloads:
            logger.warning(f"Failed to download {len(failed_downloads)} files:")
            for url in failed_downloads[:10]:  # Show first 10 failures
                logger.warning(f"  - {url}")
            if len(failed_downloads) > 10:
                logger.warning(f"  ... and {len(failed_downloads) - 10} more")

        return downloaded_files

    def upload_all_to_blob(self, local_files: Optional[List[Path]] = None) -> None:
        """
        Upload all downloaded PDFs to blob storage.

        Args:
            local_files: Optional list of files to upload. If None, uploads all PDFs in download dir.
        """
        if local_files is None:
            # Find all PDF files in the download directory
            local_files = list(self.local_download_dir.glob("*.pdf"))

        logger.info(f"Starting blob upload for {len(local_files)} files")

        uploaded_count = 0
        for i, local_path in enumerate(local_files, 1):
            try:
                logger.info(f"Uploading file {i}/{len(local_files)}: {local_path.name}")
                self.upload_file_to_blob(local_path)
                uploaded_count += 1

            except Exception as e:
                logger.error(f"Failed to upload {local_path}: {e}")
                continue

        logger.info(f"Blob upload completed: {uploaded_count}/{len(local_files)} files")

    def process_all_historical_pdfs(self) -> None:
        """Download all historical PDFs and upload to blob storage."""
        logger.info("Starting historical PDF download and upload process")

        # Get metadata first for better filename generation
        metadata_df = self.get_pdf_metadata()
        pdf_urls = self.discover_pdf_urls()

        logger.info(f"Found {len(pdf_urls)} PDFs to process")

        for i, pdf_url in enumerate(pdf_urls, 1):
            try:
                logger.info(f"Processing PDF {i}/{len(pdf_urls)}: {pdf_url}")

                # Generate filename using metadata
                filename = self.get_filename_from_metadata(pdf_url, metadata_df)

                # Download PDF
                local_path = self.download_pdf(pdf_url, filename)

                # Upload to blob
                self.upload_file_to_blob(local_path)

            except Exception as e:
                logger.error(f"Failed to process {pdf_url}: {e}")
                continue

        logger.info("Completed historical PDF processing")

    def cleanup_local_directory(self) -> None:
        """
        Clean up local directory by removing files that don't match CSV filenames.

        This removes any incorrectly named files from previous download attempts
        and keeps only files that match the FileName column in the CSV.
        """
        logger.info("Starting local directory cleanup")

        # Get valid filenames from CSV
        metadata_df = self.get_pdf_metadata()
        if metadata_df.empty:
            logger.error("No metadata found, cannot perform cleanup")
            return

        # Get all valid filenames from the FileName column
        valid_filenames = set()
        for _, row in metadata_df.iterrows():
            if "FileName" in row and pd.notna(row["FileName"]):
                valid_filenames.add(row["FileName"])

        logger.info(f"Found {len(valid_filenames)} valid filenames in CSV")

        # Get all PDF files currently in the download directory
        current_files = list(self.local_download_dir.glob("*.pdf"))
        logger.info(f"Found {len(current_files)} PDF files in local directory")

        # Check each file and remove if not in valid list
        removed_count = 0
        kept_count = 0

        for file_path in current_files:
            filename = file_path.name

            if filename in valid_filenames:
                logger.debug(f"Keeping valid file: {filename}")
                kept_count += 1
            else:
                logger.info(f"Removing invalid file: {filename}")
                try:
                    file_path.unlink()  # Delete the file
                    removed_count += 1
                except Exception as e:
                    logger.error(f"Failed to remove {filename}: {e}")

        logger.info(f"Cleanup completed:")
        logger.info(f"  - Kept {kept_count} valid files")
        logger.info(f"  - Removed {removed_count} invalid files")

        # Show some examples of valid filenames for reference
        if valid_filenames:
            sample_filenames = list(valid_filenames)[:5]
            logger.info(f"Sample valid filenames: {sample_filenames}")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download historical cholera PDFs and upload to blob storage"
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download PDFs, skip blob upload",
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Only upload existing PDFs to blob, skip download",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up local directory by removing files not in CSV",
    )
    parser.add_argument(
        "--remove-corrupted",
        action="store_true",
        help="Remove corrupted PDF files (small files with invalid headers)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # CSV URL containing the PDF links
    csv_url = "https://github.com/CBPFGMS/pfbi-data/raw/main/who_download_log.csv"

    downloader = HistoricalPDFDownloader(csv_url, stage=Config.STAGE)

    if args.remove_corrupted:
        logger.info("Corrupted file cleanup mode: removing invalid PDF files")
        downloader.cleanup_corrupted_files()
    elif args.cleanup:
        logger.info("Cleanup mode: removing invalid files from local directory")
        downloader.cleanup_local_directory()
    elif args.upload_only:
        logger.info("Upload-only mode: uploading existing PDFs to blob storage")
        downloader.upload_all_to_blob()
    elif args.download_only:
        logger.info("Download-only mode: downloading PDFs to local directory")
        downloaded_files = downloader.download_all_pdfs()
        logger.info(
            f"Downloaded {len(downloaded_files)} files to {downloader.local_download_dir}"
        )
    else:
        logger.info("Full mode: downloading PDFs and uploading to blob storage")
        downloaded_files = downloader.download_all_pdfs()
        downloader.upload_all_to_blob(downloaded_files)


if __name__ == "__main__":
    main()
