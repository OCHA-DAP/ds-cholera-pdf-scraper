#!/usr/bin/env python3
"""
Download the latest WHO cholera PDF from AFRO website.

Minimal self-contained script for quick production deployment.
All dependencies inlined - no imports from src/utils required.

Usage:
    # List available weeks
    python scripts/download_latest_who_pdf.py --list

    # Download latest week
    python scripts/download_latest_who_pdf.py

    # Download specific week
    python scripts/download_latest_who_pdf.py --week 37

    # Download and upload to blob
    python scripts/download_latest_who_pdf.py --upload

Requirements:
    beautifulsoup4, selenium, requests, ocha-stratus, azure-storage-blob
"""

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import ocha_stratus as stratus
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options
from urllib3.util.retry import Retry

# Import Config for centralized blob paths
import sys
from pathlib import Path as PathLib
sys.path.insert(0, str(PathLib(__file__).parent.parent))
from src.config import Config

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default paths (can be overridden with --output-dir)
HISTORICAL_PDFS_DIR = Path.home() / "Downloads" / "who_cholera_pdfs"
BLOB_CONTAINER = "projects"
BLOB_PROJ_DIR = "ds-cholera-pdf-scraper"


# =============================================================================
# UTILITY FUNCTIONS (inlined for self-contained deployment)
# =============================================================================

def create_download_session() -> requests.Session:
    """Create a requests session configured for WHO PDF downloads."""
    session = requests.Session()

    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    })

    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        backoff_factor=2,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


def create_chrome_options(headless: bool = True) -> Options:
    """Create Chrome options for Selenium WebDriver."""
    chrome_options = Options()

    if headless:
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

    return chrome_options


def resolve_iris_url_with_selenium(iris_url: str) -> Optional[str]:
    """
    Resolve iris.who.int URLs using browser automation.

    Necessary because iris.who.int uses JavaScript-based redirects.
    Handles both /bitstream/handle/ and /bitstreams/ URL patterns.
    """
    # Check if this is an iris.who.int URL
    if "iris.who.int" not in iris_url:
        return None

    chrome_options = create_chrome_options(headless=True)
    driver = None

    try:
        logger.info(f"Resolving iris.who.int URL with Selenium: {iris_url[:60]}...")
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(iris_url)
        time.sleep(5)  # Wait longer for JavaScript redirects

        final_url = driver.current_url

        # Check if we got redirected to a different URL
        if final_url != iris_url:
            # Look for any of these patterns in the resolved URL
            valid_patterns = [
                "bitstreams/",
                "/bitstream/",
                "/content",
                "/download"
            ]
            if any(pattern in final_url for pattern in valid_patterns):
                logger.info(f"Successfully resolved to: {final_url[:60]}...")
                return final_url

        logger.warning(f"No valid redirect found for {iris_url}")
        return None

    except WebDriverException as e:
        logger.error(f"WebDriver error: {e}")
        return None
    except Exception as e:
        logger.error(f"Error resolving iris URL: {e}")
        return None
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass


def validate_pdf_file(file_path: Path, min_size_kb: int = 10) -> bool:
    """Validate that a downloaded PDF file is not corrupted."""
    if not file_path.exists():
        return False

    # Check file size (corrupted files are typically 255-257 bytes)
    file_size = file_path.stat().st_size
    min_size_bytes = min_size_kb * 1024

    if file_size < min_size_bytes:
        logger.warning(f"File {file_path.name} is too small ({file_size} bytes), likely corrupted")
        return False

    # Basic PDF header check
    try:
        with open(file_path, "rb") as f:
            header = f.read(4)
            if not header.startswith(b"%PDF"):
                logger.warning(f"File {file_path.name} does not have valid PDF header")
                return False
    except Exception as e:
        logger.warning(f"Error reading file {file_path.name}: {e}")
        return False

    return True


def download_pdf_with_retry(
    pdf_url: str,
    local_path: Path,
    session: Optional[requests.Session] = None,
    max_retries: int = 3,
) -> bool:
    """Download a PDF file with corruption detection and retry logic."""
    if session is None:
        session = create_download_session()

    download_url = pdf_url

    # For iris.who.int URLs, resolve them FIRST before attempting download
    if "iris.who.int" in pdf_url:
        logger.info("Detected iris.who.int URL, resolving with Selenium first...")
        resolved_url = resolve_iris_url_with_selenium(pdf_url)
        if resolved_url:
            download_url = resolved_url
            logger.info(f"Using resolved URL: {download_url[:60]}...")
        else:
            logger.warning("Selenium resolution failed, will try direct download")

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                logger.info(f"Retry {attempt}/{max_retries} for {local_path.name}")

            logger.info(f"Downloading {download_url[:60]}... to {local_path.name}")
            time.sleep(0.5)  # Be respectful to the server

            response = session.get(download_url, stream=True, timeout=30, allow_redirects=True)
            response.raise_for_status()

            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Validate the downloaded file
            if validate_pdf_file(local_path):
                logger.info(f"Successfully downloaded {local_path.name}")
                return True
            else:
                logger.warning(f"Downloaded file appears corrupted, attempt {attempt + 1}/{max_retries + 1}")
                if local_path.exists():
                    local_path.unlink()

                # Try iris.who.int resolution as fallback on first attempt
                if attempt == 0 and "iris.who.int/bitstream" in pdf_url:
                    logger.info("Attempting iris.who.int resolution fallback...")
                    resolved_url = resolve_iris_url_with_selenium(pdf_url)

                    if resolved_url:
                        download_url = resolved_url
                        time.sleep(1)
                        continue

                if attempt < max_retries:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                else:
                    logger.error(f"Failed to download valid file after {max_retries + 1} attempts")
                    return False

        except requests.RequestException as e:
            logger.error(f"Failed to download {pdf_url}: {e}")

            # Try iris.who.int resolution as fallback on first attempt
            if attempt == 0 and "iris.who.int/bitstream" in pdf_url:
                logger.info("Attempting iris.who.int resolution fallback...")
                resolved_url = resolve_iris_url_with_selenium(pdf_url)

                if resolved_url:
                    download_url = resolved_url
                    time.sleep(1)
                    continue

            if attempt < max_retries:
                time.sleep(2.0 * (attempt + 1))
                continue
            else:
                logger.error(f"Failed after {max_retries + 1} attempts")
                return False

    return False


# =============================================================================
# MAIN DOWNLOADER CLASS
# =============================================================================

@dataclass
class BulletinMetadata:
    """Metadata for a weekly bulletin."""
    week: int
    year: int
    date_range: str
    pdf_url: str
    page_url: str
    publication_date: Optional[str] = None
    downloaded_path: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    def get_filename(self) -> str:
        """Generate filename based on week and year."""
        return f"OEW{self.week:02d}-{self.year}.pdf"


@dataclass
class DownloadRunMetadata:
    """Metadata about a download run execution."""
    # Bulletin info
    week: Optional[int]
    year: Optional[int]
    date_range: Optional[str]
    pdf_url: Optional[str]

    # Run context
    run_date: str
    status: str  # "success" or "failed"
    error_message: Optional[str]
    runner: str  # "github-actions" or "local"
    trigger: Optional[str]  # "schedule", "workflow_dispatch", etc.
    run_id: Optional[str]
    run_url: Optional[str]

    # Outcome
    blob_uploaded: bool
    blob_path: Optional[str]
    local_path: Optional[str]
    file_size_bytes: Optional[int]
    download_duration_seconds: Optional[float]

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    def to_jsonl(self) -> str:
        """Convert to JSONL format (single line JSON)."""
        return json.dumps(self.to_dict())


def get_run_context() -> Dict[str, Optional[str]]:
    """
    Detect execution context (GitHub Actions vs local).

    Returns dict with runner, run_id, run_url, and trigger.
    """
    github_run_id = os.getenv("GITHUB_RUN_ID")

    if github_run_id:
        github_repo = os.getenv("GITHUB_REPOSITORY", "unknown/unknown")
        return {
            "runner": "github-actions",
            "run_id": github_run_id,
            "run_url": f"https://github.com/{github_repo}/actions/runs/{github_run_id}",
            "trigger": os.getenv("GITHUB_EVENT_NAME"),
        }
    else:
        return {
            "runner": "local",
            "run_id": None,
            "run_url": None,
            "trigger": None,
        }


class LatestWHOPDFDownloader:
    """Downloads latest weekly bulletins from AFRO WHO website."""

    AFRO_OUTBREAKS_URL = (
        "https://www.afro.who.int/health-topics/disease-outbreaks/"
        "outbreaks-and-other-emergencies-updates"
    )

    def __init__(self, stage: str = "dev", output_dir: Optional[Path] = None):
        """
        Initialize the downloader.

        Args:
            stage: Environment stage (dev/staging/prod)
            output_dir: Custom output directory
        """
        self.stage = stage
        self.output_dir = output_dir or HISTORICAL_PDFS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session = create_download_session()
        self.blob_container = BLOB_CONTAINER
        self.blob_proj_dir = BLOB_PROJ_DIR

        logger.info(f"Output directory: {self.output_dir}")

    def scrape_weekly_bulletins(self) -> List[BulletinMetadata]:
        """Scrape the AFRO WHO page for weekly bulletin links."""
        logger.info(f"Scraping weekly bulletins from {self.AFRO_OUTBREAKS_URL}")

        chrome_options = create_chrome_options(headless=True)
        driver = None
        bulletins = []

        try:
            driver = webdriver.Chrome(options=chrome_options)
            driver.get(self.AFRO_OUTBREAKS_URL)
            time.sleep(3)  # Wait for page to load

            soup = BeautifulSoup(driver.page_source, "html.parser")

            # Find all links matching "Week XX: DD to DD Month YYYY"
            week_pattern = re.compile(
                r"Week\s+(\d+):\s+([\d\s]+to[\d\s]+\w+\s+\d{4})", re.IGNORECASE
            )

            for link in soup.find_all("a", href=True):
                link_text = link.get_text(strip=True)
                match = week_pattern.search(link_text)

                if match and "iris.who.int" in link["href"]:
                    week_num = int(match.group(1))
                    date_range = match.group(2).strip()

                    # Extract year from date range
                    year_match = re.search(r"\d{4}", date_range)
                    year = int(year_match.group(0)) if year_match else datetime.now().year

                    bulletin = BulletinMetadata(
                        week=week_num,
                        year=year,
                        date_range=date_range,
                        pdf_url=link["href"],
                        page_url=self.AFRO_OUTBREAKS_URL,
                    )

                    # Try to find publication date
                    parent = link.find_parent()
                    if parent:
                        date_pattern = re.compile(r"(\d{1,2}\s+\w+\s+\d{4})|(\w+\s+\d{1,2},\s+\d{4})")
                        date_match = date_pattern.search(parent.get_text())
                        if date_match:
                            bulletin.publication_date = date_match.group(0)

                    bulletins.append(bulletin)
                    logger.debug(f"Found bulletin: Week {week_num}, Year {year}")

            # Sort by year (descending) then week (descending)
            bulletins.sort(key=lambda x: (x.year, x.week), reverse=True)

            logger.info(f"Found {len(bulletins)} weekly bulletins")
            return bulletins

        except Exception as e:
            logger.error(f"Error scraping weekly bulletins: {e}")
            return []
        finally:
            if driver:
                try:
                    driver.quit()
                except Exception:
                    pass

    def get_latest_bulletin(self) -> Optional[BulletinMetadata]:
        """Get the latest weekly bulletin."""
        bulletins = self.scrape_weekly_bulletins()

        if not bulletins:
            logger.error("No bulletins found")
            return None

        latest = bulletins[0]
        logger.info(f"Latest bulletin: Week {latest.week}, Year {latest.year}")
        return latest

    def get_bulletin_by_week(self, week_num: int) -> Optional[BulletinMetadata]:
        """Get a specific weekly bulletin by week number."""
        bulletins = self.scrape_weekly_bulletins()

        # Find bulletin with matching week number (prefer current year)
        current_year = datetime.now().year
        matching = [b for b in bulletins if b.week == week_num]

        if not matching:
            logger.error(f"No bulletin found for week {week_num}")
            return None

        # Prefer current year, otherwise take most recent
        current_year_matches = [b for b in matching if b.year == current_year]
        bulletin = current_year_matches[0] if current_year_matches else matching[0]

        logger.info(f"Found bulletin: Week {bulletin.week}, Year {bulletin.year}")
        return bulletin

    def download_bulletin(
        self, bulletin: BulletinMetadata, upload_to_blob: bool = False
    ) -> Optional[BulletinMetadata]:
        """Download a bulletin PDF and optionally upload to blob storage."""
        filename = bulletin.get_filename()
        local_path = self.output_dir / filename

        # Check if already exists
        if local_path.exists():
            logger.info(f"File already exists: {local_path}")
            bulletin.downloaded_path = str(local_path)
            return bulletin

        logger.info(f"Downloading bulletin: {filename}")

        # Download the PDF
        success = download_pdf_with_retry(
            pdf_url=bulletin.pdf_url,
            local_path=local_path,
            session=self.session,
            max_retries=3,
        )

        if not success:
            logger.error(f"Failed to download {filename}")
            return None

        bulletin.downloaded_path = str(local_path)
        logger.info(f"Successfully downloaded to {local_path}")

        # Upload to blob if requested
        if upload_to_blob:
            try:
                self.upload_to_blob(local_path, filename)
                # Clean up local file after successful upload
                local_path.unlink()
                logger.info(f"Removed local file after upload: {local_path.name}")
                bulletin.downloaded_path = None  # File no longer exists locally
            except Exception as e:
                logger.error(f"Failed to upload to blob: {e}")
                # Don't fail the whole operation

        return bulletin

    def upload_to_blob(self, local_path: Path, blob_name: Optional[str] = None) -> None:
        """Upload a PDF file to blob storage using stratus."""
        if not local_path.exists():
            logger.error(f"File not found: {local_path}")
            return

        if blob_name is None:
            blob_name = local_path.name

        # Use centralized blob path from Config
        blob_base_path = Config.get_blob_paths()["raw_pdfs"]
        blob_path = f"{blob_base_path}{blob_name}"

        logger.info(f"Uploading {local_path.name} to {blob_path}")

        with open(local_path, "rb") as f:
            stratus.upload_blob_data(
                data=f,
                blob_name=blob_path,
                stage=self.stage,
                container_name=self.blob_container,
                content_type="application/pdf",
            )

        logger.info(f"Successfully uploaded {blob_name}")

    def list_available_bulletins(self) -> None:
        """List all available weekly bulletins."""
        bulletins = self.scrape_weekly_bulletins()

        if not bulletins:
            print("No bulletins found.")
            return

        print(f"\nFound {len(bulletins)} available bulletins:\n")
        print(f"{'Week':<6} {'Year':<6} {'Date Range':<35} {'PDF URL'}")
        print("-" * 100)

        for bulletin in bulletins:
            url_short = bulletin.pdf_url[:60] + "..." if len(bulletin.pdf_url) > 60 else bulletin.pdf_url
            print(f"{bulletin.week:<6} {bulletin.year:<6} {bulletin.date_range:<35} {url_short}")

    def check_pdf_exists_in_blob(self, filename: str) -> bool:
        """
        Check if a PDF already exists in blob storage.

        Returns True if exists, False otherwise.
        """
        # Use centralized blob path from Config
        blob_base_path = Config.get_blob_paths()["raw_pdfs"]
        blob_path = f"{blob_base_path}{filename}"

        try:
            # Try to get blob properties (lightweight check)
            from azure.storage.blob import BlobServiceClient

            # Get connection string from stratus
            sas_token = os.getenv(f"DSCI_AZ_BLOB_{self.stage.upper()}_SAS_WRITE")
            if not sas_token:
                logger.warning(f"No SAS token found for stage {self.stage}, cannot check blob")
                return False

            account_url = f"https://imb0chd0{self.stage}.blob.core.windows.net"
            blob_service_client = BlobServiceClient(account_url=account_url, credential=sas_token)
            blob_client = blob_service_client.get_blob_client(
                container=self.blob_container,
                blob=blob_path
            )

            # Check if blob exists
            exists = blob_client.exists()
            if exists:
                logger.info(f"PDF already exists in blob: {blob_path}")
            else:
                logger.info(f"PDF not found in blob: {blob_path}")
            return exists

        except Exception as e:
            logger.warning(f"Error checking blob existence: {e}")
            return False

    def download_log_from_blob(self, log_filename: str = "download_log.jsonl") -> Optional[Path]:
        """
        Download existing log file from blob storage.

        Returns local path if successful, None otherwise.
        """
        local_log_path = self.output_dir / log_filename
        blob_path = f"{self.blob_proj_dir}/raw/monitoring/{log_filename}"

        try:
            logger.info(f"Attempting to download existing log from blob: {blob_path}")
            # Download blob data
            blob_data = stratus.load_blob_data(
                blob_name=blob_path,
                stage=self.stage,
                container_name=self.blob_container,
            )
            # Write to local file
            with open(local_log_path, 'wb') as f:
                f.write(blob_data)

            # Count entries
            num_entries = sum(1 for _ in open(local_log_path))
            logger.info(f"✓ Downloaded existing log with {num_entries} entries from blob")
            return local_log_path
        except Exception as e:
            # Log doesn't exist yet or download failed - that's okay for first run
            logger.warning(f"Could not download existing log from blob: {e}")
            logger.info("Starting fresh log file (this is normal for first run)")
            return None

    def upload_log_to_blob(self, log_path: Path) -> None:
        """Upload log file to blob storage."""
        if not log_path.exists():
            logger.warning(f"Log file not found: {log_path}")
            return

        # Count entries before uploading
        num_entries = sum(1 for _ in open(log_path))

        blob_path = f"{self.blob_proj_dir}/raw/monitoring/{log_path.name}"
        logger.info(f"Uploading log with {num_entries} entries to {blob_path}")

        with open(log_path, "rb") as f:
            stratus.upload_blob_data(
                data=f,
                blob_name=blob_path,
                stage=self.stage,
                container_name=self.blob_container,
                content_type="application/x-ndjson",
            )

        logger.info(f"✓ Successfully uploaded {log_path.name} with {num_entries} total entries")

    def append_to_log(self, run_metadata: DownloadRunMetadata, log_filename: str = "download_log.jsonl") -> None:
        """
        Append run metadata to JSONL log file.

        Creates the log file if it doesn't exist.
        """
        log_path = self.output_dir / log_filename

        try:
            with open(log_path, "a") as f:
                f.write(run_metadata.to_jsonl() + "\n")
            logger.info(f"Appended run metadata to {log_path}")
        except Exception as e:
            logger.error(f"Failed to append to log file: {e}")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download latest WHO cholera PDF from AFRO website"
    )
    parser.add_argument(
        "--week",
        type=int,
        help="Download specific week number (default: latest)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to blob storage after download",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Custom output directory",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available bulletins without downloading",
    )
    parser.add_argument(
        "--save-metadata",
        type=Path,
        help="Save bulletin metadata to JSON file",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Get stage from environment
    stage = os.getenv("STAGE", "dev")

    downloader = LatestWHOPDFDownloader(
        stage=stage,
        output_dir=args.output_dir,
    )

    if args.list:
        downloader.list_available_bulletins()
        return

    # Get run context (GitHub Actions vs local)
    run_context = get_run_context()
    start_time = time.time()
    run_date = datetime.now().isoformat()

    # Initialize run metadata (will be populated as we go)
    run_metadata = None
    bulletin = None
    result = None

    try:
        # Get the bulletin info first (always need to know what's latest on WHO)
        if args.week:
            bulletin = downloader.get_bulletin_by_week(args.week)
        else:
            bulletin = downloader.get_latest_bulletin()

        # For runs with upload enabled: check if we already have this week in blob
        # This avoids re-downloading and re-uploading the same file
        if bulletin and args.upload:
            expected_filename = bulletin.get_filename()

            if downloader.check_pdf_exists_in_blob(expected_filename):
                logger.info(f"✓ Bulletin {expected_filename} already exists in blob - skipping download")

                # Create metadata for this redundant run
                blob_base_path = Config.get_blob_paths()["raw_pdfs"]
                blob_path = f"{blob_base_path}{expected_filename}"

                run_metadata = DownloadRunMetadata(
                    week=bulletin.week,
                    year=bulletin.year,
                    date_range=bulletin.date_range,
                    pdf_url=bulletin.pdf_url,
                    run_date=run_date,
                    status="already_exists",  # New status for clarity
                    error_message=None,
                    runner=run_context["runner"],
                    trigger=run_context["trigger"],
                    run_id=run_context["run_id"],
                    run_url=run_context["run_url"],
                    blob_uploaded=True,  # Already there
                    blob_path=blob_path,
                    local_path=None,
                    file_size_bytes=None,
                    download_duration_seconds=time.time() - start_time,
                )

                print(
                    f"\n✓ Bulletin Week {bulletin.week}, {bulletin.year} "
                    "already exists in blob storage"
                )
                print(f"  Blob path: {blob_path}")
                print("  No action needed - skipping download")

                # Create status file for GitHub Actions summary
                if args.save_metadata:
                    status_info = {
                        "status": "already_exists",
                        "week": bulletin.week,
                        "year": bulletin.year,
                        "date_range": bulletin.date_range,
                        "blob_path": blob_path,
                    }
                    with open(args.save_metadata, "w") as f:
                        json.dump(status_info, f, indent=2)

                # Exit early - no PDF download needed
                # Logging will be handled by finally block
                return

        if not bulletin:
            logger.error("No bulletin found")
            # Create failed run metadata
            run_metadata = DownloadRunMetadata(
                week=args.week if args.week else None,
                year=None,
                date_range=None,
                pdf_url=None,
                run_date=run_date,
                status="failed",
                error_message="No bulletin found",
                runner=run_context["runner"],
                trigger=run_context["trigger"],
                run_id=run_context["run_id"],
                run_url=run_context["run_url"],
                blob_uploaded=False,
                blob_path=None,
                local_path=None,
                file_size_bytes=None,
                download_duration_seconds=time.time() - start_time,
            )
            return

        # Display bulletin info
        print("\nBulletin Information:")
        print(f"  Week:       {bulletin.week}")
        print(f"  Year:       {bulletin.year}")
        print(f"  Date Range: {bulletin.date_range}")
        print(f"  PDF URL:    {bulletin.pdf_url}")
        if bulletin.publication_date:
            print(f"  Published:  {bulletin.publication_date}")
        print()

        # Download the bulletin
        result = downloader.download_bulletin(bulletin, upload_to_blob=args.upload)

        if not result:
            logger.error("Download failed")
            # Create failed run metadata
            run_metadata = DownloadRunMetadata(
                week=bulletin.week,
                year=bulletin.year,
                date_range=bulletin.date_range,
                pdf_url=bulletin.pdf_url,
                run_date=run_date,
                status="failed",
                error_message="Download failed",
                runner=run_context["runner"],
                trigger=run_context["trigger"],
                run_id=run_context["run_id"],
                run_url=run_context["run_url"],
                blob_uploaded=False,
                blob_path=None,
                local_path=None,
                file_size_bytes=None,
                download_duration_seconds=time.time() - start_time,
            )
            return

        print(f"\nSuccessfully downloaded to: {result.downloaded_path}")

        # Calculate file size if local file exists
        file_size = None
        local_path = result.downloaded_path
        if result.downloaded_path:
            try:
                file_size = Path(result.downloaded_path).stat().st_size
            except Exception:
                pass

        # Determine blob path if uploaded
        blob_path = None
        if args.upload:
            # Use centralized blob path from Config
            blob_base_path = Config.get_blob_paths()["raw_pdfs"]
            blob_path = f"{blob_base_path}{bulletin.get_filename()}"

        # Create successful run metadata
        run_metadata = DownloadRunMetadata(
            week=bulletin.week,
            year=bulletin.year,
            date_range=bulletin.date_range,
            pdf_url=bulletin.pdf_url,
            run_date=run_date,
            status="success",
            error_message=None,
            runner=run_context["runner"],
            trigger=run_context["trigger"],
            run_id=run_context["run_id"],
            run_url=run_context["run_url"],
            blob_uploaded=args.upload,
            blob_path=blob_path,
            local_path=local_path,
            file_size_bytes=file_size,
            download_duration_seconds=time.time() - start_time,
        )

        # Save metadata if requested (for GHA summary display)
        if args.save_metadata:
            with open(args.save_metadata, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"Metadata saved to: {args.save_metadata}")

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        # Create failed run metadata
        run_metadata = DownloadRunMetadata(
            week=bulletin.week if bulletin else (args.week if args.week else None),
            year=bulletin.year if bulletin else None,
            date_range=bulletin.date_range if bulletin else None,
            pdf_url=bulletin.pdf_url if bulletin else None,
            run_date=run_date,
            status="failed",
            error_message=str(e),
            runner=run_context["runner"],
            trigger=run_context["trigger"],
            run_id=run_context["run_id"],
            run_url=run_context["run_url"],
            blob_uploaded=False,
            blob_path=None,
            local_path=None,
            file_size_bytes=None,
            download_duration_seconds=time.time() - start_time,
        )
        raise

    finally:
        # Always append to log if we have run metadata
        if run_metadata:
            # If uploading to blob, download existing log first
            if args.upload:
                downloader.download_log_from_blob()

            # Append to local log
            downloader.append_to_log(run_metadata)

            # Upload log to blob if requested
            if args.upload:
                log_path = downloader.output_dir / "download_log.jsonl"
                if log_path.exists():
                    downloader.upload_log_to_blob(log_path)


if __name__ == "__main__":
    main()
