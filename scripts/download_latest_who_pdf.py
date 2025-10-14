#!/usr/bin/env python3
"""
Download the latest WHO cholera PDF from AFRO website.

This script scrapes the AFRO WHO outbreaks page to find and download the latest
(or a specific) weekly cholera bulletin PDF. It can optionally upload the PDF
to blob storage for processing.

Usage:
    # Download latest week
    python scripts/download_latest_who_pdf.py

    # Download specific week
    python scripts/download_latest_who_pdf.py --week 37

    # List available weeks without downloading
    python scripts/download_latest_who_pdf.py --list

    # Download and upload to blob
    python scripts/download_latest_who_pdf.py --upload
"""

import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import ocha_stratus as stratus
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import WebDriverException

from src.config import Config
from src.utils.pdf_download_utils import (
    create_chrome_options,
    create_download_session,
    download_pdf_with_retry,
)

logger = logging.getLogger(__name__)


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


class LatestWHOPDFDownloader:
    """Downloads latest weekly bulletins from AFRO WHO website."""

    AFRO_OUTBREAKS_URL = (
        "https://www.afro.who.int/health-topics/disease-outbreaks/"
        "outbreaks-and-other-emergencies-updates"
    )

    def __init__(
        self,
        stage: str = "dev",
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize the downloader.

        Args:
            stage: Environment stage (dev/staging/prod)
            output_dir: Custom output directory (defaults to Config.HISTORICAL_PDFS_DIR)
        """
        self.stage = stage
        self.output_dir = output_dir or Config.HISTORICAL_PDFS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session = create_download_session()
        self.blob_container = Config.BLOB_CONTAINER
        self.blob_proj_dir = Config.BLOB_PROJ_DIR

        logger.info(f"Output directory: {self.output_dir}")

    def scrape_weekly_bulletins(self) -> List[BulletinMetadata]:
        """
        Scrape the AFRO WHO page for weekly bulletin links.

        Returns:
            List of bulletin metadata, sorted by week number (descending)
        """
        logger.info(f"Scraping weekly bulletins from {self.AFRO_OUTBREAKS_URL}")

        chrome_options = create_chrome_options(headless=True)
        driver = None
        bulletins = []

        try:
            driver = webdriver.Chrome(options=chrome_options)
            driver.get(self.AFRO_OUTBREAKS_URL)

            # Wait for page to load
            time.sleep(3)

            # Get page source and parse with BeautifulSoup
            soup = BeautifulSoup(driver.page_source, "html.parser")

            # Find all links that match the weekly bulletin pattern
            # Pattern: "Week XX: DD to DD Month YYYY"
            week_pattern = re.compile(
                r"Week\s+(\d+):\s+([\d\s]+to[\d\s]+\w+\s+\d{4})", re.IGNORECASE
            )

            # Search for links containing week information
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

                    # Try to find publication date (might be in parent elements)
                    parent = link.find_parent()
                    if parent:
                        # Look for date patterns in parent text
                        date_pattern = re.compile(
                            r"(\d{1,2}\s+\w+\s+\d{4})|(\w+\s+\d{1,2},\s+\d{4})"
                        )
                        parent_text = parent.get_text()
                        date_match = date_pattern.search(parent_text)
                        if date_match:
                            bulletin.publication_date = date_match.group(0)

                    bulletins.append(bulletin)
                    logger.debug(
                        f"Found bulletin: Week {week_num}, Year {year}, "
                        f"URL: {link['href'][:50]}..."
                    )

            # Sort by year (descending) then week (descending)
            bulletins.sort(key=lambda x: (x.year, x.week), reverse=True)

            logger.info(f"Found {len(bulletins)} weekly bulletins")
            return bulletins

        except WebDriverException as e:
            logger.error(f"WebDriver error while scraping: {e}")
            return []
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
        """
        Get the latest weekly bulletin.

        Returns:
            Metadata for the latest bulletin, or None if not found
        """
        bulletins = self.scrape_weekly_bulletins()

        if not bulletins:
            logger.error("No bulletins found")
            return None

        latest = bulletins[0]
        logger.info(
            f"Latest bulletin: Week {latest.week}, Year {latest.year}, "
            f"Date range: {latest.date_range}"
        )
        return latest

    def get_bulletin_by_week(self, week_num: int) -> Optional[BulletinMetadata]:
        """
        Get a specific weekly bulletin by week number.

        Args:
            week_num: Week number to find

        Returns:
            Metadata for the requested bulletin, or None if not found
        """
        bulletins = self.scrape_weekly_bulletins()

        # Find the bulletin with matching week number (prefer current year)
        current_year = datetime.now().year
        matching = [b for b in bulletins if b.week == week_num]

        if not matching:
            logger.error(f"No bulletin found for week {week_num}")
            return None

        # Prefer bulletins from current year, otherwise take the most recent
        current_year_matches = [b for b in matching if b.year == current_year]
        bulletin = current_year_matches[0] if current_year_matches else matching[0]

        logger.info(
            f"Found bulletin: Week {bulletin.week}, Year {bulletin.year}, "
            f"Date range: {bulletin.date_range}"
        )
        return bulletin

    def download_bulletin(
        self, bulletin: BulletinMetadata, upload_to_blob: bool = False
    ) -> Optional[BulletinMetadata]:
        """
        Download a bulletin PDF and optionally upload to blob storage.

        Args:
            bulletin: Bulletin metadata
            upload_to_blob: Whether to upload to blob storage

        Returns:
            Updated bulletin metadata with downloaded_path, or None if failed
        """
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
            except Exception as e:
                logger.error(f"Failed to upload to blob: {e}")
                # Don't fail the whole operation if blob upload fails

        return bulletin

    def upload_to_blob(self, local_path: Path, blob_name: Optional[str] = None) -> None:
        """
        Upload a PDF file to blob storage using stratus.

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

        logger.info(f"Uploading {local_path.name} to {blob_path}")

        # Use stratus.upload_blob_data for simple upload
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
            print(
                f"{bulletin.week:<6} {bulletin.year:<6} "
                f"{bulletin.date_range:<35} {url_short}"
            )


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
        help="Custom output directory (default: from config)",
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
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    downloader = LatestWHOPDFDownloader(
        stage=Config.STAGE,
        output_dir=args.output_dir,
    )

    if args.list:
        downloader.list_available_bulletins()
        return

    # Get the bulletin
    if args.week:
        bulletin = downloader.get_bulletin_by_week(args.week)
    else:
        bulletin = downloader.get_latest_bulletin()

    if not bulletin:
        logger.error("No bulletin found")
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
        return

    print(f"\nSuccessfully downloaded to: {result.downloaded_path}")

    # Save metadata if requested
    if args.save_metadata:
        with open(args.save_metadata, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"Metadata saved to: {args.save_metadata}")


if __name__ == "__main__":
    main()
