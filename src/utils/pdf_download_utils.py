#!/usr/bin/env python3
"""
Shared utilities for PDF downloading and validation.

This module provides reusable components for downloading WHO PDFs,
particularly those hosted on iris.who.int which require special handling.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


def create_download_session() -> requests.Session:
    """
    Create a requests session configured for WHO PDF downloads.

    Returns:
        Configured requests.Session with retry strategy and browser-like headers
    """
    session = requests.Session()

    # Add browser-like headers to handle iris.who.int URLs
    session.headers.update(
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
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


def create_chrome_options(headless: bool = True) -> Options:
    """
    Create Chrome options for Selenium WebDriver.

    Args:
        headless: Whether to run Chrome in headless mode (default: True)

    Returns:
        Configured Chrome Options
    """
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

    This is necessary because iris.who.int uses JavaScript-based redirects
    that cannot be handled by simple HTTP requests.

    Args:
        iris_url: The iris.who.int bitstream URL

    Returns:
        Direct download URL or None if resolution fails
    """
    if "iris.who.int/bitstream" not in iris_url:
        return None

    chrome_options = create_chrome_options(headless=True)
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


def validate_pdf_file(file_path: Path, min_size_kb: int = 10) -> bool:
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


def download_pdf_with_retry(
    pdf_url: str,
    local_path: Path,
    session: Optional[requests.Session] = None,
    max_retries: int = 3,
) -> bool:
    """
    Download a single PDF file with corruption detection and retry logic.

    Args:
        pdf_url: URL of the PDF to download
        local_path: Local path where the file should be saved
        session: Optional requests session (will create one if not provided)
        max_retries: Maximum number of retry attempts for corrupted files

    Returns:
        True if download successful, False otherwise
    """
    if session is None:
        session = create_download_session()

    download_url = pdf_url

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                logger.info(f"Retry {attempt}/{max_retries} for {local_path.name}")

            logger.info(f"Downloading {download_url} to {local_path}")

            # Add a small delay to avoid overwhelming the server
            time.sleep(0.5)

            # Follow redirects to get the actual PDF content
            response = session.get(
                download_url, stream=True, timeout=30, allow_redirects=True
            )

            response.raise_for_status()

            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Validate the downloaded file
            if validate_pdf_file(local_path):
                logger.info(f"Successfully downloaded {local_path.name}")
                return True
            else:
                logger.warning(
                    f"Downloaded file {local_path.name} appears corrupted, "
                    f"attempt {attempt + 1}/{max_retries + 1}"
                )
                if local_path.exists():
                    local_path.unlink()  # Remove corrupted file

                # Try iris.who.int resolution as fallback if this is the first attempt
                if attempt == 0 and "iris.who.int/bitstream" in pdf_url:
                    logger.info(
                        f"Attempting iris.who.int resolution fallback for corrupted {local_path.name}"
                    )
                    resolved_url = resolve_iris_url_with_selenium(pdf_url)

                    if resolved_url:
                        logger.info(
                            f"Got resolved URL, retrying download: {resolved_url}"
                        )
                        download_url = resolved_url  # Use resolved URL for remaining attempts
                        time.sleep(1)
                        continue
                    else:
                        logger.warning(
                            f"iris.who.int resolution failed for {local_path.name}"
                        )

                if attempt < max_retries:
                    # Wait before retry
                    time.sleep(2.0 * (attempt + 1))
                    continue
                else:
                    logger.error(
                        f"Failed to download valid file after "
                        f"{max_retries + 1} attempts"
                    )
                    return False

        except requests.RequestException as e:
            logger.error(f"Failed to download {pdf_url}: {e}")

            # Try iris.who.int resolution as fallback if this is the first attempt
            if attempt == 0 and "iris.who.int/bitstream" in pdf_url:
                logger.info(
                    f"Attempting iris.who.int resolution fallback for {local_path.name}"
                )
                resolved_url = resolve_iris_url_with_selenium(pdf_url)

                if resolved_url:
                    logger.info(
                        f"Got resolved URL, retrying download: {resolved_url}"
                    )
                    download_url = resolved_url  # Use resolved URL for remaining attempts
                    time.sleep(1)
                    continue
                else:
                    logger.warning(f"iris.who.int resolution failed for {local_path.name}")

            if attempt < max_retries:
                time.sleep(2.0 * (attempt + 1))
                continue
            else:
                logger.error(f"Failed after {max_retries + 1} attempts")
                return False

    return False
