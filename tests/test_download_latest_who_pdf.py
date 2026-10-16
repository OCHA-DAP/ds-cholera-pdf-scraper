"""
Tests for parsing the AFRO weekly bulletin listing and publication pages.

Fixtures mirror the card markup on
https://www.afro.who.int/health-topics/disease-outbreaks/outbreaks-and-other-emergencies-updates
as of October 2026.
"""

import pytest

from scripts.download_latest_who_pdf import parse_listing_page, parse_publication_page

CARD_WITH_DOWNLOAD = """
<div class="pub-list-body">
  <div class="pub-card-meta"><small class="pub-card-date">5 October 2026</small></div>
  <h3 class="pub-list-title">
    <a href="/countries/senegal/publication/weekly-bulletin-week-40">
      Weekly Bulletin on Outbreak and other Emergencies: Week 40: 28 September to 05 October 2026
    </a>
  </h3>
  <div class="pub-card-actions">
    <a class="pub-btn pub-btn-download" href="https://iris.who.int/bitstreams/abc/download">Download</a>
  </div>
</div>
"""

CARD_WITHOUT_DOWNLOAD = """
<div class="pub-list-body">
  <h3 class="pub-list-title">
    <a href="/publications/weekly-bulletin-week-10">
      Weekly Bulletin on Outbreak and other Emergencies: Week 10: 02 to 08 March 2026
    </a>
  </h3>
</div>
"""

CARD_NOT_A_WEEKLY_BULLETIN = """
<div class="pub-list-body">
  <h3 class="pub-list-title"><a href="/publications/annual-report">Annual report 2025</a></h3>
</div>
"""


class TestParseListingPage:
    def test_card_with_download_button(self):
        [bulletin] = parse_listing_page(CARD_WITH_DOWNLOAD)

        assert (bulletin.week, bulletin.year) == (40, 2026)
        assert bulletin.date_range == "28 September to 05 October 2026"
        assert bulletin.pdf_url == "https://iris.who.int/bitstreams/abc/download"
        assert bulletin.page_url == (
            "https://www.afro.who.int/countries/senegal/publication/weekly-bulletin-week-40"
        )
        assert bulletin.publication_date == "5 October 2026"

    def test_card_without_download_button_is_left_unresolved(self):
        [bulletin] = parse_listing_page(CARD_WITHOUT_DOWNLOAD)

        assert (bulletin.week, bulletin.year) == (10, 2026)
        assert bulletin.pdf_url is None
        assert bulletin.page_url == (
            "https://www.afro.who.int/publications/weekly-bulletin-week-10"
        )

    def test_non_weekly_cards_are_skipped(self):
        html = CARD_NOT_A_WEEKLY_BULLETIN + CARD_WITH_DOWNLOAD
        assert [b.week for b in parse_listing_page(html)] == [40]


class TestParsePublicationPage:
    def test_returns_iris_link(self):
        html = """
        <a class="btn-pub-iris" href="https://iris.who.int/bitstreams/xyz/download">
          Download from IRIS
        </a>
        """
        assert parse_publication_page(html, "page") == (
            "https://iris.who.int/bitstreams/xyz/download"
        )

    def test_missing_iris_link_raises(self):
        with pytest.raises(RuntimeError, match="found 0"):
            parse_publication_page("<p>no link here</p>", "page")
