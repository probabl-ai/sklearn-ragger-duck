from pathlib import Path

from bs4 import BeautifulSoup

from ragger_duck.scraping._shared import _extract_text_from_section

TEST_HMTL_FILE = Path(__file__).parent / "data" / "user_guide_doc" / "calibration.html"


def test_extract_text_from_section():
    """Check the behavior of the `_extract_text_from_section` function."""
    with open(TEST_HMTL_FILE, "r") as file:
        soup = BeautifulSoup(file, "html.parser")
    sections = soup.section
    for section in sections:
        text = _extract_text_from_section(section)
        assert text is None or isinstance(text, str)
