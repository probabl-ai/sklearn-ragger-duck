from pathlib import Path

from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ragger_duck.scraping._shared import (
    _chunk_document,
    _extract_text_from_section,
)

TEST_HMTL_FILE = Path(__file__).parent / "data" / "user_guide_doc" / "calibration.html"


def test_extract_text_from_section():
    """Check the behavior of the `_extract_text_from_section` function."""
    with open(TEST_HMTL_FILE, "r") as file:
        soup = BeautifulSoup(file, "html.parser")
    sections = soup.section
    for section in sections:
        text = _extract_text_from_section(section)
        assert text is None or isinstance(text, str)

    # FIXME: write more tests to check the exact behavior depending on tags.


def test_chunk_document():
    """Check the behavior of the `_chunk_document` function."""
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=40,
        chunk_overlap=30,
        length_function=len,
    )
    texts = []
    with open(TEST_HMTL_FILE, "r") as file:
        soup = BeautifulSoup(file, "html.parser")
        for section in soup.section:
            text = _extract_text_from_section(section)
            if text is not None and len(text) > 20:
                texts.append(
                    {
                        "text": text,
                        "source": "https://some_source.com",
                    }
                )
    chunks = _chunk_document(text_splitter, texts[1])
    for chunk in chunks:
        assert isinstance(chunk, dict)
        assert len(chunk["text"]) <= 40
        assert chunk["source"] == "https://some_source.com"
