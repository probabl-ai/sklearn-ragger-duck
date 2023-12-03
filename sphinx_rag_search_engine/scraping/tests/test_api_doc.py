"""Test the utilities for scraping the API documentation."""
from pathlib import Path
from types import GeneratorType

import pytest

from sphinx_rag_search_engine.scraping import (
    extract_api_doc_from_single_file,
    extract_api_doc,
)


API_TEST_FOLDER = Path(__file__).parent / "data" / "api_doc"
HTML_TEST_FILES = [
    "sklearn.base.BaseEstimator.html", "sklearn.base.is_classifier.html"
]
SKLEARN_API_URL = "https://scikit-learn.org/stable/modules/generated/"


def test_extract_api_doc_from_single_file_not_html():
    """Check that we raise an error if the provided file is not an HTML file."""
    path_file = Path(__file__)
    err_msg = "is not an HTML file"
    with pytest.raises(ValueError, match=err_msg):
        extract_api_doc_from_single_file(path_file)


@pytest.mark.parametrize("html_file", HTML_TEST_FILES)
def test_extract_api_doc_from_single_file(html_file):
    """Check that we have some meaningful scraping results when parsing
    a single HTML file.
    """
    path_file = API_TEST_FOLDER / html_file
    text = extract_api_doc_from_single_file(path_file)
    assert isinstance(text, dict)
    assert set(text.keys()) == {"source", "text"}
    assert text["source"] == SKLEARN_API_URL + html_file
    expected_strings = ["Parameters :", "Returns :"]
    for string in expected_strings:
        assert string in text["text"]


@pytest.mark.parametrize("n_jobs", [None, 1, 2])
def test_extract_api_doc(n_jobs):
    """Checking the the behaviour of the `extract_api_doc` function."""
    gen_scraped_files = extract_api_doc(API_TEST_FOLDER, n_jobs=n_jobs)
    assert isinstance(gen_scraped_files, GeneratorType)

    scraped_files = list(gen_scraped_files)
    assert len(scraped_files) == 2
    sources = sorted([file["source"] for file in scraped_files])
    expected_sources = sorted([
        SKLEARN_API_URL + html_file for html_file in HTML_TEST_FILES
    ])
    assert sources == expected_sources
