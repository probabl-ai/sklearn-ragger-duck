"""Test the utilities for scraping the API documentation."""
from pathlib import Path

import pytest

from sphinx_rag_search_engine.scraping import extract_api_doc_from_single_file


def test_extract_api_doc_from_single_file_not_html():
    """Check that we raise an error if the provided file is not an HTML file."""
    path_file = Path(__file__)
    err_msg = "is not an HTML file"
    with pytest.raises(ValueError, match=err_msg):
        extract_api_doc_from_single_file(path_file)
