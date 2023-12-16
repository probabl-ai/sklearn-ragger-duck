"""
The module :mod:`ragger_duck.scraping` contains functions to scrape the documentation
website of scikit-learn.
"""

from ._api_doc import (
    APIDocExtractor,
    extract_api_doc,
    extract_api_doc_from_single_file,
)

__all__ = [
    "extract_api_doc",
    "extract_api_doc_from_single_file",
    "APIDocExtractor",
]
