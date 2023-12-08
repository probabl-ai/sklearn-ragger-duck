"""The module :mod:`rag_based_llm.scraping` contains functions to scrape
the documentation website of scikit-learn.
"""

from ._api_doc import (
    extract_api_doc,
    extract_api_doc_from_single_file,
    APIDocExtractor,
)

__all__ = [
    "extract_api_doc",
    "extract_api_doc_from_single_file",
    "APIDocExtractor",
]