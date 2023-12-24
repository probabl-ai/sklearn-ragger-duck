"""
The module :mod:`ragger_duck.scraping` contains functions to scrape the documentation
website of scikit-learn.
"""

from ._api_doc import (
    APIDocExtractor,
    APINumPyDocExtractor,
    extract_api_doc,
    extract_api_doc_from_single_file,
)
from ._user_guide import (
    UserGuideDocExtractor,
)

__all__ = [
    "extract_api_doc",
    "extract_api_doc_from_single_file",
    "APIDocExtractor",
    "APINumPyDocExtractor",
    "UserGuideDocExtractor",
]
