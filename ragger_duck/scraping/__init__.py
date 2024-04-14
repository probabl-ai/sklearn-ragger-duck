"""
The module :mod:`ragger_duck.scraping` contains functions to scrape the documentation
website of scikit-learn.
"""

from ._api_doc import APINumPyDocExtractor
from ._example_gallery import GalleryExampleExtractor
from ._user_guide import UserGuideDocExtractor

__all__ = [
    "APINumPyDocExtractor",
    "GalleryExampleExtractor",
    "UserGuideDocExtractor",
]
