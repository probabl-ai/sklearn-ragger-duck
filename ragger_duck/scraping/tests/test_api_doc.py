"""Test the utilities for scraping the API documentation."""

import importlib
from pathlib import Path

import pytest

from ragger_duck.scraping import APINumPyDocExtractor
from ragger_duck.scraping._api_doc import _extract_function_doc_numpydoc

API_TEST_FOLDER = Path(__file__).parent / "data" / "api_doc"
NUMPYDOC_SCRAPING_TEST_FOLDER = Path(__file__).parent / "data" / "numpydoc_docscrape"
HTML_TEST_FILES = ["sklearn.base.BaseEstimator.html", "sklearn.base.is_classifier.html"]
SKLEARN_API_URL = "https://scikit-learn.org/stable/modules/generated/"


def test_api_numpydoc_extractor():
    """Check the APINumPyDocExtractor class."""
    extractor = APINumPyDocExtractor()
    output_extractor = extractor.fit_transform(API_TEST_FOLDER)
    possible_source = [SKLEARN_API_URL + html_file for html_file in HTML_TEST_FILES]
    for output in output_extractor:
        assert isinstance(output, dict)
        assert set(output.keys()) == {"source", "text"}
        assert isinstance(output["source"], str)
        assert isinstance(output["text"], str)
        assert output["source"] in possible_source

    assert extractor._get_tags()["stateless"]


def test__extract_function_doc_numpydoc_function():
    """Test the _extract_function_doc_numpydoc function for a scikit-learn function.

    This function is private but it is core of the APINumPyDocExtractor class.
    We test it separately to check all the different cases of the scrapper.

    We test the APINumPyDocExtractor separately to check some higher level
    behaviours.
    """
    html_file = (
        NUMPYDOC_SCRAPING_TEST_FOLDER
        / "sklearn.feature_extraction.image.extract_patches_2d.html"
    )
    full_name = html_file.stem
    module_name, class_or_function_name = full_name.rsplit(".", maxsplit=1)
    url_source = SKLEARN_API_URL + html_file.name
    module = importlib.import_module(module_name)
    class_or_function = getattr(module, class_or_function_name)

    extracted_doc = _extract_function_doc_numpydoc(
        class_or_function, full_name, url_source
    )
    # The following assert is manually checked. If scikit-learn modify the docstring
    # then we might be in trouble.
    assert 5 < len(extracted_doc) < 9
    for doc in extracted_doc:
        assert isinstance(doc, dict)
        assert doc["source"] == url_source


def test__extract_function_doc_numpydoc_class():
    """Test the _extract_function_doc_numpydoc function for a scikit-learn class.

    This function is private but it is core of the APINumPyDocExtractor class.
    We test it separately to check all the different cases of the scrapper.

    We test the APINumPyDocExtractor separately to check some higher level
    behaviours.
    """
    html_file = (
        NUMPYDOC_SCRAPING_TEST_FOLDER / "sklearn.ensemble.RandomForestClassifier.html"
    )
    full_name = html_file.stem
    module_name, class_or_function_name = full_name.rsplit(".", maxsplit=1)
    url_source = SKLEARN_API_URL + html_file.name
    module = importlib.import_module(module_name)
    class_or_function = getattr(module, class_or_function_name)

    extracted_doc = _extract_function_doc_numpydoc(
        class_or_function, full_name, url_source
    )
    # The following assert is manually checked. If scikit-learn modify the docstring
    # then we might be in trouble.
    assert 32 < len(extracted_doc) < 38
    for doc in extracted_doc:
        assert isinstance(doc, dict)
        assert doc["source"] == url_source


def test__extract_function_doc_numpydoc_type_error():
    """Check that we bypass the TypeError when the class or function
    does not have a __doc__ attribute."""

    class Dummy:
        pass

    match = "Fail to parse the docstring"
    with pytest.warns(UserWarning, match=match):
        extracted_doc = _extract_function_doc_numpydoc(Dummy, "dummy", "dummy")
    assert extracted_doc is None
