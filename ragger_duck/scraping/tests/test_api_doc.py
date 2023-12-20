"""Test the utilities for scraping the API documentation."""

import importlib
from pathlib import Path

import pytest

from ragger_duck.scraping import (
    APIDocExtractor,
    APINumPyDocExtractor,
    extract_api_doc,
    extract_api_doc_from_single_file,
)
from ragger_duck.scraping._api_doc import _extract_function_doc_numpydoc

API_TEST_FOLDER = Path(__file__).parent / "data" / "api_doc"
NUMPYDOC_SCRAPING_TEST_FOLDER = Path(__file__).parent / "data" / "numpydoc_docscrape"
HTML_TEST_FILES = ["sklearn.base.BaseEstimator.html", "sklearn.base.is_classifier.html"]
SKLEARN_API_URL = "https://scikit-learn.org/stable/modules/generated/"


def test_extract_api_doc_from_single_file_not_html():
    """Check that we raise an error if the provided file is not an HTML file."""
    path_file = Path(__file__)
    err_msg = "is not an HTML file"
    with pytest.raises(ValueError, match=err_msg):
        extract_api_doc_from_single_file(path_file)


@pytest.mark.parametrize(
    "extract_function", [extract_api_doc_from_single_file, extract_api_doc]
)
def test_input_not_path_from_pathlib(extract_function):
    """Check that we raise an error if the input is not a pathlib.Path."""
    err_msg = "should be a pathlib.Path"
    with pytest.raises(ValueError, match=err_msg):
        extract_function("not a pathlib.Path")


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
    output = extract_api_doc(API_TEST_FOLDER, n_jobs=n_jobs)
    assert isinstance(output, list)

    assert len(output) == 2
    assert all([isinstance(elt, dict) for elt in output])
    sources = sorted([file["source"] for file in output])
    expected_sources = sorted(
        [SKLEARN_API_URL + html_file for html_file in HTML_TEST_FILES]
    )
    assert sources == expected_sources


@pytest.mark.parametrize("n_jobs", [None, 1, 2])
@pytest.mark.parametrize("chunk_size", [20, None])
def test_api_doc_extractor(n_jobs, chunk_size):
    """Check the APIDocExtractor class."""
    extractor = APIDocExtractor(chunk_size=chunk_size, chunk_overlap=0, n_jobs=n_jobs)
    output_extractor = extractor.fit_transform(API_TEST_FOLDER)
    possible_source = [SKLEARN_API_URL + html_file for html_file in HTML_TEST_FILES]
    for output in output_extractor:
        assert isinstance(output, dict)
        assert set(output.keys()) == {"source", "text"}
        assert isinstance(output["source"], str)
        assert isinstance(output["text"], str)
        if chunk_size is not None:
            assert len(output["text"]) <= chunk_size
        assert output["source"] in possible_source

    assert extractor._get_tags()["stateless"]


def test_api_doc_extractor_error_empty():
    """Check that we raise an error if the folder does not contain any HTML file."""
    path_folder = Path(__file__).parent
    err_msg = "No API documentation was extracted. Please check the input folder."
    with pytest.raises(ValueError, match=err_msg):
        APIDocExtractor().fit_transform(path_folder)


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
    assert len(extracted_doc) == 8
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
    assert len(extracted_doc) == 35
    for doc in extracted_doc:
        assert isinstance(doc, dict)
        assert doc["source"] == url_source
