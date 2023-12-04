"""Test the utilities for scraping the API documentation."""
from pathlib import Path
from types import GeneratorType

import pytest

from sphinx_rag_search_engine.scraping import (
    extract_api_doc_from_single_file,
    extract_api_doc,
    APIDocExtractor,
)


API_TEST_FOLDER = Path(__file__).parent / "data" / "api_doc"
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


@pytest.mark.parametrize("return_as", ["generator", "generator_unordered", "list"])
@pytest.mark.parametrize("n_jobs", [None, 1, 2])
def test_extract_api_doc(return_as, n_jobs):
    """Checking the the behaviour of the `extract_api_doc` function."""
    output = extract_api_doc(API_TEST_FOLDER, return_as=return_as, n_jobs=n_jobs)
    if return_as == "list":
        assert isinstance(output, list)
    else:
        assert isinstance(output, GeneratorType)

    scraped_files = list(output)
    assert len(scraped_files) == 2
    assert all([isinstance(elt, dict) for elt in scraped_files])
    sources = sorted([file["source"] for file in scraped_files])
    expected_sources = sorted(
        [SKLEARN_API_URL + html_file for html_file in HTML_TEST_FILES]
    )
    assert sources == expected_sources


@pytest.mark.parametrize("output_type", ["generator", "generator_unordered", "list"])
@pytest.mark.parametrize("n_jobs", [None, 1, 2])
def test_api_doc_extractor(output_type, n_jobs):
    """Check the APIDocExtractor class."""
    extractor = APIDocExtractor(output_type=output_type, n_jobs=n_jobs)
    output_extractor = extractor.fit_transform(API_TEST_FOLDER)
    output_function = extract_api_doc(
        API_TEST_FOLDER, return_as=output_type, n_jobs=n_jobs
    )

    if output_type == "list":
        assert isinstance(output_extractor, list)
        assert isinstance(output_function, list)
    else:
        assert isinstance(output_extractor, GeneratorType)
        assert isinstance(output_function, GeneratorType)

    output_extractor = list(output_extractor)
    output_function = list(output_function)

    sources_extractor = sorted([file["source"] for file in output_extractor])
    sources_function = sorted([file["source"] for file in output_function])
    assert sources_extractor == sources_function

    texts_extractor = sorted([file["text"] for file in output_extractor])
    texts_function = sorted([file["text"] for file in output_function])
    assert texts_extractor == texts_function
