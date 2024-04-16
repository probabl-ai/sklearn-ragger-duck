from pathlib import Path

import pytest

from ragger_duck.scraping import UserGuideDocExtractor
from ragger_duck.scraping._user_guide import (
    _extract_user_guide_doc,
    extract_user_guide_doc_from_single_file,
)

USER_GUIDE_TEST_FOLDER = Path(__file__).parent / "data" / "user_guide_doc"
HTML_TEST_FILES = ["calibration.html", "clustering.html"]
SKLEARN_API_URL = "https://scikit-learn.org/stable/modules/"


@pytest.mark.parametrize("n_jobs", [None, 1, 2])
@pytest.mark.parametrize("chunk_size", [None, 300])
def test_user_guide_doc_extractor(chunk_size, n_jobs):
    """Check the behavior of the `UserGuideDocExtractor` class."""
    extractor = UserGuideDocExtractor(chunk_size=chunk_size, n_jobs=n_jobs)
    output_extract = extractor.fit_transform(USER_GUIDE_TEST_FOLDER)
    possible_source = [SKLEARN_API_URL + html_file for html_file in HTML_TEST_FILES]
    for chunk in output_extract:
        assert isinstance(chunk, dict)
        assert isinstance(chunk["source"], str)
        assert isinstance(chunk["text"], str)
        if chunk_size is not None:
            assert len(chunk["text"]) <= chunk_size
        assert chunk["source"] in possible_source

    tags = extractor._get_tags()
    assert tags["X_types"] == ["string"]
    assert tags["stateless"] is True


def test_user_guide_doc_extractor_black_list():
    """Check the behavior of the `UserGuideDocExtractor` class when using a black list
    of files."""
    extractor = UserGuideDocExtractor(
        folders_to_exclude=["clustering"], chunk_size=None, n_jobs=1
    )
    output_extract = extractor.fit_transform(USER_GUIDE_TEST_FOLDER)
    assert len(output_extract) == 1
    assert "clustering" not in output_extract[0]["source"]


def test_extract_user_guide_doc_from_single_file_error():
    """Check that we raise the proper error message when the input is not as
    expected."""
    err_msg = "The User Guide HTML file should be a pathlib.Path object."
    with pytest.raises(ValueError, match=err_msg):
        extract_user_guide_doc_from_single_file("not_a_pathlib.Path")

    err_msg = "is not an HTML file. Please provide an HTML file."
    with pytest.raises(ValueError, match=err_msg):
        extract_user_guide_doc_from_single_file(Path(__file__))


def test__extract_user_guide_doc_error():
    """Check that we raise the proper error message for the `_extract_user_guide_doc`
    function."""
    err_msg = "should be a pathlib.Path object."
    with pytest.raises(ValueError, match=err_msg):
        _extract_user_guide_doc("not_a_pathlib.Path", [])


def test_user_guide_extractor_no_extraction_error():
    """Check that we raise an error if we don't succeed to extract any text."""
    extractor = UserGuideDocExtractor(chunk_size=None, n_jobs=1)
    err_msg = "No User Guide documentation was extracted"
    with pytest.raises(ValueError, match=err_msg):
        extractor.fit_transform(Path(__file__).parent / "data" / "empty")
