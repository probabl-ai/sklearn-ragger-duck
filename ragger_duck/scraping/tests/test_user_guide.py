from pathlib import Path

import pytest

from ragger_duck.scraping import UserGuideDocExtractor

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
