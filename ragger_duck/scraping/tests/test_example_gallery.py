from pathlib import Path

import pytest

from ragger_duck.scraping import GalleryExampleExtractor

EXAMPLES_TEST_FOLDER = Path(__file__).parent / "data" / "gallery" / "examples"
EXAMPLE_NAMES = ["plot_tree_regression", "plot_linear_model_coefficient_interpretation"]
SKLEARN_EXAMPLES_ROOT_URL = "https://scikit-learn.org/stable/auto_examples/"


def test_gallery_example_extractor():
    """Check the behavior of the `GalleryExampleExtractor` class."""
    extractor = GalleryExampleExtractor(chunk_size=1_500, chunk_overlap=10)
    output_extract = extractor.fit_transform(EXAMPLES_TEST_FOLDER)
    expected_sources = [
        SKLEARN_EXAMPLES_ROOT_URL + example_name + ".html"
        for example_name in EXAMPLE_NAMES
    ]
    for output in output_extract:
        assert output["source"] in expected_sources


def test_gallery_example_extractor_no_extraction_error():
    """Check that we raise an error if we don't succeed to extract any text."""
    extractor = GalleryExampleExtractor(chunk_size=1_500, chunk_overlap=10)
    err_msg = "No documentation from the examples was extracted"
    with pytest.raises(ValueError, match=err_msg):
        extractor.fit_transform(Path(__file__).parent / "data" / "gallery" / "empty")
