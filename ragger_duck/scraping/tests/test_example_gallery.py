from pathlib import Path

from ragger_duck.scraping import GalleryExampleExtractor

EXAMPLES_TEST_FOLDER = Path(__file__).parent / "data" / "gallery" / "stable"


def test_gallery_example_extractor():
    extractor = GalleryExampleExtractor(chunk_size=1_500, chunk_overlap=10)
    output_extract = extractor.fit_transform(EXAMPLES_TEST_FOLDER)
    for xx in output_extract:
        print("XXXXX")
        print(xx["text"])
