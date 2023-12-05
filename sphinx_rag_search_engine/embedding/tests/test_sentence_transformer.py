from pathlib import Path

from sphinx_rag_search_engine.embedding import SentenceTransformer


def test_sentence_transformer():
    """Check that the SentenceTransformer wrapper works as expected."""
    cache_folder_path = Path(__file__).parent / "data"
    model_name_or_path = "sentence-transformers/paraphrase-albert-small-v2"

    embedder = SentenceTransformer(
        model_name_or_path=model_name_or_path,
        cache_folder=str(cache_folder_path),
        show_progress_bar=False,
    )

    input_texts = [
        {
            "source": "source 1",
            "text": "hello world",
        },
        {
            "source": "source 2",
            "text": "hello world",
        },
    ]
    text_embedded = embedder.fit_transform(input_texts)
    assert text_embedded.shape == (len(input_texts, 768))


# TODO: add test for checking when the input of transform is a string or a
# list of strings.