from collections.abc import Iterable
from pathlib import Path

import pytest

from rag_based_llm.embedding import SentenceTransformer


@pytest.mark.parametrize(
    "input_texts", [
        [
            {"source": "source 1", "text": "hello world"},
            {"source": "source 2", "text": "hello world"},
        ],
        ["hello world", "hello world"],
        "hello world",
    ],
)
def test_sentence_transformer(input_texts):
    """Check that the SentenceTransformer wrapper works as expected when the input."""
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
    n_sentences = len(input_texts) if isinstance(input_texts, Iterable) else 1
    text_embedded = embedder.fit_transform(input_texts)
    assert text_embedded.shape == (n_sentences, 768)
