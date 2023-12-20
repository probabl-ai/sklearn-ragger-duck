from pathlib import Path

import pytest

from ragger_duck.embedding import SentenceTransformer
from ragger_duck.retrieval import SemanticRetriever


@pytest.mark.parametrize(
    "input_texts, output",
    [
        (
            [
                {"source": "source 1", "text": "xxx"},
                {"source": "source 2", "text": "yyy"},
            ],
            [{"source": "source 1", "text": "xxx"}],
        ),
        (["xxx", "yyy"], ["xxx"]),
    ],
)
def test_semantic_retriever(input_texts, output):
    """Check that the SemanticRetriever wrapper works as expected"""
    cache_folder_path = (
        Path(__file__).parent.parent.parent / "embedding" / "tests" / "data"
    )
    model_name_or_path = "sentence-transformers/paraphrase-albert-small-v2"

    embedder = SentenceTransformer(
        model_name_or_path=model_name_or_path,
        cache_folder=str(cache_folder_path),
        show_progress_bar=False,
    )

    faiss = SemanticRetriever(embedding=embedder, top_k=1).fit(input_texts)
    assert faiss.query("xx") == output


def test_semantic_retriever_error():
    """Check that we raise an error when the input is not a string at inference time."""
    cache_folder_path = (
        Path(__file__).parent.parent.parent / "embedding" / "tests" / "data"
    )
    model_name_or_path = "sentence-transformers/paraphrase-albert-small-v2"

    embedder = SentenceTransformer(
        model_name_or_path=model_name_or_path,
        cache_folder=str(cache_folder_path),
        show_progress_bar=False,
    )

    input_texts = [{"source": "source 1", "text": "xxx"}]
    faiss = SemanticRetriever(embedding=embedder, top_k=1).fit(input_texts)
    with pytest.raises(TypeError):
        faiss.query(["xxxx"])


@pytest.mark.parametrize(
    "input_texts",
    [
        [
            {"source": "source 1", "text": "xxx"},
            {"source": "source 2", "text": "yyy"},
        ],
        ["xxx", "yyy"],
    ],
)
def test_semantic_retriever_max_documents_at_fit(input_texts):
    """Check that return at max the number of documents in the training set."""
    cache_folder_path = (
        Path(__file__).parent.parent.parent / "embedding" / "tests" / "data"
    )
    model_name_or_path = "sentence-transformers/paraphrase-albert-small-v2"

    embedder = SentenceTransformer(
        model_name_or_path=model_name_or_path,
        cache_folder=str(cache_folder_path),
        show_progress_bar=False,
    )

    faiss = SemanticRetriever(embedding=embedder, top_k=20).fit(input_texts)
    assert len(faiss.query("xx")) == len(input_texts)
