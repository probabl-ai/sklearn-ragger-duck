from pathlib import Path

import pytest
from sentence_transformers import CrossEncoder

from ragger_duck.embedding import SentenceTransformer
from ragger_duck.retrieval import BM25Retriever, RetrieverReranker, SemanticRetriever


@pytest.mark.parametrize(
    "params, n_documents",
    [
        ({"threshold": 0.5}, 2),
        ({"threshold": None}, 8),
        (
            {
                "max_top_k": 2,
            },
            2,
        ),
        ({"threshold": 0.5, "min_top_k": 4}, 4),
    ],
)
@pytest.mark.parametrize(
    "input_texts",
    [
        [
            {"source": "source 1", "text": "xxx"},
            {"source": "source 2", "text": "yyy"},
            {"source": "source 3", "text": "zzz"},
            {"source": "source 4", "text": "aaa"},
        ],
        ["xxx", "yyy", "zzz", "aaa"],
    ],
)
def test_retriever_reranker(input_texts, params, n_documents):
    """Check that the hybrid retriever works as expected."""
    bm25 = BM25Retriever(top_k=10).fit(input_texts)

    cache_folder_path = (
        Path(__file__).parent.parent.parent / "embedding" / "tests" / "data"
    )
    model_name_or_path = "sentence-transformers/paraphrase-albert-small-v2"
    embedder = SentenceTransformer(
        model_name_or_path=model_name_or_path,
        cache_folder=str(cache_folder_path),
        show_progress_bar=False,
    )
    faiss = SemanticRetriever(embedding=embedder, top_k=10).fit(input_texts)

    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder = CrossEncoder(model_name=model_name)
    retriever_reranker = RetrieverReranker(
        retrievers=[bm25, faiss],
        cross_encoder=cross_encoder,
        drop_duplicates=False,
        **params,
    )
    retriever_reranker.fit()  # just for parameter validation
    context = retriever_reranker.query("xxx")
    assert len(context) == n_documents


@pytest.mark.parametrize(
    "input_texts",
    [
        [
            {"source": "source 1", "text": "xxx"},
            {"source": "source 2", "text": "yyy"},
            {"source": "source 3", "text": "zzz"},
            {"source": "source 4", "text": "aaa"},
        ],
        ["xxx", "yyy", "zzz", "aaa"],
    ],
)
@pytest.mark.parametrize(
    "drop_duplicates, n_retrieved_documents", [(True, 4), (False, 8)]
)
def test_retriever_reranker_drop_duplicate(
    input_texts, drop_duplicates, n_retrieved_documents
):
    """Check the behaviour of the drop_duplicates parameter."""
    bm25 = BM25Retriever(top_k=10).fit(input_texts)
    cache_folder_path = (
        Path(__file__).parent.parent.parent / "embedding" / "tests" / "data"
    )
    model_name_or_path = "sentence-transformers/paraphrase-albert-small-v2"
    embedder = SentenceTransformer(
        model_name_or_path=model_name_or_path,
        cache_folder=str(cache_folder_path),
        show_progress_bar=False,
    )
    faiss = SemanticRetriever(embedding=embedder, top_k=10).fit(input_texts)

    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder = CrossEncoder(model_name=model_name)
    retriever_reranker = RetrieverReranker(
        retrievers=[bm25, faiss],
        cross_encoder=cross_encoder,
        max_top_k=None,  # we don't limit the number of retrieved documents
        drop_duplicates=drop_duplicates,
    )
    retriever_reranker.fit()
    context = retriever_reranker.query("xxx")
    assert len(context) == n_retrieved_documents


def test_retriever_reranker_no_query_results():
    """Check the results when no query results are found."""
    input_text = [
        {"source": "source 1", "text": "xxx"},
        {"source": "source 2", "text": "yyy"},
        {"source": "source 3", "text": "zzz"},
        {"source": "source 4", "text": "aaa"},
    ]
    bm25 = BM25Retriever(top_k=10).fit(input_text)
    cache_folder_path = (
        Path(__file__).parent.parent.parent / "embedding" / "tests" / "data"
    )
    model_name_or_path = "sentence-transformers/paraphrase-albert-small-v2"
    embedder = SentenceTransformer(
        model_name_or_path=model_name_or_path,
        cache_folder=str(cache_folder_path),
        show_progress_bar=False,
    )
    faiss = SemanticRetriever(embedding=embedder, top_k=10).fit(input_text)

    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder = CrossEncoder(model_name=model_name)
    # set top-k to the minimum and a high threshold to avoid any match
    retriever_reranker = RetrieverReranker(
        retrievers=[bm25, faiss],
        cross_encoder=cross_encoder,
        min_top_k=None,
        threshold=5.0,
    )
    retriever_reranker.fit()
    assert not len(retriever_reranker.query("no match"))


def test_retriever_reranker_tags():
    """Check the stateless parameter of the reranker retriever."""
    bm25 = BM25Retriever(top_k=10)
    cache_folder_path = (
        Path(__file__).parent.parent.parent / "embedding" / "tests" / "data"
    )
    model_name_or_path = "sentence-transformers/paraphrase-albert-small-v2"
    embedder = SentenceTransformer(
        model_name_or_path=model_name_or_path,
        cache_folder=str(cache_folder_path),
        show_progress_bar=False,
    )
    faiss = SemanticRetriever(embedding=embedder, top_k=10)

    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder = CrossEncoder(model_name=model_name)
    retriever_reranker = RetrieverReranker(
        retrievers=[bm25, faiss], cross_encoder=cross_encoder
    )
    assert retriever_reranker._get_tags()["stateless"]
