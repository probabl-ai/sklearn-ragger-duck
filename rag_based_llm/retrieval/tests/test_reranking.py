from pathlib import Path

import pytest
from sentence_transformers import CrossEncoder

from rag_based_llm.embedding import SentenceTransformer
from rag_based_llm.retrieval import BM25Retriever
from rag_based_llm.retrieval import RetrieverReranker
from rag_based_llm.retrieval import SemanticRetriever


@pytest.mark.parametrize(
    "params, n_documents",
    [
        ({"threshold": 0.5}, 2),
        ({"threshold": None}, 14),
        (
            {
                "max_top_k": 2,
            },
            2,
        ),
        ({"threshold": 0.5, "min_top_k": 4}, 4),
    ],
)
def test_retriever_reranker(params, n_documents):
    """Check that the hybrid retriever works as expected."""
    input_texts = [
        {"source": "source 1", "text": "xxx"},
        {"source": "source 2", "text": "yyy"},
        {"source": "source 3", "text": "zzz"},
        {"source": "source 4", "text": "aaa"},
    ]
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
        cross_encoder=cross_encoder,
        semantic_retriever=faiss,
        lexical_retriever=bm25,
        **params,
    )
    context = retriever_reranker.query("xxx")
    assert len(context) == n_documents
