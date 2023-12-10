import pytest
from sklearn.feature_extraction.text import CountVectorizer

from rag_based_llm.retrieval import BM25Retriever


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
@pytest.mark.parametrize("count_vectorizer", [None, CountVectorizer()])
def test_lexical_retriever(input_texts, output, count_vectorizer):
    """Check that the SemanticRetriever wrapper works as expected"""
    bm25 = BM25Retriever(count_vectorizer=count_vectorizer, n_neighbors=1).fit(
        input_texts
    )
    assert bm25.k_neighbors("xxx") == output


def test_lexical_retriever_error():
    """Check that we raise an error when the input is not a string at inference time."""
    input_texts = [{"source": "source 1", "text": "xxx"}]
    bm25 = BM25Retriever(n_neighbors=1).fit(input_texts)
    with pytest.raises(TypeError):
        bm25.k_neighbors(["xxx"])
