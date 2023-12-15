import pytest
from rag.retrieval import BM25Retriever
from sklearn.feature_extraction.text import CountVectorizer


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
    bm25 = BM25Retriever(count_vectorizer=count_vectorizer, top_k=1).fit(input_texts)
    assert bm25.query("xxx") == output


def test_lexical_retriever_error():
    """Check that we raise an error when the input is not a string at inference time."""
    input_texts = [{"source": "source 1", "text": "xxx"}]
    bm25 = BM25Retriever(top_k=1).fit(input_texts)
    with pytest.raises(TypeError):
        bm25.query(["xxx"])
