from pathlib import Path

from sphinx_rag_search_engine.embedding import SentenceTransformer
from sphinx_rag_search_engine.retrieval import FAISS


def test_xxx():
    cache_folder_path = (
        Path(__file__).parent.parent.parent / "embedding" / "tests" / "data"
    )
    model_name_or_path = "sentence-transformers/paraphrase-albert-small-v2"

    embedder = SentenceTransformer(
        model_name_or_path=model_name_or_path,
        cache_folder=str(cache_folder_path),
        show_progress_bar=False,
    )

    input_texts = [
        {
            "source": "source 1",
            "text": "xxx",
        },
        {
            "source": "source 2",
            "text": "yyy",
        },
    ]
    faiss = FAISS(embedding=embedder, n_neighbors=1).fit(input_texts)
    test_texts = [
        {
            "source": "test 1",
            "text": "xx",
        },
    ]
    assert faiss.k_neighbors(test_texts) == [[input_texts[0]]]
