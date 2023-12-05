from pathlib import Path

from sphinx_rag_search_engine.embedding import SentenceTransformer
from sphinx_rag_search_engine.retrieval import FAISS


def test_xxx():
    cache_folder_path = (
        Path(__file__).parent.parent.parent / "embedding" / "tests" / "data"
    )
    model_name_or_path = "sentence-transformers/paraphrase-albert-small-v2"

    embedder = SentenceTransformer(
        model_name_or_path=model_name_or_path, cache_folder=str(cache_folder_path)
    )
    faiss = FAISS(embedding=embedder).fit([{"source": "hello world", "text": "hello world"}])
