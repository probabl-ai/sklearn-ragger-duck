from pathlib import Path

from sphinx_rag_search_engine.embedding import SentenceTransformer


def test_xxx():
    cache_folder_path = Path(__file__).parent / "data"
    model_name_or_path = "sentence-transformers/paraphrase-albert-small-v2"

    embedder = SentenceTransformer(
        model_name_or_path=model_name_or_path, cache_folder=str(cache_folder_path)
    )
    print(
        embedder.fit_transform([{"source": "hello world", "text": "hello world"}])
    )
