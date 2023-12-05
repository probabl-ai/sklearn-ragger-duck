"""The module :mod:`sphinx_rag_search_engine.embedding` contains functions to embed
transformers allowing to embed text.
"""

from ._sentence_transformer import SentenceTransformer

__all__ = ["SentenceTransformer"]