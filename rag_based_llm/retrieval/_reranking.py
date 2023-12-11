from numbers import Integral, Real

from sklearn.base import BaseEstimator
from sklearn.utils._param_validation import HasMethods, Interval


class RetrieverReranker(BaseEstimator):
    """Hybrid retriever (lexical and semantic) followed by a cross-encoder reranker.

    Parameters
    ----------
    semantic_retriever : semantic retriever
        Semantic retriever used to retrieve the context.

    lexical_retriever : lexical retriever
        Lexical retriever used to retrieve the context.

    cross_encoder : :obj:`sentence_transformers.CrossEncoder`
        Cross-encoder used to rerank the results of the hybrid retriever.

    min_top_k : int, default=None
        Minimum number of document to retrieve. If None, it is possible to return
        less than `min_top_k` documents.

    max_top_k : int, default=None
        Maximum number of document to retrieve. If None, all the documents are
        retrieved.

    threshold : float, default=None
        Threshold to filter the scores of the `cross_encoder`. If None, the
        scores are note filtered based on a threshold.
    """
    _parameter_constraints = {
        "semantic_retriever": [HasMethods(["fit", "query"])],
        "lexical_retriever": [HasMethods(["fit", "query"])],
        "min_top_k": [Interval(Integral, left=0, right=None, closed="left"), None],
        "max_top_k": [Interval(Integral, left=0, right=None, closed="left"), None],
        "threshold": [Real, None],
    }

    def __init__(
        self,
        *,
        semantic_retriever,
        lexical_retriever,
        cross_encoder,
        min_top_k=None,
        max_top_k=None,
        threshold=None,
    ):
        self.semantic_retriever = semantic_retriever
        self.lexical_retriever = lexical_retriever
        self.cross_encoder = cross_encoder
        self.min_top_k = min_top_k
        self.max_top_k = max_top_k
        self.threshold = threshold

    def fit(self, X=None, y=None):
        """Compute the vocabulary and the idf.

        Parameters
        ----------
        X : list of str or dict
            The input data.

        y : None
            This parameter is ignored.

        Returns
        -------
        self
            The fitted estimator.
        """
        self._validate_params()
        return self

    @staticmethod
    def _get_context(search):
        if isinstance(search, dict):
            return search["text"]
        return search

    def query(
        self,
        *,
        query,
        lexical_query=None,
        semantic_query=None,
    ):
        """Retrieve the most relevant documents for the query.

        Parameters
        ----------
        query : str
            The user query.

        lexical_query : str, default=None
            A specific query to retrieve the context of the lexical search. If None,
            `query` is used.

        semantic_query : str, default=None
            A specific query to retrieve the context of the semantic search. If None,
            `query` is used.

        Returns
        -------
        list of str or dict
            The list of the most relevant document from the training set.
        """
        if lexical_query is None:
            lexical_query = query
        if semantic_query is None:
            semantic_query = query

        if self.lexical_retriever is not None:
            lexical_search = self.lexical_retriever.query(lexical_query)
        else:
            lexical_search = []

        if self.semantic_retriever is not None:
            semantic_search = self.semantic_retriever.query(semantic_query)
        else:
            semantic_search = []

        unranked_search = lexical_search + semantic_search
        if not unranked_search:
            return []

        merged_search = [
            (query, self._get_context(search))
            for search in lexical_search + semantic_search
        ]
        scores = self.cross_encoder.predict(merged_search)
        indices = scores.argsort()[::-1]
        sorted_scores = scores[indices]
        if self.threshold is not None:
            indices_thresholded = indices[sorted_scores > self.threshold]
        else:
            indices_thresholded = indices

        if self.min_top_k is not None or self.max_top_k is not None:
            if self.min_top_k is not None and len(indices_thresholded) < self.min_top_k:
                indices_thresholded = indices[: self.min_top_k]
            elif (
                self.max_top_k is not None and len(indices_thresholded) > self.max_top_k
            ):
                indices_thresholded = indices[: self.max_top_k]

        return [unranked_search[idx] for idx in indices_thresholded]

    def _more_tags(self):
        return {"stateless": True}
