from numbers import Integral, Real

from sklearn.base import BaseEstimator, _fit_context
from sklearn.utils._param_validation import HasMethods, Interval


class RetrieverReranker(BaseEstimator):
    """Hybrid retriever (lexical and semantic) followed by a cross-encoder reranker.

    We can accept several retrievers in case you want to rerank the results of
    several retrievers.

    Parameters
    ----------
    retrievers : list of retriever instances
        The retrievers to use for retrieving the context. We expect the retrievers to
        implement a `query` method.

    cross_encoder : :class:`~sentence_transformers.cross_encoder.CrossEncoder`
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

    drop_duplicates : bool, default=True
        Whether to drop duplicates from the retrieved documents. This step is done
        right after the retrieval step.
    """

    _parameter_constraints = {
        "retrievers": [list],
        "cross_encoder": [HasMethods(["predict"])],
        "min_top_k": [Interval(Integral, left=0, right=None, closed="left"), None],
        "max_top_k": [Interval(Integral, left=0, right=None, closed="left"), None],
        "threshold": [Real, None],
        "drop_duplicates": [bool],
    }

    def __init__(
        self,
        *,
        retrievers,
        cross_encoder,
        min_top_k=None,
        max_top_k=None,
        threshold=None,
        drop_duplicates=True,
    ):
        self.retrievers = retrievers
        self.cross_encoder = cross_encoder
        self.min_top_k = min_top_k
        self.max_top_k = max_top_k
        self.threshold = threshold
        self.drop_duplicates = drop_duplicates

    @_fit_context(prefer_skip_nested_validation=False)
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
        return self

    @staticmethod
    def _get_context(search):
        if isinstance(search, dict):
            return search["text"]
        return search

    def query(self, query):
        """Retrieve the most relevant documents for the query.

        Parameters
        ----------
        query : str
            The user query.

        Returns
        -------
        list of str or dict
            The list of the most relevant document from the training set.
        """
        unranked_search = []
        for retriever in self.retrievers:
            unranked_search += retriever.query(query)

        if not unranked_search:
            return []

        if self.drop_duplicates:
            filtered_unranked_search = []
            chunk_already_seen = set()
            for search in unranked_search:
                if self._get_context(search) in chunk_already_seen:
                    continue
                chunk_already_seen.add(self._get_context(search))
                filtered_unranked_search.append(search)
            unranked_search = filtered_unranked_search

        merged_search = [
            (query, self._get_context(search)) for search in unranked_search
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
