import logging
import time
from numbers import Integral

import faiss
from sklearn.base import BaseEstimator
from sklearn.utils._param_validation import HasMethods, Interval
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


class SemanticRetriever(BaseEstimator):
    """Retrieve the k-nearest neighbors using a semantic embedding.

    The index is build using the FAISS library.

    Parameters
    ----------
    embedding : transformer
        An embedding following the scikit-learn transformer API.

    top_k : int, default=1
        Number of documents to retrieve.

    Attributes
    ----------
    X_fit_ : list of str or dict
        The input data.

    X_embedded_ : ndarray of shape (n_sentences, n_features)
        The embedded data.
    """

    _parameter_constraints = {
        "embedding": [HasMethods(["fit_transform", "transform"])],
        "top_k": [Interval(Integral, left=1, right=None, closed="left")],
    }

    def __init__(self, *, embedding, top_k=1):
        self.embedding = embedding
        self.top_k = top_k

    def fit(self, X, y=None):
        """Embed the sentences and create the index.

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
        self.X_fit_ = X
        start = time.time()
        self.X_embedded_ = self.embedding.fit_transform(X)
        self.index_ = faiss.IndexFlatIP(self.X_embedded_.shape[1])
        self.index_.add(self.X_embedded_)
        logger.info(f"Index created in {time.time() - start:.2f}s")
        return self

    def query(self, query):
        """Retrieve the most relevant documents for the query.

        The inner product is used to compute the cosine similarity meaning that
        we expect the embedding to be normalized.

        Parameters
        ----------
        query : str
            The input data.

        Returns
        -------
        list of str or dict
            The list of the most relevant document from the training set.
        """
        check_is_fitted(self, "X_fit_")
        if not isinstance(query, str):
            raise TypeError(f"query should be a string, got {type(query)}.")
        start = time.time()
        X_embedded = self.embedding.transform(query)
        # normalize vectors to compute the cosine similarity
        _, indices = self.index_.search(X_embedded, self.top_k)
        logger.info(f"Semantic search done in {time.time() - start:.2f}s")
        if isinstance(self.X_fit_[0], dict):
            return [
                {
                    "source": self.X_fit_[neighbor]["source"],
                    "text": self.X_fit_[neighbor]["text"],
                }
                for neighbor in indices[0]
                if neighbor != -1
            ]
        else:  # isinstance(self.X_fit_[0], str)
            return [self.X_fit_[neighbor] for neighbor in indices[0] if neighbor != -1]
