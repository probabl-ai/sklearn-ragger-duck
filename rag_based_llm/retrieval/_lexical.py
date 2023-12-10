from numbers import Integral

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils._param_validation import HasMethods, Interval
from sklearn.utils.validation import check_is_fitted


class BM25Retriever(BaseEstimator):
    """Retrieve the k-nearest neighbors using a lexical search based on BM25.

    Parameters
    ----------
    count_vectorizer : transformer, default=None
        A count vectorizer to compute the count of terms in documents. If None, a
        :class:`sklearn.feature_extraction.text.CountVectorizer` is used.

    n_neighbors : int, default=1
        Number of neighbors to retrieve.

    Attributes
    ----------
    X_fit_ : list of str or dict
        The input data.

    X_embedded_ : ndarray of shape (n_sentences, n_features)
        The embedded data.
    """

    _parameter_constraints = {
        "count_vectorizer": [HasMethods(["fit_transform", "transform"]), None],
        "n_neighbors": [Interval(Integral, left=1, right=None, closed="left")],
    }

    def __init__(self, *, count_vectorizer=None, n_neighbors=1, b=0.75, k1=1.6):
        self.count_vectorizer = count_vectorizer
        self.n_neighbors = n_neighbors
        self.b = b
        self.k1 = k1

    def fit(self, X, y=None):
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
        self.X_fit_ = X

        if isinstance(X[0], dict):
            X = [x["text"] for x in X]

        if self.count_vectorizer is None:
            self.count_vectorizer_ = CountVectorizer().fit(X)
        else:
            self.count_vectorizer_ = clone(self.count_vectorizer).fit(X)

        self.X_counts_ = self.count_vectorizer_.transform(X)
        self.n_terms_by_document_ = self.X_counts_.sum(axis=1).A1
        self.averaged_document_length_ = self.n_terms_by_document_.mean()

        # compute idf
        n_documents = len(self.X_fit_)
        n_documents_by_term = self.X_counts_.sum(axis=0).A1
        numerator = n_documents - n_documents_by_term + 0.5
        denominator = n_documents_by_term + 0.5
        self.idf_ = np.log(numerator / denominator + 1)
        self.idf_[self.idf_ < 0] = 0.25 * np.mean(self.idf_)
        return self

    def k_neighbors(self, query, *, n_neighbors=None):
        """Retrieve the k-nearest neighbors.

        Parameters
        ----------
        query : str
            The input data.

        n_neighbors : int, default=None
           The number of neighbors to retrieve. If None, the `n_neighbors` from the
           constructor is used.

        Returns
        -------
        list of str or dict
            The k-nearest neighbors from the training set.
        """
        check_is_fitted(self, "X_fit_")
        if not isinstance(query, str):
            raise TypeError(f"query should be a string, got {type(query)}.")
        n_neighbors = n_neighbors or self.n_neighbors
        query_terms_indices = self.count_vectorizer_.transform([query]).indices
        counts_query_in_X_fit = self.X_counts_[:, query_terms_indices].toarray()
        idf = self.idf_[query_terms_indices]
        numerator = counts_query_in_X_fit * (self.k1 + 1)
        denominator = counts_query_in_X_fit + self.k1 * (
            1
            - self.b
            + self.b
            * (
                self.n_terms_by_document_.reshape(-1, 1)
                / self.averaged_document_length_
            )
        )
        scores = (idf * numerator / denominator).sum(axis=1)
        indices = scores.argsort()[::-1][:n_neighbors]
        if isinstance(self.X_fit_[0], dict):
            return [
                {
                    "source": self.X_fit_[neighbor]["source"],
                    "text": self.X_fit_[neighbor]["text"],
                }
                for neighbor in indices
            ]
        else:  # isinstance(self.X_fit_[0], str)
            return [self.X_fit_[neighbor] for neighbor in indices]
