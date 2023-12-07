from numbers import Integral

import faiss
from sklearn.base import BaseEstimator
from sklearn.utils._param_validation import HasMethods, Interval


class FAISS(BaseEstimator):
    """K-Nearest Neighbors using FAISS indexing.

    Parameters
    ----------
    embedding : transformer
        An embedding following the scikit-learn transformer API.

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
        "embedding": [HasMethods(["fit_transform", "transform"])],
        "n_neighbors": [Interval(Integral, left=1, right=None, closed="left")],
    }

    def __init__(self, *, embedding, n_neighbors=1):
        self.embedding = embedding
        self.n_neighbors = n_neighbors

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
        self.X_embedded_ = self.embedding.fit_transform(X)
        # normalize vectors to compute the cosine similarity
        faiss.normalize_L2(self.X_embedded_)
        self.index_ = faiss.IndexFlatIP(self.X_embedded_.shape[1])
        self.index_.add(self.X_embedded_)
        return self

    def k_neighbors(self, X, n_neighbors=None):
        """Retrieve the k-nearest neighbors.

        Parameters
        ----------
        X : str or list of str or dict
            The input data.

        n_neighbors : int, default=None
           The number of neighbors to retrieve. If None, the `n_neighbors` from the
           constructor is used.

        Returns
        -------
        list of str or dict
            The k-nearest neighbors from the training set.
        """
        n_neighbors = n_neighbors or self.n_neighbors
        X_embedded = self.embedding.transform(X)
        # normalize vectors to compute the cosine similarity
        faiss.normalize_L2(X_embedded)
        _, indices = self.index_.search(X_embedded, n_neighbors)
        if isinstance(self.X_fit_[0], dict):
            return [
                [
                    {
                        "source": self.X_fit_[neighbor]["source"],
                        "text": self.X_fit_[neighbor]["text"],
                    }
                    for neighbor in neighbors
                ]
                for neighbors in indices
            ]
        else:  # isinstance(self.X_fit_[0], str)
            return [
                [self.X_fit_[neighbor] for neighbor in neighbors]
                for neighbors in indices
            ]
