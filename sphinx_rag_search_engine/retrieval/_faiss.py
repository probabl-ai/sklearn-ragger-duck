from numbers import Integral

import faiss
from sklearn.base import BaseEstimator
from sklearn.utils._param_validation import HasMethods, Interval


class FAISS(BaseEstimator):

    _parameter_constraints = {
        "embedding": [HasMethods(["fit_transform", "transform"])],
        "top_k": [Interval(Integral, left=1, right=None, closed="left")],
    }

    def __init__(self, *, embedding, n_neighbors=1):
        self.embedding = embedding
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        self._validate_params()
        self.X_fit_ = X
        self.X_embedded_ = self.embedding.fit_transform(X)
        # normalize vectors to compute the cosine similarity
        faiss.normalize_L2(self.X_embedded_)
        self.index_ = faiss.IndexFlatIP(self.X_embedded_.shape[1])
        self.index_.add(self.X_embedded_)
        return self

    def k_neighbors(self, X, n_neighbors=None):
        n_neighbors = n_neighbors or self.n_neighbors
        X_embedded = self.embedding.transform(X)
        # normalize vectors to compute the cosine similarity
        faiss.normalize_L2(X_embedded)
        _, indices = self.index_.search(X_embedded, n_neighbors)
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