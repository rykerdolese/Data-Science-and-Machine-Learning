# FILE: src/rice2025/knn.py
"""A simple k-nearest-neighbors classifier/regressor.
KNN supports k (int >=1) and mode 'classification' or 'regression'.
For classification labels can be hashable; for regression labels must be numeric.
"""
from typing import List
from collections import Counter

from .metrics import euclidean
from .postprocess import majority_vote, average_label


class KNN:
    def __init__(self, k=3, mode='classification'):
        if not isinstance(k, int) or k < 1:
            raise ValueError('k must be int >= 1')
        if mode not in ('classification', 'regression'):
            raise ValueError("mode must be 'classification' or 'regression'")
        self.k = k
        self.mode = mode
        self._fit_X = None
        self._fit_y = None

    def fit(self, X: List[List[float]], y: List):
        if X is None or y is None:
            raise ValueError('X and y must not be None')
        if len(X) != len(y):
            raise ValueError('X and y must have same length')
        if len(X) == 0:
            raise ValueError('Training set must not be empty')
        self._fit_X = [list(map(float, row)) for row in X]
        self._fit_y = list(y)
        return self

    def _check_fitted(self):
        if self._fit_X is None or self._fit_y is None:
            raise RuntimeError('Model is not fitted yet')

    def _neighbors(self, x):
        self._check_fitted()
        dists = []
        xf = list(map(float, x))
        for xi, yi in zip(self._fit_X, self._fit_y):
            d = euclidean(xf, xi)
            dists.append((d, yi))
        dists.sort(key=lambda t: t[0])
        return dists[: self.k]

    def predict(self, X: List[List[float]]):
        if X is None:
            raise ValueError('X must not be None')
        self._check_fitted()
        preds = []
        for x in X:
            neigh = self._neighbors(x)
            labels = [lab for (_, lab) in neigh]
            if self.mode == 'classification':
                preds.append(majority_vote(labels))
            else:
                preds.append(average_label(labels))
        return preds