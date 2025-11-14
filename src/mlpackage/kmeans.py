import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create K-means class
class KMeans:
    """
    K-Means clustering algorithm.

    Parameters
    ----------
    K : int, default=3
        Number of clusters.
    max_iters : int, default=100
        Maximum number of iterations.
    tol : float, default=1e-4
        Convergence tolerance.

    Attributes
    ----------
    centroids : np.ndarray
        Cluster centroids.
    labels_ : np.ndarray
        Labels of each point.
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.
    """

    def __init__(self, K=3, max_iters=100, tol=1e-4):
        self.K = K
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X):
        """
        Fit K-Means to the dataset.
        """
        if X.size == 0:
            raise ValueError("Empty dataset provided.")
        np.random.seed(42)
        random_indices = np.random.choice(X.shape[0], self.K, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iters):
            labels = self._assign_clusters(X)
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.K)])
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break
            self.centroids = new_centroids

        self.labels_ = self._assign_clusters(X)
        self.inertia_ = self._compute_inertia(X)
        return self

    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _compute_inertia(self, X):
        return sum(np.linalg.norm(X[self.labels_ == k] - self.centroids[k])**2 for k in range(self.K))

    def predict(self, X):
        """Predict nearest cluster index for each sample."""
        if self.centroids is None:
            raise AttributeError("Model not fitted yet.")
        return self._assign_clusters(X)

    def score(self, X):
        """Return the negative inertia (for sklearn-like scoring)."""
        if self.centroids is None:
            raise AttributeError("Model not fitted yet.")
        return -self._compute_inertia(X)