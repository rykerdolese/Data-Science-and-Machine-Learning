"""
DBSCAN Clustering Implementation

This module provides a DBSCAN clustering algorithm similar to sklearn's interface.

Classes
-------
DBSCAN(eps=0.5, min_samples=5)
    Density-based spatial clustering of applications with noise.
"""

import numpy as np
from collections import deque

class DBSCAN:
    """
    Density-Based Spatial Clustering of Applications with Noise (DBSCAN).

    Parameters
    ----------
    eps : float, default=0.5
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other.
    
    min_samples : int, default=5
        The number of samples in a neighborhood for a point to be considered
        as a core point (including the point itself).
    
    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point. Noisy samples are labeled as -1.
    """

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X):
        """
        Fit the DBSCAN model from the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        labels = np.full(n_samples, -1, dtype=int)
        cluster_id = 0

        visited = np.zeros(n_samples, dtype=bool)

        def region_query(idx):
            """Return indices of neighbors within eps distance."""
            distances = np.linalg.norm(X - X[idx], axis=1)
            return np.where(distances <= self.eps)[0]

        for i in range(n_samples):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = region_query(i)
            if len(neighbors) < self.min_samples:
                labels[i] = -1  # noise
            else:
                labels[i] = cluster_id
                queue = deque(neighbors.tolist())
                while queue:
                    j = queue.popleft()
                    if not visited[j]:
                        visited[j] = True
                        j_neighbors = region_query(j)
                        if len(j_neighbors) >= self.min_samples:
                            queue.extend(j_neighbors.tolist())
                    if labels[j] == -1:
                        labels[j] = cluster_id
                cluster_id += 1

        self.labels_ = labels
        return self

    def fit_predict(self, X):
        """
        Fit DBSCAN to X and return cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels for each point.
        """
        self.fit(X)
        return self.labels_