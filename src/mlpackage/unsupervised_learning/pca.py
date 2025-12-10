"""
PCA Module for Dimensionality Reduction.

This module provides a Principal Component Analysis (PCA) implementation
similar to scikit-learn's PCA. It supports fitting, transforming, and
computing explained variance of the principal components.

Example usage
-------------
from mlpackage.pca import PCA
from mlpackage.preprocess import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(pca.explained_variance_ratio_)
"""

import numpy as np

class PCA:
    """
    Principal Component Analysis (PCA) implementation.

    Parameters
    ----------
    n_components : int or None
        Number of principal components to keep. If None, all components are kept.

    Attributes
    ----------
    components_ : np.ndarray
        Principal axes in feature space, shape (n_components, n_features).
    explained_variance_ : np.ndarray
        Variance explained by each principal component.
    explained_variance_ratio_ : np.ndarray
        Percentage of variance explained by each principal component.
    mean_ : np.ndarray
        Mean of each feature.
    """
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X):
        """
        Fit the PCA model to the data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : object
            Fitted PCA instance.
        """
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        cov_matrix = np.cov(X_centered, rowvar=False)
        eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)

        sorted_idx = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[sorted_idx]
        eig_vecs = eig_vecs[:, sorted_idx]

        if self.n_components is not None:
            eig_vals = eig_vals[:self.n_components]
            eig_vecs = eig_vecs[:, :self.n_components]

        self.components_ = eig_vecs.T
        self.explained_variance_ = eig_vals
        self.explained_variance_ratio_ = eig_vals / np.sum(eig_vals)

        return self

    def transform(self, X):
        """
        Apply dimensionality reduction to X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_transformed : np.ndarray, shape (n_samples, n_components)
            Transformed data.
        """
        if self.components_ is None:
            raise AttributeError("PCA not fitted yet. Call `fit` first.")
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X):
        """
        Fit the PCA model and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to fit and transform.

        Returns
        -------
        X_transformed : np.ndarray, shape (n_samples, n_components)
            Transformed data.
        """
        self.fit(X)
        return self.transform(X)
