"""
label_propagation_custom.py
---------------------------

A from-scratch implementation of graph-based Label Propagation for 
semi-supervised learning.

This module provides:
- Construction of an RBF-kernel similarity graph.
- Row-normalized transition matrix for probability propagation.
- Iterative label propagation with optional clamping of labeled nodes.
- One-hot encoding of initial labels and soft label distribution updates.
- Final hard-label predictions via argmax over propagated distributions.

The algorithm assumes:
- Input labels use `-1` to mark unlabeled samples.
- Features are continuous and reasonably scaled (RBF kernel sensitive to scale).

This implementation is designed primarily for instructional and 
prototype experimentation.
"""

from typing import Optional
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class LabelPropagationCustom:
    """
    Graph-based Label Propagation for semi-supervised learning.

    This implementation uses an RBF kernel to compute the similarity graph
    and iteratively propagates labels from labeled points to unlabeled points.

    Attributes
    ----------
    alpha : float
        Propagation factor (0 < alpha < 1). Higher values allow labels to propagate further.
    sigma : float
        RBF kernel bandwidth.
    max_iter : int
        Maximum number of iterations for propagation.
    tol : float
        Convergence tolerance. Iteration stops when max change < tol.
    clamp : bool
        Whether to keep original labels fixed during propagation.

    After fitting
    -------------
    classes_ : ndarray
        Unique label classes.
    n_classes_ : int
        Number of unique classes.
    Y : ndarray of shape (n_samples, n_classes)
        One-hot encoded initial label matrix.
    T : ndarray of shape (n_samples, n_samples)
        Row-normalized RBF transition matrix.
    F : ndarray of shape (n_samples, n_classes)
        Final soft label distribution matrix after propagation.
    y_pred : ndarray of shape (n_samples,)
        Final predicted labels as class indices.
    """

    def __init__(self, alpha: float = 0.9, sigma: float = 1.0,
                 max_iter: int = 1000, tol: float = 1e-4, clamp: bool = True):
        """
        Initialize the Label Propagation model.

        Parameters
        ----------
        alpha : float
            Weight for propagation versus initial labels (0 < alpha < 1).
        sigma : float
            Width of the RBF kernel used for similarity computation.
        max_iter : int
            Maximum number of propagation iterations.
        tol : float
            Convergence tolerance for stopping criterion.
        clamp : bool
            If True, labeled points remain fixed during propagation.
        """
        self.alpha = alpha
        self.sigma = sigma
        self.max_iter = max_iter
        self.tol = tol
        self.clamp = clamp

    def _rbf_affinity(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the RBF affinity matrix for input features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        W : ndarray of shape (n_samples, n_samples)
            RBF affinity (similarity) matrix.
        """
        sq_dists = (
            np.sum(X**2, axis=1).reshape(-1, 1)
            + np.sum(X**2, axis=1)
            - 2 * (X @ X.T)
        )

        W = np.exp(-sq_dists / (2 * self.sigma**2))
        np.fill_diagonal(W, 0)  # Remove self-similarities
        return W

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LabelPropagationCustom":
        """
        Fit the label propagation model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input feature matrix.
        y : ndarray of shape (n_samples,)
            Label vector where labeled points have integer labels
            and unlabeled points are marked with -1.

        Returns
        -------
        self : LabelPropagationCustom
            The fitted model.
        """
        self.X = X
        n = X.shape[0]
        labeled_mask = (y != -1)

        # --- One-hot encode labeled points ---
        encoder = OneHotEncoder(sparse_output=False)
        true_labels = y[labeled_mask].reshape(-1, 1)
        Y_labeled = encoder.fit_transform(true_labels)

        self.classes_ = encoder.categories_[0]
        self.n_classes_ = len(self.classes_)

        # --- Initialize label matrix ---
        self.Y = np.zeros((n, self.n_classes_))
        self.Y[labeled_mask] = Y_labeled

        # --- Build similarity and transition matrices ---
        W = self._rbf_affinity(X)
        T = W / (W.sum(axis=1, keepdims=True) + 1e-12)
        self.T = T

        # --- Label propagation iterations ---
        F = self.Y.copy()

        for _ in range(self.max_iter):
            F_new = self.alpha * (T @ F) + (1 - self.alpha) * self.Y

            if self.clamp:
                F_new[labeled_mask] = self.Y[labeled_mask]

            if np.linalg.norm(F_new - F) < self.tol:
                break

            F = F_new

        self.F = F
        self.y_pred = np.argmax(F, axis=1)

        return self

    def predict(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Return predicted labels.

        Parameters
        ----------
        X : ndarray, optional
            Not used. Included for API compatibility with sklearn.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted labels as integer class indices.
        """
        return self.y_pred

