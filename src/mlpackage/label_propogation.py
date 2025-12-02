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

    After fitting:
    --------------
    classes_ : array
        Array of label classes.
    n_classes_ : int
        Number of unique classes.
    Y : ndarray
        One-hot encoded label matrix.
    T : ndarray
        Row-normalized RBF affinity (transition) matrix.
    F : ndarray
        Final propagated label distribution matrix.
    y_pred : ndarray
        Predicted labels (argmax over F).
    """

    def __init__(self, alpha: float = 0.9, sigma: float = 1.0, max_iter: int = 1000, tol: float = 1e-4, clamp: bool = True):
        """
        Parameters
        ----------
        alpha : float
            Weight for propagation vs. initial labels.
        sigma : float
            RBF kernel width.
        max_iter : int
            Maximum iterations for label propagation.
        tol : float
            Convergence tolerance.
        clamp : bool
            Whether to clamp labeled points during iterations.
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
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        W : np.ndarray
            Affinity matrix of shape (n_samples, n_samples).
        """
        sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + \
                   np.sum(X**2, axis=1) - \
                   2 * (X @ X.T)
        W = np.exp(-sq_dists / (2 * self.sigma**2))
        np.fill_diagonal(W, 0)  # Remove self-loops
        return W

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LabelPropagationCustom":
        """
        Fit the label propagation model.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Labels with -1 for unlabeled points.

        Returns
        -------
        self : LabelPropagationCustom
            Fitted model.
        """
        self.X = X
        n = X.shape[0]
        labeled_mask = (y != -1)

        # Step 1: One-hot encode labeled points
        encoder = OneHotEncoder(sparse_output=False)
        true_labels = y[labeled_mask].reshape(-1, 1)
        Y_labeled = encoder.fit_transform(true_labels)

        self.classes_ = encoder.categories_[0]
        self.n_classes_ = len(self.classes_)

        # Step 2: Initialize full label matrix
        self.Y = np.zeros((n, self.n_classes_))
        self.Y[labeled_mask] = Y_labeled

        # Step 3: Build affinity and transition matrix
        W = self._rbf_affinity(X)
        T = W / (W.sum(axis=1, keepdims=True) + 1e-12)
        self.T = T

        # Step 4: Iterative propagation
        F = self.Y.copy()
        for _ in range(self.max_iter):
            F_new = self.alpha * (T @ F) + (1 - self.alpha) * self.Y
            if self.clamp:
                F_new[labeled_mask] = self.Y[labeled_mask]

            if np.linalg.norm(F_new - F) < self.tol:
                break

            F = F_new

        self.F = F

        # Step 5: Final predictions
        self.y_pred = np.argmax(F, axis=1)

        return self

    def predict(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Return predicted labels.

        Parameters
        ----------
        X : np.ndarray, optional
            Not used, kept for API compatibility.

        Returns
        -------
        y_pred : np.ndarray
            Predicted labels for all points.
        """
        return self.y_pred
