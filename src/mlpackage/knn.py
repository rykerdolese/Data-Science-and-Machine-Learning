"""
knn.py
------

A simple, from-scratch implementation of the K-Nearest Neighbors (KNN) 
classification algorithm.

This module provides:
- A `KNN` class implementing basic KNN classification.
- Methods for fitting training data, predicting labels, computing accuracy,
  generating a confusion matrix, and visualizing the decision boundary.

The implementation follows the standard KNN procedure:
1. Store the training data.
2. For each query point, compute Euclidean distances to all training samples.
3. Select the k nearest neighbors.
4. Predict the label that appears most frequently among them.

This implementation is intended for educational and demonstration purposes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class KNN:
    """
    K-Nearest Neighbors classifier.

    Parameters
    ----------
    k : int, default=3
        Number of neighbors to consider when making predictions.
    """

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        """
        Store the training dataset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Training labels.
        """
        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y, dtype=int)

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to classify.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted labels.
        """
        X = np.array(X, dtype=float)
        predictions = []

        for i in range(X.shape[0]):
            x = X[i]
            distances = np.linalg.norm(self.X_train - x, axis=1)
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            most_common = np.bincount(k_nearest_labels).argmax()
            predictions.append(most_common)

        return np.array(predictions)
    
    def accuracy(self, X, y):
        """
        Compute classification accuracy.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            True labels.

        Returns
        -------
        float
            Fraction of correctly classified samples.
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def confusion_matrix(self, X, y):
        """
        Compute confusion matrix for predictions on X.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            True labels.

        Returns
        -------
        pandas.DataFrame
            Confusion matrix indexed by true labels and columns as predicted labels.
        """
        predictions = self.predict(X)
        unique_labels = np.unique(y)
        matrix = pd.DataFrame(0, index=unique_labels, columns=unique_labels)

        for true_label, pred_label in zip(y, predictions):
            matrix.loc[true_label, pred_label] += 1
        return matrix
    
    def draw_decision_boundary(self, X, y):
        """
        Plot the KNN decision boundary for a 2D dataset.

        Parameters
        ----------
        X : array-like of shape (n_samples, 2)
            2D feature matrix (only first two features used).
        y : array-like
            True labels.

        Notes
        -----
        This method only works with 2D datasets.
        """
        X = np.array(X, dtype=float)

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.1),
            np.arange(y_min, y_max, 0.1)
        )

        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Decision Boundary')
        plt.show()

