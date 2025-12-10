"""
logistic_regression.py

A minimal, from-scratch implementation of Logistic Regression using
batch gradient descent. Supports probability prediction, class prediction,
and accuracy scoring. This implementation is intended for educational
purposes and mirrors the behavior of sklearn's LogisticRegression
(without regularization or solvers).

The model optimizes the binary cross-entropy loss:

    L = −[ y log(p) + (1 − y) log(1 − p) ]

Typical usage
-------------
model = LogisticRegression(learning_rate=0.01, n_iterations=5000)
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)
preds = model.predict(X_test)
acc = model.score(X_test, y_test)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LogisticRegression:
    """
    A simple Logistic Regression classifier using batch gradient descent.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size for gradient descent.
    n_iterations : int, default=1000
        Number of gradient descent steps.

    Attributes
    ----------
    weights : np.ndarray
        Weight vector for each feature (shape: n_features,).
    bias : float
        Intercept term.
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """Initialize the logistic regression model."""
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        np.random.seed(42)  # For reproducibility

    def sigmoid(self, z):
        """
        Compute the sigmoid activation function.

        The input is clipped to avoid overflow in exp() for large values.

        Parameters
        ----------
        z : np.ndarray or float
            Input linear term.

        Returns
        -------
        np.ndarray or float
            Sigmoid transformation of the input.
        """
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Fit the logistic regression model using batch gradient descent.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Binary labels (0 or 1) of shape (n_samples,).

        Raises
        ------
        ValueError
            If X or y is empty, or if sample counts do not match.
        """
        if X.size == 0 or y.size == 0:
            raise ValueError("Empty X or y provided to fit method.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must match.")

        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features)
        self.bias = 0.0

        for i in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Optional debug logging
            if (i + 1) % 1000 == 0:
                loss = -np.mean(
                    y * np.log(y_predicted + 1e-15) +
                    (1 - y) * np.log(1 - y_predicted + 1e-15)
                )
                print(f"Iteration {i + 1}, Loss: {loss:.6f}")

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Probabilities for the positive class (values in [0, 1]).
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X):
        """
        Predict binary class labels (0 or 1).

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    def score(self, X, y):
        """
        Compute model accuracy.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            True binary labels.

        Returns
        -------
        float
            Fraction of correctly predicted samples (accuracy).
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
