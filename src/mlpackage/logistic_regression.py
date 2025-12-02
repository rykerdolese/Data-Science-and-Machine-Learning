import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

class LogisticRegression:
    """
    A simple Logistic Regression classifier using batch gradient descent.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size for gradient descent.
    n_iterations : int, default=1000
        Number of iterations for gradient descent.

    Attributes
    ----------
    weights : np.ndarray
        Weight vector for each feature.
    bias : float
        Intercept term.
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        np.random.seed(42)  # For reproducibility

    def sigmoid(self, z):
        """Compute the sigmoid function with clipping for numerical stability."""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Fit the logistic regression model using batch gradient descent.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features).
        y : np.ndarray
            Target labels (n_samples,), must be 0 or 1.
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

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Optional verbose logging every 1000 iterations
            if (i + 1) % 1000 == 0:
                loss = -np.mean(
                    y * np.log(y_predicted + 1e-15) + (1 - y) * np.log(1 - y_predicted + 1e-15)
                )
                print(f"Iteration {i + 1}, Loss: {loss:.6f}")

    def predict_proba(self, X):
        """
        Predict class probabilities for input samples.
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X):
        """
        Predict binary class labels (0 or 1).
        """
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    def score(self, X, y):
        """
        Compute accuracy score of the model.

        Returns
        -------
        float
            Fraction of correctly classified samples.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
