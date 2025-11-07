import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class Perceptron:
    """
    Perceptron classifier for binary classification.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size for weight updates.
    n_iterations : int, default=1000
        Number of passes over the dataset.

    Attributes
    ----------
    weights : np.ndarray
        Learned weights.
    bias : float
        Learned bias term.
    loss_ : list
        Loss at each iteration.
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_ = []
        np.random.seed(42)

    def _calculate_loss(self, y_true, y_pred):
        """Mean squared error."""
        return np.mean((y_true - y_pred) ** 2)

    def fit(self, X, y):
        """
        Train the Perceptron on data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input features.
        y : np.ndarray, shape (n_samples,)
            Binary target labels (0 or 1).
        """
        n_samples, n_features = X.shape
        self.weights = np.random.rand(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = 1 if linear_output >= 0 else 0
                update = self.learning_rate * (y[idx] - y_pred)
                self.weights += update * x_i
                self.bias += update

            # Compute loss for this iteration
            y_pred_full = self.predict(X)
            self.loss_.append(self._calculate_loss(y, y_pred_full))
        return self

    def predict(self, X):
        """Predict binary labels for input features."""
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)

    def accuracy(self, X, y):
        """Compute accuracy of predictions."""
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def plot_loss(self):
        """Plot loss curve over iterations."""
        plt.plot(range(len(self.loss_)), self.loss_, marker='o')
        plt.title('Perceptron Loss over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()

    def confusion_matrix(self, X, y):
        """Return confusion matrix as a pandas DataFrame."""
        y_pred = self.predict(X)
        return pd.crosstab(y, y_pred, rownames=['Actual'], colnames=['Predicted'])
