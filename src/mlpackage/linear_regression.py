import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

class LinearRegression:
    """
    A simple implementation of Linear Regression using the Normal Equation.
    
    Attributes
    ----------
    coefficients : np.ndarray
        The slope coefficients for each feature.
    intercept : float
        The bias term (intercept) of the regression model.
    fitted : bool
        Indicates whether the model has been trained.
    """

    def __init__(self):
        self.coefficients = None
        self.intercept = None
        self.fitted = False

    def fit(self, X, y):
        """
        Fit the linear regression model using the Normal Equation.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Target vector of shape (n_samples,).

        Raises
        ------
        ValueError
            If X or y is empty.
        """
        if X.size == 0 or y.size == 0:
            raise ValueError("Empty X or y provided to fit method.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must match.")

        # Add bias term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # Compute parameters using pseudoinverse (handles singular matrices)
        theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        self.intercept = theta[0]
        self.coefficients = theta[1:]
        self.fitted = True

    def predict(self, X):
        """
        Predict target values using the trained model.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted target values.
        """
        if not self.fitted:
            raise AttributeError("Model not fitted yet.")
        return np.dot(X, self.coefficients) + self.intercept

    def rmse(self, X, y):
        """
        Compute the Root Mean Squared Error (RMSE).

        Returns
        -------
        float
            The RMSE of the model's predictions.
        """
        y_pred = self.predict(X)
        return np.sqrt(np.mean((y - y_pred) ** 2))

    def R_squared(self, X, y):
        """
        Compute the coefficient of determination (R²).

        Returns
        -------
        float
            The R² score (1 = perfect prediction, 0 = mean prediction).
        """
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)
