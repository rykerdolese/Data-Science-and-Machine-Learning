"""
preprocessing.py

Simple preprocessing helpers for machine learning:

- Train/test split for NumPy arrays and pandas DataFrames
- Min-max scaling
- Standardization (z-score)
- Ordinal encoding for categorical features

These implementations are intentionally lightweight for clarity and unit testing.
"""

from typing import Tuple
import numpy as np
import pandas as pd
import random


def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None) -> Tuple:
    """
    Split arrays or DataFrames into random train and test subsets.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Input features.
    y : np.ndarray, pd.Series, or pd.DataFrame
        Target values.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    shuffle : bool, default=True
        Whether to shuffle the data before splitting.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test
        Split datasets.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    
    test_size = int(n_samples * test_size)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    
    if isinstance(X, (pd.DataFrame, pd.Series)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    else:
        X_train, X_test = X[train_idx], X[test_idx]
    
    if isinstance(y, (pd.DataFrame, pd.Series)):
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        y_train, y_test = y[train_idx], y[test_idx]
    
    return X_train, X_test, y_train, y_test


class MinMaxScaler:
    """
    Scale features to a specified range (default 0 to 1).

    Parameters
    ----------
    feature_range : tuple (min, max), default=(0,1)
        Desired range of transformed data.

    Attributes
    ----------
    min_ : np.ndarray
        Minimum value for each feature.
    max_ : np.ndarray
        Maximum value for each feature.
    scale_ : np.ndarray
        Scale factor for each feature.
    min_adj_ : np.ndarray
        Adjustment to shift features into desired range.
    """
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        
    def fit(self, X):
        """Compute min, max, and scaling factor from data."""
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / (self.max_ - self.min_)
        self.min_adj_ = self.feature_range[0] - self.min_ * self.scale_
        return self
    
    def transform(self, X):
        """Scale features of X according to fitted min and scale."""
        return X * self.scale_ + self.min_adj_
    
    def fit_transform(self, X):
        """Fit to data then transform it."""
        return self.fit(X).transform(X)


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.

    Attributes
    ----------
    mean_ : np.ndarray
        Mean of each feature in the training data.
    std_ : np.ndarray
        Standard deviation of each feature in the training data.
    """
    def fit(self, X):
        """Compute mean and standard deviation for scaling."""
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0  # Prevent division by zero
        return self
    
    def transform(self, X):
        """Standardize features of X according to fitted mean and std."""
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X):
        """Fit to data then transform it."""
        return self.fit(X).transform(X)


class OrdinalEncoder:
    """
    Encode categorical features as integers.

    Parameters
    ----------
    categories : 'auto' or list of lists, default='auto'
        - 'auto': determine categories from data
        - list of lists: predefined category order for each feature

    Attributes
    ----------
    category_maps : dict
        Maps each feature's categories to integer labels.
    """
    def __init__(self, categories='auto'):
        self.categories = categories
        self.category_maps = {}

    def fit(self, X):
        """
        Fit the encoder to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Categorical input features.

        Returns
        -------
        self : OrdinalEncoder
            Fitted encoder.
        """
        X = np.array(X)
        n_features = X.shape[1]
        self.category_maps = {}

        for i in range(n_features):
            if self.categories == 'auto':
                cats = np.unique(X[:, i])
            else:
                cats = self.categories[i]
            self.category_maps[i] = {cat: idx for idx, cat in enumerate(cats)}

        return self

    def transform(self, X):
        """
        Transform categorical features to integer encodings.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Categorical input features.

        Returns
        -------
        X_out : np.ndarray, shape (n_samples, n_features)
            Integer-encoded features.
        """
        X = np.array(X)
        X_out = np.zeros_like(X, dtype=float)

        for i in range(X.shape[1]):
            mapping = self.category_maps[i]
            X_out[:, i] = [mapping[val] for val in X[:, i]]

        return X_out

    def fit_transform(self, X):
        """Fit to data then transform it."""
        return self.fit(X).transform(X)





