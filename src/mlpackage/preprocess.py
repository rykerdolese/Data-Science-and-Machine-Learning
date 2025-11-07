"""Simple preprocessing helpers: normalization, min-max scaling and train/test split.
These implementations are intentionally lightweight for unit testing and clarity.
"""
from typing import Tuple
import math
import numpy as np

import random

import numpy as np
import pandas as pd

def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None):
    """
    Split arrays or DataFrames into random train and test subsets.
    Works with both NumPy arrays and pandas DataFrames/Series.
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
    
    # Handle pandas or numpy separately
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
    Scale features to a given range (default 0 to 1).
    """
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        
    def fit(self, X):
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / (self.max_ - self.min_)
        self.min_adj_ = self.feature_range[0] - self.min_ * self.scale_
        return self
    
    def transform(self, X):
        return X * self.scale_ + self.min_adj_
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)

class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.
    """
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        # Avoid divide-by-zero
        self.std_[self.std_ == 0] = 1.0
        return self
    
    def transform(self, X):
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)


import numpy as np
import pandas as pd

class OrdinalEncoder:
    def __init__(self, categories='auto'):
        """
        categories: list of lists defining category order for each feature,
                    or 'auto' to infer from data.
        """
        self.categories = categories
        self.category_maps = {}

    def fit(self, X):
        """
        X: pandas DataFrame or 2D array of categorical features.
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
        Transform categorical values to their integer encodings.
        """
        X = np.array(X)
        X_out = np.zeros_like(X, dtype=float)

        for i in range(X.shape[1]):
            mapping = self.category_maps[i]
            X_out[:, i] = [mapping[val] for val in X[:, i]]

        return X_out

    def fit_transform(self, X):
        return self.fit(X).transform(X)




