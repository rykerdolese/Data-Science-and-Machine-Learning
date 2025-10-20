# FILE: src/rice2025/preprocess.py
"""Simple preprocessing helpers: normalization, min-max scaling and train/test split.
These implementations are intentionally lightweight for unit testing and clarity.
"""
from typing import Tuple
import math

import random


def normalize(X):
    """L2-normalize a 2D list/iterable of numeric rows.
    Returns a new list of lists.
    """
    if X is None:
        raise ValueError("Input X is None")
    out = []
    for i, row in enumerate(X):
        if row is None:
            raise ValueError(f"Row {i} is None")
        norm = math.sqrt(sum((float(x) ** 2 for x in row)))
        if norm == 0:
            out.append([0.0 for _ in row])
        else:
            out.append([float(x) / norm for x in row])
    return out


def scale_minmax(X, feature_range=(0.0, 1.0)):
    """Min-max scale columns of a 2D iterable into feature_range.
    Returns a new list of lists.
    """
    if not X:
        return []
    lo, hi = feature_range
    # transpose
    cols = list(zip(*X))
    scaled_cols = []
    for col in cols:
        col_f = [float(x) for x in col]
        cmin = min(col_f)
        cmax = max(col_f)
        if cmax == cmin:
            # constant column -> middle of range
            scaled_cols.append([ (lo + hi) / 2.0 for _ in col_f])
        else:
            scaled_cols.append([lo + (x - cmin) * (hi - lo) / (cmax - cmin) for x in col_f])
    # transpose back
    return [list(row) for row in zip(*scaled_cols)]


def train_test_split(X, y, test_size=0.25, shuffle=True, seed=None):
    """Split X, y into train/test. X is list of rows, y is list.
    Returns X_train, X_test, y_train, y_test
    """
    if X is None or y is None:
        raise ValueError("X and y must not be None")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    n = len(X)
    if n == 0:
        return [], [], [], []
    if not 0 <= test_size <= 1:
        raise ValueError("test_size must be in [0,1]")
    indices = list(range(n))
    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(indices)
    split_at = int(n * (1 - test_size))
    train_idx = indices[:split_at]
    test_idx = indices[split_at:]
    X_train = [X[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]
    return X_train, X_test, y_train, y_test