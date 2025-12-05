"""
decision_tree_regressor.py
Simple Decision Tree and Random Forest regressor for educational use.

This module provides a minimal, easy-to-read implementation of:
- A Decision Tree Regressor using variance reduction.
- A Random Forest Regressor that ensembles multiple DecisionTreeRegressor models.

The goal is to keep the code compact and transparent so that students can
understand how tree-based models work internally, without relying on scikit-learn.

Example
-------
>>> import numpy as np
>>> from decision_tree_regressor import DecisionTreeRegressor, RandomForestRegressor
>>>
>>> X = np.array([[0], [1], [2], [3]])
>>> y = np.array([0.1, 0.9, 2.1, 3.0])
>>>
>>> tree = DecisionTreeRegressor(max_depth=2)
>>> tree.fit(X, y)
>>> tree.predict(X)
array([0.1, 0.9, 2.1, 3.0])
"""

import numpy as np

class DecisionTreeRegressor:
    """
    A simple Decision Tree Regressor using variance reduction for educational purposes.

    Parameters
    ----------
    max_depth : int or None, default=None
        Maximum depth of the tree. If None, tree grows until leaves are pure.

    Attributes
    ----------
    tree : tuple or float
        Internal tree representation:
        (feature_index, threshold, left_subtree, right_subtree)
        or a leaf value.
    """

    def __init__(self, max_depth=None):
        """Initialize the DecisionTreeRegressor with optional maximum depth."""
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        """
        Fit the decision tree regressor to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : DecisionTreeRegressor
            The fitted estimator.
        """
        self.tree = self._build_tree(X, y)
        return self

    def _variance(self, y):
        """
        Compute variance of target values.

        Parameters
        ----------
        y : array-like
            Target values.

        Returns
        -------
        float
            Variance of y.
        """
        return np.var(y) if len(y) > 0 else 0

    def _variance_reduction(self, parent_y, left_y, right_y):
        """
        Compute variance reduction of a candidate split.

        Parameters
        ----------
        parent_y : array-like
            Target values at the parent node.
        left_y : array-like
            Target values in the left split.
        right_y : array-like
            Target values in the right split.

        Returns
        -------
        float
            Variance reduction from the split.
        """
        p_left = len(left_y) / len(parent_y)
        p_right = 1 - p_left
        return self._variance(parent_y) - (p_left * self._variance(left_y) + p_right * self._variance(right_y))

    def _best_split(self, X, y):
        """
        Find the best feature and threshold to split on based on variance reduction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        best_feature : int or None
            Index of the best feature for splitting.
        best_threshold : float or None
            Threshold for the split.
        """
        num_samples, num_features = X.shape
        best_var_red, best_feature, best_threshold = -1, None, None

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left_idx = X[:, feature] <= t
                right_idx = ~left_idx
                if left_idx.sum() == 0 or right_idx.sum() == 0:
                    continue
                var_red = self._variance_reduction(y, y[left_idx], y[right_idx])
                if var_red > best_var_red:
                    best_var_red, best_feature, best_threshold = var_red, feature, t

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix at current node.
        y : array-like of shape (n_samples,)
            Target values at current node.
        depth : int
            Current depth of the tree.

        Returns
        -------
        tuple or float
            Tree node represented as
            (feature_index, threshold, left_tree, right_tree)
            or a leaf value.
        """
        if len(np.unique(y)) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            return np.mean(y)

        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return np.mean(y)

        left_idx = X[:, best_feature] <= best_threshold
        right_idx = ~left_idx
        left_tree = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_tree = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        return (best_feature, best_threshold, left_tree, right_tree)

    def _predict_sample(self, sample, tree):
        """
        Predict the target value for a single sample.

        Parameters
        ----------
        sample : array-like of shape (n_features,)
            Single input sample.
        tree : tuple or float
            Current node of the decision tree.

        Returns
        -------
        float
            Predicted target value.
        """
        if not isinstance(tree, tuple):
            return tree
        feature, threshold, left, right = tree
        return self._predict_sample(sample, left if sample[feature] <= threshold else right)

    def predict(self, X):
        """
        Predict target values for input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted target values.
        """
        return np.array([self._predict_sample(x, self.tree) for x in X])

    def score(self, X, y):
        """
        Compute R² (coefficient of determination) score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            True target values.

        Returns
        -------
        float
            R² score.
        """
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y))**2)
        ss_res = np.sum((y - y_pred)**2)
        return 1 - ss_res / ss_total


class RandomForestRegressor:
    """
    A simple Random Forest Regressor built on DecisionTreeRegressor.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees.
    max_depth : int or None, default=None
        Maximum depth per tree.
    max_features : {'sqrt', None}, default='sqrt'
        Number of features to consider per tree.

    Attributes
    ----------
    trees : list of (DecisionTreeRegressor, feature_indices)
        Ensemble of trees and corresponding feature subsets.
    """

    def __init__(self, n_estimators=100, max_depth=None, max_features='sqrt'):
        """Initialize the RandomForestRegressor."""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        """
        Fit the Random Forest Regressor using bootstrap aggregation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : RandomForestRegressor
            Fitted model.
        """
        n_samples, n_features = X.shape
        self.trees = []

        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X[indices], y[indices]

            if self.max_features == 'sqrt':
                features_indices = np.random.choice(n_features, int(np.sqrt(n_features)), replace=False)
            else:
                features_indices = np.arange(n_features)

            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X_sample[:, features_indices], y_sample)
            self.trees.append((tree, features_indices))

        return self

    def predict(self, X):
        """
        Predict target values using the Random Forest ensemble.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted target values (averaged across trees).
        """
        predictions = np.zeros((X.shape[0], len(self.trees)))
        for i, (tree, features_indices) in enumerate(self.trees):
            predictions[:, i] = tree.predict(X[:, features_indices])
        return np.mean(predictions, axis=1)

