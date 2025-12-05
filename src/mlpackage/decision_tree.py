"""
decision_tree.py
Simple Decision Tree and Random Forest implementation for educational use.

This module provides a minimal, easy-to-read implementation of:
- A Decision Tree classifier using entropy and information gain.
- A Random Forest classifier that ensembles multiple DecisionTree models.

The goal is to keep the code compact and transparent so that students can
understand how tree-based models work internally, without relying on
scikit-learn.

Example
-------
>>> import numpy as np
>>> from my_module import DecisionTree, RandomForest
>>>
>>> X = np.array([[0, 0],
...               [0, 1],
...               [1, 0],
...               [1, 1]])
>>> y = np.array([0, 0, 1, 1])
>>>
>>> tree = DecisionTree(max_depth=2)
>>> tree.fit(X, y)
>>> tree.predict(X)
array([0, 0, 1, 1])
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define basic decision tree class
import numpy as np

class DecisionTree:
    """
    A simple Decision Tree classifier using information gain and entropy.

    This implementation provides an educational, lightweight decision tree
    without relying on scikit-learn. It supports recursive binary splitting
    on continuous-valued features.

    Parameters
    ----------
    max_depth : int or None, default=None
        Maximum depth of the tree. If None, the tree grows until all leaves
        are pure.

    Attributes
    ----------
    tree : tuple or int
        The internal representation of the learned tree.
        Format:
            (feature_index, threshold, left_subtree, right_subtree)
        or a class label at a leaf.
    """

    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        """
        Fit the decision tree classifier to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Target class labels (integer-encoded).

        Returns
        -------
        self : DecisionTree
            The fitted estimator.

        Raises
        ------
        ValueError
            If X or y is empty or has inconsistent lengths.
        """
        if X.size == 0 or y.size == 0:
            raise ValueError("Empty X or y provided.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must match.")
        self.tree = self._build_tree(X, y)
        return self

    def _entropy(self, y):
        """
        Compute the entropy of a label distribution.

        Parameters
        ----------
        y : array-like
            Class labels.

        Returns
        -------
        float
            Entropy value.
        """
        counts = np.bincount(y)
        probabilities = counts[counts > 0] / len(y)
        return -np.sum(probabilities * np.log2(probabilities))

    def _information_gain(self, parent_y, left_y, right_y):
        """
        Compute information gain for a candidate split.

        Parameters
        ----------
        parent_y : array-like
            Labels at the parent node.
        left_y : array-like
            Labels in the left subset.
        right_y : array-like
            Labels in the right subset.

        Returns
        -------
        float
            Information gain of the split.
        """
        p_left = len(left_y) / len(parent_y)
        p_right = 1 - p_left
        return (
            self._entropy(parent_y)
            - (p_left * self._entropy(left_y) + p_right * self._entropy(right_y))
        )

    def _best_split(self, X, y):
        """
        Find the best feature and threshold to split on.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Labels.

        Returns
        -------
        best_feature : int or None
            Index of the best feature for splitting.
        best_threshold : float or None
            Threshold to split on.
        """
        num_samples, num_features = X.shape
        best_gain, best_feature, best_threshold = -1, None, None

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left_idx = X[:, feature] <= t
                right_idx = X[:, feature] > t
                if left_idx.sum() == 0 or right_idx.sum() == 0:
                    continue

                gain = self._information_gain(y, y[left_idx], y[right_idx])
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, feature, t

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.

        Parameters
        ----------
        X : array-like
            Feature matrix at current node.
        y : array-like
            Labels at current node.
        depth : int
            Current depth in the tree.

        Returns
        -------
        tuple or int
            Tree node represented as
            (feature_index, threshold, left_tree, right_tree)
            or a class label at a leaf.
        """
        unique_classes, counts = np.unique(y, return_counts=True)

        # Leaf: pure or max depth reached
        if len(unique_classes) == 1 or (
            self.max_depth is not None and depth >= self.max_depth
        ):
            return unique_classes[np.argmax(counts)]

        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return unique_classes[np.argmax(counts)]

        left_idx = X[:, best_feature] <= best_threshold
        right_idx = ~left_idx

        left_tree = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_tree = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return (best_feature, best_threshold, left_tree, right_tree)

    def _predict_sample(self, sample, tree):
        """
        Predict the class label for a single sample.

        Parameters
        ----------
        sample : array-like of shape (n_features,)
            A single input sample.
        tree : tuple or int
            Current node of the decision tree.

        Returns
        -------
        int
            Predicted class label.
        """
        if not isinstance(tree, tuple):
            return tree
        feature, threshold, left, right = tree
        return self._predict_sample(
            left if sample[feature] <= threshold else right,
            left if sample[feature] <= threshold else right,
        )

    def predict(self, X):
        """
        Predict class labels for input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels.

        Raises
        ------
        AttributeError
            If the model has not been fitted yet.
        """
        if self.tree is None:
            raise AttributeError("Model not fitted yet.")
        return np.array([self._predict_sample(x, self.tree) for x in X])

    def score(self, X, y):
        """
        Compute accuracy of the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        float
            Accuracy score.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class RandomForest:
    """
    A simple Random Forest classifier built on top of the custom DecisionTree.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the ensemble.
    max_depth : int or None, default=None
        Maximum depth for each tree.
    max_features : {'sqrt', None}, default='sqrt'
        Strategy for selecting feature subsets per tree.

    Attributes
    ----------
    trees : list of (DecisionTree, ndarray)
        A list of (tree, feature_indices) tuples.
    """

    def __init__(self, n_estimators=100, max_depth=None, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        """
        Fit the Random Forest model using bootstrap aggregation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Training labels.

        Returns
        -------
        self : RandomForest
            Fitted model.
        """
        num_samples, num_features = X.shape

        for _ in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(num_samples, size=num_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            # Feature subsampling
            if self.max_features == 'sqrt':
                features_indices = np.random.choice(
                    num_features, size=int(np.sqrt(num_features)), replace=False
                )
            else:
                features_indices = np.arange(num_features)

            # Train tree
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample[:, features_indices], y_sample)
            self.trees.append((tree, features_indices))

        return self

    def predict(self, X):
        """
        Predict class labels using majority vote across trees.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        predictions = np.zeros((X.shape[0], len(self.trees)))

        for i, (tree, features_indices) in enumerate(self.trees):
            predictions[:, i] = tree.predict(X[:, features_indices])

        return np.array([
            np.bincount(predictions[i].astype(int)).argmax()
            for i in range(predictions.shape[0])
        ])
