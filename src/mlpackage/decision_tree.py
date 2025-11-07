import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Define basic decision tree class

import numpy as np

class DecisionTree:
    """
    A simple Decision Tree classifier using information gain and entropy.

    Parameters
    ----------
    max_depth : int or None, default=None
        Maximum depth of the tree. If None, the tree grows until pure leaves.

    Attributes
    ----------
    tree : tuple or int
        The learned tree structure.
    """

    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        """
        Fit the decision tree classifier to the training data.
        """
        if X.size == 0 or y.size == 0:
            raise ValueError("Empty X or y provided.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must match.")
        self.tree = self._build_tree(X, y)
        return self

    def _entropy(self, y):
        """Compute entropy of label distribution."""
        counts = np.bincount(y)
        probabilities = counts[counts > 0] / len(y)
        return -np.sum(probabilities * np.log2(probabilities))

    def _information_gain(self, parent_y, left_y, right_y):
        """Compute information gain from a proposed split."""
        p_left = len(left_y) / len(parent_y)
        p_right = 1 - p_left
        return self._entropy(parent_y) - (p_left * self._entropy(left_y) + p_right * self._entropy(right_y))

    def _best_split(self, X, y):
        """Find the best split for a given dataset."""
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
        """Recursively build the tree structure."""
        unique_classes, counts = np.unique(y, return_counts=True)
        if len(unique_classes) == 1 or (self.max_depth is not None and depth >= self.max_depth):
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
        """Predict a single sample recursively."""
        if not isinstance(tree, tuple):
            return tree
        feature, threshold, left, right = tree
        return self._predict_sample(left if sample[feature] <= threshold else right, left if sample[feature] <= threshold else right)

    def predict(self, X):
        """Predict labels for input samples."""
        if self.tree is None:
            raise AttributeError("Model not fitted yet.")
        return np.array([self._predict_sample(x, self.tree) for x in X])

    def score(self, X, y):
        """Return accuracy score."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


# Now we can create our own Random Forest implementation using the DecisionTree class
class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        num_samples, num_features = X.shape
        for _ in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(num_samples, size=num_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            
            # Randomly select features
            if self.max_features == 'sqrt':
                features_indices = np.random.choice(num_features, size=int(np.sqrt(num_features)), replace=False)
            else:
                features_indices = np.arange(num_features)
            
            # Fit a decision tree on the sampled data and selected features
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample[:, features_indices], y_sample)
            self.trees.append((tree, features_indices))

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.trees)))
        for i, (tree, features_indices) in enumerate(self.trees):
            predictions[:, i] = tree.predict(X[:, features_indices])
        
        # Majority vote
        return np.array([np.bincount(predictions[i].astype(int)).argmax() for i in range(predictions.shape[0])])
