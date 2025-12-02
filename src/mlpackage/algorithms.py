import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        # Ensure numpy arrays
        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y, dtype=int)

    def predict(self, X):
        
        predictions = []

        for i in range(X.shape[0]):
            x = X[i]  # no iloc needed now
            distances = np.linalg.norm(self.X_train - x, axis=1)
            #print(distances)
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]

            most_common = np.bincount(k_nearest_labels).argmax()
            predictions.append(most_common)

        return np.array(predictions)
    
    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def confusion_matrix(self, X, y):
        predictions = self.predict(X)
        unique_labels = np.unique(y)
        matrix = pd.DataFrame(0, index=unique_labels, columns=unique_labels)
        for true_label, pred_label in zip(y, predictions):
            matrix.loc[true_label, pred_label] += 1
        return matrix
    
    def draw_decision_boundary(self, X, y):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Decision Boundary')
        plt.show()

# Define basic decision tree class
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)
        print("Decision Tree built successfully.")

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        unique_classes, class_counts = np.unique(y, return_counts=True)
        
        # If all samples belong to the same class or max depth reached
        if len(unique_classes) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            return unique_classes[np.argmax(class_counts)]
        
        # Find the best split
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            return unique_classes[np.argmax(class_counts)]
        
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold
        
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return (best_feature, best_threshold, left_subtree, right_subtree)

    def _best_split(self, X, y):
        num_samples, num_features = X.shape
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold
                
                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue
                
                gain = self._information_gain(y, y[left_indices], y[right_indices])

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold

    def _information_gain(self, parent_y, left_y, right_y):
        p_left = len(left_y) / len(parent_y)
        p_right = len(right_y) / len(parent_y)
        
        gain = self._entropy(parent_y) - (p_left * self._entropy(left_y) + p_right * self._entropy(right_y))

        return gain

    def _entropy(self, y):
        class_counts = np.bincount(y)
        probabilities = class_counts / len(y)
        return -np.sum(probabilities[probabilities > 0] * np.log2(probabilities[probabilities > 0]))

    def predict(self, X):
        predictions = [self._predict_sample(sample, self.tree) for sample in X]
        return np.array(predictions)

    def _predict_sample(self, sample, tree):
        if not isinstance(tree, tuple):
            return tree
        
        feature, threshold, left_subtree, right_subtree = tree
        
        if sample[feature] <= threshold:
            return self._predict_sample(sample, left_subtree)
        else:
            return self._predict_sample(sample, right_subtree)

    def print_tree(self, tree=None, depth=0):
        if tree is None:
            tree = self.tree
        
        if isinstance(tree, tuple):
            feature, threshold, left_subtree, right_subtree = tree
            print(f"{' ' * depth * 2}Feature {feature} <= {threshold}:")
            self.print_tree(left_subtree, depth + 1)
            print(f"{' ' * depth * 2}Feature {feature} > {threshold}:")
            self.print_tree(right_subtree, depth + 1)
        else:
            print(f"{' ' * depth * 2}Class: {tree}")

class LinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        # Add a bias term (intercept)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # add x0 = 1 to each instance
        # Normal equation: theta_best = (X_b^T * X_b)^-1 * X_b^T * y
        self.coefficients = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept = self.coefficients[0]
        self.coefficients = self.coefficients[1:]

    def predict(self, X):
        return np.dot(X, self.coefficients) + self.intercept
    
    def rmse(self, X, y):
        y_pred = self.predict(X)
        return np.sqrt(np.mean((y - y_pred) ** 2))
    
    def R_squared(self, X, y):
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)


