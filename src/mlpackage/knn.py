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
