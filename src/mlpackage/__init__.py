# Import models from submodules
from .supervised_learning.linear_regression import LinearRegression
from .supervised_learning.logistic_regression import LogisticRegression
from .supervised_learning.perceptron import Perceptron
from .supervised_learning.multilayer_perceptron import MLP
from .supervised_learning.knn import KNN
from .supervised_learning.decision_tree_classifier import DecisionTreeClassifier, RandomForestClassifier
from .supervised_learning.decision_tree_regressor import DecisionTreeRegressor, RandomForestRegressor
from .unsupervised_learning.kmeans import KMeans
from .unsupervised_learning.label_propogation import LabelPropagationCustom
from .unsupervised_learning.pca import PCA
from .unsupervised_learning.dbscan import DBSCAN

# Optional: define what shows up when someone does `from mlpackage import *`
__all__ = [
    "LinearRegression",
    "LogisticRegression",
    "Perceptron",
    "MLP",
    "KNN",
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "DecisionTreeRegressor",
    "RandomForestRegressor",
    "KMeans",
    "LabelPropagationCustom",
    "PCA",
    "DBSCAN"
]