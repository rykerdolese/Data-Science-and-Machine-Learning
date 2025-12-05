# Import models from submodules
from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression
from .perceptron import Perceptron
from .multilayer_perceptron import MLP
from .knn import KNN
from .decision_tree import DecisionTree, RandomForest
from .decision_tree_regressor import DecisionTreeRegressor, RandomForestRegressor
from .kmeans import KMeans
from .label_propogation import LabelPropagationCustom

# Optional: define what shows up when someone does `from mlpackage import *`
__all__ = [
    "LinearRegression",
    "LogisticRegression",
    "Perceptron",
    "MLP",
    "KNN",
    "DecisionTree",
    "RandomForest",
    "DecisionTreeRegressor",
    "RandomForestRegressor",
    "KMeans",
    "LabelPropagationCustom"
]