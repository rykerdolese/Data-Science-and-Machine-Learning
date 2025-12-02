# FILE: src/rice2025/__init__.py
"""Package initializer for rice2025.
Exports convenience names for the public API used in tests.
"""
from .preprocess import normalize, scale_minmax, train_test_split
from .metrics import euclidean, manhattan
from .knn import KNN
from .postprocess import majority_vote, average_label
from .supervised_learning import *


__all__ = [
"normalize",
"scale_minmax",
"train_test_split",
"euclidean",
"manhattan",
"KNN",
"majority_vote",
"average_label",
]