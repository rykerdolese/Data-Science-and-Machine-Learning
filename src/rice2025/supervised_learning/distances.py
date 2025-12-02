# FILE: src/rice2025/metrics.py
"""Distance metric implementations: euclidean and manhattan.
"""
from typing import Iterable
import math

__all__ = ["euclidean", "manhattan"]

# underscore prefix to indicate internal use
def _validate_pair(a: Iterable, b: Iterable):
    if a is None or b is None:
        raise ValueError("Inputs must not be None")
    if len(a) != len(b):
        raise ValueError("Inputs must have same length")


def euclidean(a, b):
    _validate_pair(a, b)
    s = 0.0
    for x, y in zip(a, b):
        s += (float(x) - float(y)) ** 2
    return math.sqrt(s)


def manhattan(a, b):
    _validate_pair(a, b)
    s = 0.0
    for x, y in zip(a, b):
        s += abs(float(x) - float(y))
    return s