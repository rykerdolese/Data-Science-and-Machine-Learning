# FILE: src/rice2025/postprocess.py
"""Postprocessing helpers: majority vote (classification) and average label (regression).
"""
from collections import Counter
from typing import Iterable


def majority_vote(labels: Iterable):
    if labels is None:
        raise ValueError('labels must not be None')
    labels = list(labels)
    if not labels:
        raise ValueError('labels must not be empty')
    counts = Counter(labels)
    most_common = counts.most_common()
    # tie-breaking: choose smallest label by sorted order to be deterministic
    top_count = most_common[0][1]
    candidates = [lab for lab, c in most_common if c == top_count]
    return sorted(candidates)[0]


def average_label(labels: Iterable):
    if labels is None:
        raise ValueError('labels must not be None')
    labels = list(labels)
    if not labels:
        raise ValueError('labels must not be empty')
    total = 0.0
    for x in labels:
        total += float(x)
    return total / len(labels)