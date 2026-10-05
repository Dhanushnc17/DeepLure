"""
Unit tests for evaluation metrics (CMC, mAP, ROC-AUC, EER).
"""

import numpy as np
import pytest
from utils.metrics import evaluate_identification, evaluate_verification


def test_evaluate_identification():
    # 4 queries, 5 gallery items
    # Query 0: class 0, Gallery matches: [class 0, class 1, class 2] -> Rank 1 match
    # Query 1: class 1, Gallery matches: [class 2, class 1, class 0] -> Rank 2 match
    dist_matrix = np.array([
        [0.1, 0.5, 0.9, 0.8],
        [0.8, 0.2, 0.3, 0.9],
    ])
    query_labels = np.array([0, 1])
    gallery_labels = np.array([0, 1, 1, 2])

    results = evaluate_identification(dist_matrix, query_labels, gallery_labels, top_k=[1, 2])
    assert "Rank-1" in results
    assert "mAP" in results
    assert results["Rank-1"] == 100.0


def test_evaluate_verification():
    sims = np.array([0.9, 0.85, 0.2, 0.15, 0.95, 0.1])
    gt = np.array([1, 1, 0, 0, 1, 0])

    results = evaluate_verification(sims, gt)
    assert results["ROC-AUC"] == 100.0
    assert results["EER"] == 0.0
    assert results["Max_F1"] == 100.0
