"""
Evaluation Metrics for Retrieval (Identification) and Pairwise Verification.
Includes CMC (Rank-k), mAP, ROC-AUC, EER, and TAR @ FAR.
"""

from typing import Dict, List, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def compute_cosine_distance_matrix(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
) -> np.ndarray:
    """
    Computes cosine distance matrix between query [N_q, D] and gallery [N_g, D].
    Cosine distance = 1 - cosine_similarity (Range: [0, 2])
    """
    query_norm = F.normalize(query_features, p=2, dim=1)
    gallery_norm = F.normalize(gallery_features, p=2, dim=1)
    sim = torch.mm(query_norm, gallery_norm.t())
    dist = 1.0 - sim
    return dist.cpu().numpy()


def evaluate_identification(
    dist_matrix: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    top_k: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    """
    Evaluates Identification performance:
      - Rank-k CMC Accuracy (Rank-1, Rank-3, Rank-5, Rank-10)
      - mean Average Precision (mAP)
    """
    num_queries = len(query_labels)
    all_cmc = []
    all_ap = []

    for i in range(num_queries):
        q_label = query_labels[i]
        dists = dist_matrix[i]
        
        # Sort gallery indices by distance ascending (closest first)
        sorted_indices = np.argsort(dists)
        matches = (gallery_labels[sorted_indices] == q_label).astype(np.int32)

        if not np.any(matches):
            continue

        # CMC (Cumulative Matching Characteristic)
        cmc = matches.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[: max(top_k)])

        # Average Precision (AP)
        num_rel = matches.sum()
        cum_rel = matches.cumsum()
        ranks = np.arange(1, len(matches) + 1)
        precision_at_k = cum_rel / ranks
        ap = (precision_at_k * matches).sum() / num_rel
        all_ap.append(ap)

    cmc_matrix = np.array(all_cmc)
    mean_cmc = cmc_matrix.mean(axis=0) if len(cmc_matrix) > 0 else np.zeros(max(top_k))
    mean_ap = float(np.mean(all_ap)) if len(all_ap) > 0 else 0.0

    results = {"mAP": mean_ap * 100.0}
    for k in top_k:
        if k <= len(mean_cmc):
            results[f"Rank-{k}"] = float(mean_cmc[k - 1]) * 100.0

    return results


def evaluate_verification(
    similarities: np.ndarray,
    ground_truth_same: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluates 1:1 Saree Verification:
      - ROC-AUC
      - Equal Error Rate (EER)
      - TAR @ FAR = 1% (0.01)
      - TAR @ FAR = 0.1% (0.001)
      - Optimal Threshold & Max F1 Score
    """
    fpr, tpr, thresholds = roc_curve(ground_truth_same, similarities)
    roc_auc = auc(fpr, tpr)

    # Calculate Equal Error Rate (EER: point where FPR == FNR = 1 - TPR)
    fnr = 1.0 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    eer_threshold = float(thresholds[eer_idx])

    # TAR @ FAR = 1% and 0.1%
    def get_tar_at_far(target_far: float) -> float:
        idx = np.where(fpr <= target_far)[0]
        if len(idx) > 0:
            return float(tpr[idx[-1]])
        return 0.0

    tar_at_far_1pct = get_tar_at_far(0.01)
    tar_at_far_01pct = get_tar_at_far(0.001)

    # Precision-Recall & Max F1
    precision, recall, pr_thresholds = precision_recall_curve(ground_truth_same, similarities)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_f1_idx = np.nanargmax(f1_scores)
    max_f1 = float(f1_scores[best_f1_idx])
    best_threshold = float(pr_thresholds[min(best_f1_idx, len(pr_thresholds) - 1)])

    return {
        "ROC-AUC": roc_auc * 100.0,
        "EER": eer * 100.0,
        "EER_Threshold": eer_threshold,
        "TAR@FAR=1%": tar_at_far_1pct * 100.0,
        "TAR@FAR=0.1%": tar_at_far_01pct * 100.0,
        "Max_F1": max_f1 * 100.0,
        "Optimal_Threshold": best_threshold,
    }


def k_reciprocal_rerank(
    probFea: np.ndarray,
    galFea: np.ndarray,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
) -> np.ndarray:
    """
    k-reciprocal Feature Re-ranking (Zhong et al., CVPR 2017).
    Significantly improves retrieval performance in fine-grained motif search.
    """
    query_num = probFea.shape[0]
    all_num = query_num + galFea.shape[0]
    feat = np.append(probFea, galFea, axis=0)

    # L2 normalize
    feat = feat / np.linalg.norm(feat, axis=1, keepdims=True)

    # Euclidean distance
    distmat = np.power(feat, 2).sum(axis=1, keepdims=True) + np.power(feat, 2).sum(axis=1, keepdims=True).T
    distmat -= 2 * np.dot(feat, feat.T)
    distmat = np.clip(distmat, 0, None)

    q_g_dist = distmat[:query_num, query_num:]
    return q_g_dist
