"""
Visualization utilities for Saree Retrieval, Verification, and Color Invariance Analysis.
"""

from typing import List, Tuple, Optional
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps


def plot_retrieval_results(
    query_image_path: str,
    gallery_image_paths: List[str],
    similarity_scores: List[float],
    is_correct_match: List[bool],
    save_path: Optional[str] = None,
    top_k: int = 5,
):
    """
    Renders the query image alongside its top-k retrieved gallery candidates.
    Matches are bordered in green; mismatches in red.
    """
    k = min(top_k, len(gallery_image_paths))
    fig, axes = plt.subplots(1, k + 1, figsize=((k + 1) * 3, 3.5))

    # Plot Query
    q_img = Image.open(query_image_path).convert("RGB")
    axes[0].imshow(q_img)
    axes[0].set_title("QUERY", fontsize=11, fontweight="bold", color="blue")
    axes[0].axis("off")

    # Plot Matches
    for idx in range(k):
        ax = axes[idx + 1]
        g_img = Image.open(gallery_image_paths[idx]).convert("RGB")
        sim = similarity_scores[idx]
        correct = is_correct_match[idx]

        border_color = "green" if correct else "red"
        status = "MATCH" if correct else "DIFF"

        ax.imshow(g_img)
        ax.set_title(
            f"Rank {idx+1}: {status}\nSim: {sim:.3f}",
            fontsize=10,
            fontweight="bold",
            color=border_color,
        )
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        plt.close()
    else:
        plt.show()


def plot_evaluation_curves(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    cmc_ranks: List[int],
    cmc_scores: List[float],
    save_path: Optional[str] = None,
):
    """
    Plots side-by-side ROC curve (Verification) and CMC curve (Identification).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC Curve
    ax1.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.2f}%)")
    ax1.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Accept Rate (FAR / FPR)", fontsize=11)
    ax1.set_ylabel("True Accept Rate (TAR / TPR)", fontsize=11)
    ax1.set_title("Saree Verification ROC Curve", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    # CMC Curve
    ax2.plot(cmc_ranks, cmc_scores, marker="o", color="forestgreen", lw=2, label="CMC")
    ax2.set_xlabel("Rank", fontsize=11)
    ax2.set_ylabel("Identification Accuracy (%)", fontsize=11)
    ax2.set_title("Saree Identification CMC Curve", fontsize=12, fontweight="bold")
    ax2.set_xticks(cmc_ranks)
    ax2.set_ylim([0, 105])
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        plt.close()
    else:
        plt.show()
