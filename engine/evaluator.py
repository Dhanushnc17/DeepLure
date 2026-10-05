"""
Evaluation Engine for Saree Retrieval (Identification) and Verification.
Generates metrics for gallery-query retrieval and pairwise verification.
"""

from typing import Dict, Tuple, List, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from utils.metrics import (
    compute_cosine_distance_matrix,
    evaluate_identification,
    evaluate_verification,
    k_reciprocal_rerank,
)


class SareeEvaluator:
    """
    Evaluates Identification (1:N search) and Verification (1:1 pair matching).
    """

    def __init__(self, model: torch.nn.Module, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.device = device

    @torch.no_grad()
    def extract_dataset_features(
        self,
        loader: DataLoader,
    ) -> Tuple[torch.Tensor, np.ndarray, List[str]]:
        """
        Extracts L2-normalized feature embeddings from a DataLoader.
        Returns:
            features: [N, D] torch tensor
            labels: [N] numpy array
            paths: list of image filepaths
        """
        self.model.eval()
        features_list = []
        labels_list = []
        paths_list = []

        for batch in tqdm(loader, desc="Extracting features", leave=False):
            if len(batch) >= 3:
                images, labels, *rest = batch
                paths = rest[-1] if isinstance(rest[-1], (list, tuple)) else []
            else:
                images, labels = batch
                paths = []

            images = images.to(self.device)
            emb = self.model.extract_features(images)

            features_list.append(emb.cpu())
            labels_list.append(labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels))
            if paths:
                paths_list.extend(paths)

        all_features = torch.cat(features_list, dim=0)
        all_labels = np.concatenate(labels_list, axis=0)
        return all_features, all_labels, paths_list

    def evaluate(
        self,
        gallery_loader: DataLoader,
        query_loader: DataLoader,
        top_k: List[int] = [1, 3, 5, 10],
        use_reranking: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """
        Full evaluation pipeline on Query vs Gallery split.
        """
        # 1. Extract Embeddings
        gal_feats, gal_labels, gal_paths = self.extract_dataset_features(gallery_loader)
        qry_feats, qry_labels, qry_paths = self.extract_dataset_features(query_loader)

        # 2. Identification (1:N Retrieval)
        if use_reranking:
            dist_mat = k_reciprocal_rerank(qry_feats.numpy(), gal_feats.numpy())
        else:
            dist_mat = compute_cosine_distance_matrix(qry_feats, gal_feats)

        ident_metrics = evaluate_identification(
            dist_matrix=dist_mat,
            query_labels=qry_labels,
            gallery_labels=gal_labels,
            top_k=top_k,
        )

        # 3. Verification (1:1 Pairwise matching from Query vs Gallery combinations)
        # Cosine similarity matrix: sim = 1.0 - dist
        sim_mat = 1.0 - compute_cosine_distance_matrix(qry_feats, gal_feats)
        # Flatten pairs
        sims_flat = sim_mat.flatten()
        # Binary target: 1 if query and gallery share same motif design_id, else 0
        ground_truth = (qry_labels[:, None] == gal_labels[None, :]).flatten().astype(np.int32)

        verif_metrics = evaluate_verification(
            similarities=sims_flat,
            ground_truth_same=ground_truth,
        )

        return {
            "identification": ident_metrics,
            "verification": verif_metrics,
            "counts": {
                "num_query": len(qry_labels),
                "num_gallery": len(gal_labels),
                "num_pairs_evaluated": len(ground_truth),
                "num_positive_pairs": int(ground_truth.sum()),
                "num_negative_pairs": int((1 - ground_truth).sum()),
            },
        }
