"""
Contrastive Loss implementations for Metric Learning.
Includes Supervised Contrastive Loss (SupCon) and Batch-Hard Triplet Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss.
    Ref: Khosla et al. "Supervised Contrastive Learning" (NeurIPS 2020).
    Pulls embeddings with the same motif together regardless of colorway,
    and pushes embeddings of different motifs apart.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: L2-normalized feature vectors, shape [B, D]
            labels: Ground truth motif class labels, shape [B]
        """
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)

        mask = torch.eq(labels, labels.T).float().to(device)

        # Compute cosine similarity matrix
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature,
        )

        # Numerical stability trick: subtract max per row
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask out self-contrast (diagonal)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # Compute log-probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        # Loss
        loss = -mean_log_prob_pos
        return loss.mean()


class BatchHardTripletLoss(nn.Module):
    """
    Triplet Loss with online Batch-Hard mining (Hermans et al., 2017).
    For each anchor, finds the hardest positive (furthest distance)
    and hardest negative (closest distance).
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Pairwise euclidean distance matrix
        dist_mat = torch.cdist(embeddings, embeddings, p=2)

        # Masks
        is_pos = labels.unsqueeze(0) == labels.unsqueeze(1)
        is_neg = labels.unsqueeze(0) != labels.unsqueeze(1)

        # Hardest positive: max distance among positives
        dist_ap = (dist_mat * is_pos.float()).max(dim=1)[0]

        # Hardest negative: min distance among negatives
        max_dist = dist_mat.max()
        dist_an = (dist_mat + is_pos.float() * max_dist).min(dim=1)[0]

        # Triplet loss
        loss = F.relu(dist_ap - dist_an + self.margin)
        return loss.mean()
