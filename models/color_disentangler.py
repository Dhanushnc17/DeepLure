"""
Adversarial Color Disentanglement Head.
Uses a Gradient Reversal Layer (GRL) to train the feature extractor to strip out
color cues while maintaining discriminative motif geometry.
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    In the forward pass, this acts as an identity.
    In the backward pass, it multiplies gradients by -alpha.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha: float = 0.2):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)


class AdversarialColorHead(nn.Module):
    """
    Discriminates the dominant color palette from the extracted embedding.
    With GRL, the backbone is penalized if color can be predicted from its embedding.
    """
    def __init__(self, embedding_dim: int, num_color_clusters: int = 8, grl_alpha: float = 0.2):
        super().__init__()
        self.grl = GradientReversalLayer(alpha=grl_alpha)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_color_clusters),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        reversed_features = self.grl(embedding)
        return self.mlp(reversed_features)
