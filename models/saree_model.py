"""
End-to-End Color-Invariant Saree Recognition Model.
Combines Feature Backbone + GeM Pooling + Projection + Margin Head + Color Disentangler.
"""

from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import FeatureBackbone
from models.head import GeM, ArcMarginProduct, SubCenterArcMarginProduct
from models.color_disentangler import AdversarialColorHead


class ColorInvariantSareeNet(nn.Module):
    """
    Complete end-to-end model for saree motif identification & verification.
    """

    def __init__(
        self,
        backbone_name: str = "convnext_nano.in12k_ft_in1k",
        num_classes: int = 100,
        embedding_dim: int = 256,
        pretrained: bool = True,
        pooling: str = "gem",
        gem_p: float = 3.0,
        gem_trainable: bool = True,
        dropout: float = 0.1,
        loss_type: str = "subcenter_arcface",
        margin_s: float = 30.0,
        margin_m: float = 0.40,
        sub_centers: int = 3,
        use_color_disentanglement: bool = True,
        num_color_clusters: int = 8,
        grl_alpha: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.use_color_disentanglement = use_color_disentanglement
        self.loss_type = loss_type

        # 1. Feature Backbone
        self.backbone = FeatureBackbone(backbone_name, pretrained=pretrained)
        in_features = self.backbone.feature_dim

        # 2. Pooling Layer
        if pooling.lower() == "gem":
            self.pooling = GeM(p=gem_p, trainable=gem_trainable)
        elif pooling.lower() == "max":
            self.pooling = nn.AdaptiveMaxPool2d((1, 1))
        else:
            self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        # 3. Projection & Embedding Bottleneck
        self.neck = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(in_features, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )

        # 4. Metric Learning Margin Head
        if loss_type == "subcenter_arcface":
            self.head = SubCenterArcMarginProduct(
                in_features=embedding_dim,
                out_features=num_classes,
                sub_centers=sub_centers,
                s=margin_s,
                m=margin_m,
            )
        elif loss_type == "arcface":
            self.head = ArcMarginProduct(
                in_features=embedding_dim,
                out_features=num_classes,
                s=margin_s,
                m=margin_m,
            )
        else:
            self.head = nn.Linear(embedding_dim, num_classes, bias=False)

        # 5. Optional Adversarial Color Head
        if use_color_disentanglement:
            self.color_head = AdversarialColorHead(
                embedding_dim=embedding_dim,
                num_color_clusters=num_color_clusters,
                grl_alpha=grl_alpha,
            )
        else:
            self.color_head = None

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts L2-normalized embedding representation for inference/retrieval.
        Shape: [B, embedding_dim]
        """
        feats = self.backbone(x)
        pooled = self.pooling(feats).flatten(1)
        emb = self.neck(pooled)
        return F.normalize(emb, p=2, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training & inference.
        Returns:
            dict containing:
              - 'embedding': [B, embedding_dim] (L2 normalized)
              - 'logits': [B, num_classes] (Angular margin logits if labels provided)
              - 'color_logits': [B, num_color_clusters] (if color disentanglement enabled)
        """
        feats = self.backbone(x)
        pooled = self.pooling(feats).flatten(1)
        raw_emb = self.neck(pooled)
        norm_emb = F.normalize(raw_emb, p=2, dim=1)

        output = {"embedding": norm_emb}

        if labels is not None and self.head is not None:
            if isinstance(self.head, (ArcMarginProduct, SubCenterArcMarginProduct)):
                output["logits"] = self.head(raw_emb, labels)
            else:
                output["logits"] = self.head(norm_emb)

        if self.color_head is not None and self.training:
            output["color_logits"] = self.color_head(raw_emb)

        return output
