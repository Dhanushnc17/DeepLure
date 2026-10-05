"""
Backbone loader supporting modern CNNs and Vision Transformers via timm.
Optimized for fine-grained textile and motif feature extraction.
"""

from typing import Tuple
import torch
import torch.nn as nn
import timm


class FeatureBackbone(nn.Module):
    """
    Wraps timm models to extract 2D spatial feature maps before pooling.
    Supported backbones:
      - 'convnext_nano.in12k_ft_in1k' (74M MACs, ~15M params, exceptional spatial motif representation)
      - 'convnext_tiny.in12k_ft_in1k' (~28M params)
      - 'efficientnet_b0' (5.3M params, ultra-efficient)
      - 'mobilenetv4_conv_small.e1200_r224_in1k' (3.8M params, ultra-fast edge)
      - 'resnet50' (standard reference)
    """

    def __init__(self, backbone_name: str = "convnext_nano.in12k_ft_in1k", pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone_name

        try:
            # Load feature extractor without classification head
            self.model = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0,  # Remove classifier
                features_only=False,
            )
            # Find feature dimension
            if hasattr(self.model, "num_features"):
                self.feature_dim = self.model.num_features
            elif hasattr(self.model, "head") and hasattr(self.model.head, "fc"):
                self.feature_dim = self.model.head.fc.in_features
            else:
                # Probe with dummy tensor
                dummy = torch.randn(1, 3, 224, 224)
                feat = self.model.forward_features(dummy)
                if isinstance(feat, torch.Tensor):
                    self.feature_dim = feat.shape[1] if feat.ndim == 4 else feat.shape[-1]
                else:
                    self.feature_dim = 512
        except Exception as e:
            # Fallback to standard torchvision-style simple ConvNeXt/ResNet if offline or timm missing weights
            print(f"[Warning] Error loading {backbone_name}: {e}. Falling back to default resnet18/efficientnet_b0.")
            self.model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
            self.feature_dim = self.model.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts spatial feature maps [B, C, H, W] from input [B, 3, H, W].
        """
        feat = self.model.forward_features(x)
        # If backbone returns token sequence [B, N, C] (e.g. ViT), permute to [B, C, H, W]
        if feat.ndim == 3:
            B, N, C = feat.shape
            # If class token exists (N = 1 + H*W)
            if (N - 1) > 0 and int((N - 1) ** 0.5) ** 2 == (N - 1):
                feat = feat[:, 1:, :]  # drop cls token
                H = W = int((N - 1) ** 0.5)
            else:
                H = W = int(N ** 0.5)
            feat = feat.transpose(1, 2).reshape(B, C, H, W)
        return feat
