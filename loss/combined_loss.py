"""
Unified Multi-Task Objective for Color-Invariant Motif Learning.
Combines Angular Margin Loss + Supervised Contrastive Loss + Adversarial Color Regularization.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.arcface_loss import LabelSmoothingCrossEntropy
from loss.contrastive_loss import SupConLoss


class CombinedSareeLoss(nn.Module):
    """
    Multi-objective loss function:
      L_total = L_margin(motif) + lambda_supcon * L_supcon(motif) + lambda_adv * L_color(palette)
    """

    def __init__(
        self,
        label_smoothing: float = 0.05,
        use_supcon: bool = True,
        supcon_weight: float = 0.5,
        supcon_temp: float = 0.07,
        use_adversarial: bool = True,
        adversarial_weight: float = 0.15,
    ):
        super().__init__()
        self.margin_criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        self.use_supcon = use_supcon
        self.supcon_weight = supcon_weight
        self.supcon_criterion = SupConLoss(temperature=supcon_temp) if use_supcon else None

        self.use_adversarial = use_adversarial
        self.adversarial_weight = adversarial_weight
        self.color_criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        motif_labels: torch.Tensor,
        color_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes total combined loss and individual components for metrics tracking.
        """
        metrics = {}
        total_loss = torch.tensor(0.0, device=motif_labels.device)

        # 1. Angular Margin Classification Loss
        if "logits" in model_outputs:
            l_margin = self.margin_criterion(model_outputs["logits"], motif_labels)
            total_loss = total_loss + l_margin
            metrics["loss_margin"] = l_margin.item()

        # 2. Supervised Contrastive Loss on L2 embeddings
        if self.use_supcon and "embedding" in model_outputs:
            l_supcon = self.supcon_criterion(model_outputs["embedding"], motif_labels)
            total_loss = total_loss + self.supcon_weight * l_supcon
            metrics["loss_supcon"] = l_supcon.item()

        # 3. Adversarial Color Disentanglement Loss
        if (
            self.use_adversarial
            and "color_logits" in model_outputs
            and color_labels is not None
            and (color_labels >= 0).any()
        ):
            # Only calculate for valid color labels
            valid_mask = color_labels >= 0
            if valid_mask.sum() > 0:
                l_color = self.color_criterion(
                    model_outputs["color_logits"][valid_mask],
                    color_labels[valid_mask],
                )
                total_loss = total_loss + self.adversarial_weight * l_color
                metrics["loss_color_adv"] = l_color.item()

        metrics["loss_total"] = total_loss.item()
        return total_loss, metrics
