"""
Pooling layers and Metric Learning Angular Margin Heads.
Includes Generalized Mean (GeM) Pooling, ArcFace, SubCenter ArcFace, and CosFace.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    """
    Generalized Mean (GeM) Pooling.
    Computes f(x) = (1/|X| sum_{x in X} x^p )^(1/p)
    Enables network to focus on salient geometric motifs / localized weave patterns.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6, trainable: bool = True):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p, requires_grad=trainable)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clamp inputs above eps for numerical stability with power p
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1)),
        ).pow(1.0 / self.p)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class ArcMarginProduct(nn.Module):
    """
    Standard ArcFace (Additive Angular Margin) Head.
    Ref: Deng et al. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition" (CVPR 2019)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 30.0,
        m: float = 0.40,
        easy_margin: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input_embeddings: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # L2-normalize features and weights
        cosine = F.linear(F.normalize(input_embeddings), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=input_embeddings.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class SubCenterArcMarginProduct(nn.Module):
    """
    SubCenter ArcFace Head.
    Ref: Deng et al. "Sub-center ArcFace: Boosting Face Recognition by Large-scale Noisy Web Faces" (ECCV 2020)
    Allows each textile design class to have K sub-centers on the hypersphere,
    accommodating distinct colorways / drape perspectives per motif.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sub_centers: int = 3,
        s: float = 30.0,
        m: float = 0.40,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sub_centers = sub_centers
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features * sub_centers, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input_embeddings: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # input_embeddings: [B, D]
        # self.weight: [C * K, D]
        norm_feat = F.normalize(input_embeddings)
        norm_weight = F.normalize(self.weight)
        cosine = F.linear(norm_feat, norm_weight)  # [B, C * K]
        
        # Reshape to [B, C, K] and take max over sub-centers
        B = cosine.size(0)
        cosine = cosine.view(B, self.out_features, self.sub_centers)
        cosine, _ = torch.max(cosine, dim=2)  # [B, C]

        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=input_embeddings.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output
