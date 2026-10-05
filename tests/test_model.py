"""
Unit tests for ColorInvariantSareeNet and metric learning heads.
"""

import pytest
import torch
from models.saree_model import ColorInvariantSareeNet
from models.head import GeM, ArcMarginProduct, SubCenterArcMarginProduct
from models.color_disentangler import GradientReversalLayer, AdversarialColorHead


def test_gem_pooling():
    x = torch.rand(4, 64, 16, 16)
    gem = GeM(p=3.0)
    out = gem(x)
    assert out.shape == (4, 64, 1, 1)


def test_arc_margin_product():
    head = ArcMarginProduct(in_features=128, out_features=10, s=30.0, m=0.40)
    emb = torch.randn(8, 128)
    labels = torch.randint(0, 10, (8,))
    logits = head(emb, labels)
    assert logits.shape == (8, 10)


def test_subcenter_arcface():
    head = SubCenterArcMarginProduct(in_features=128, out_features=10, sub_centers=3, s=30.0, m=0.40)
    emb = torch.randn(8, 128)
    labels = torch.randint(0, 10, (8,))
    logits = head(emb, labels)
    assert logits.shape == (8, 10)


def test_grl_backward_pass():
    grl = GradientReversalLayer(alpha=0.5)
    x = torch.ones(4, 16, requires_grad=True)
    out = grl(x) * 2.0
    loss = out.sum()
    loss.backward()
    # Backward pass multiplies gradient by -alpha (-0.5 * 2.0 = -1.0)
    assert torch.allclose(x.grad, torch.ones(4, 16) * -1.0)


def test_color_invariant_saree_net_forward():
    model = ColorInvariantSareeNet(
        backbone_name="efficientnet_b0",
        num_classes=20,
        embedding_dim=128,
        pretrained=False,
        use_color_disentanglement=True,
        num_color_clusters=6,
    )
    x = torch.randn(4, 3, 224, 224)
    labels = torch.tensor([0, 1, 2, 3])

    # 1. Feature extraction
    emb = model.extract_features(x)
    assert emb.shape == (4, 128)
    # Check L2 normalization (norm should be 1.0)
    norms = torch.norm(emb, p=2, dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    # 2. Training forward pass
    model.train()
    out = model(x, labels=labels)
    assert "embedding" in out
    assert "logits" in out
    assert "color_logits" in out
    assert out["logits"].shape == (4, 20)
    assert out["color_logits"].shape == (4, 6)
