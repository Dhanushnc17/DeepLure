"""
Standalone Evaluation Script for Identification and Verification.
Usage:
    python scripts/evaluate.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth
"""

import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.augmentations import get_val_transforms
from data.dataset import build_datasets_from_folder
from models.saree_model import ColorInvariantSareeNet
from engine.evaluator import SareeEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate Color-Invariant Saree Recognition Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to checkpoint")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data directory")
    parser.add_argument("--reranking", action="store_true", help="Apply k-reciprocal re-ranking")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data_dir = args.data_dir if args.data_dir else cfg["data"]["data_dir"]
    device = cfg["project"]["device"] if torch.cuda.is_available() else "cpu"

    val_transform = get_val_transforms(image_size=tuple(cfg["data"]["image_size"]))

    train_ds, gallery_ds, query_ds = build_datasets_from_folder(
        data_dir=data_dir,
        train_transform=val_transform,
        val_transform=val_transform,
        val_ratio=cfg["data"]["split"]["val_ratio"],
        query_ratio=cfg["data"]["split"]["query_ratio"],
        seed=cfg["project"]["seed"],
    )

    gallery_loader = DataLoader(gallery_ds, batch_size=cfg["evaluation"]["batch_size"], shuffle=False)
    query_loader = DataLoader(query_ds, batch_size=cfg["evaluation"]["batch_size"], shuffle=False)

    # Initialize model
    model = ColorInvariantSareeNet(
        backbone_name=cfg["model"]["backbone"],
        num_classes=train_ds.num_classes,
        embedding_dim=cfg["model"]["embedding_dim"],
        pretrained=False,
        pooling=cfg["model"]["pooling"],
        gem_p=cfg["model"]["gem_p"],
        gem_trainable=False,
        loss_type=cfg["loss"]["type"],
        use_color_disentanglement=False,
    )

    # Load weights
    if os.path.exists(args.checkpoint):
        checkpoint_data = torch.load(args.checkpoint, map_location=device)
        if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
            model.load_state_dict(checkpoint_data["model_state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint_data, strict=False)
        print(f"[+] Loaded weights from checkpoint: {args.checkpoint}")
    else:
        print(f"[!] Checkpoint '{args.checkpoint}' not found. Evaluating with initialized backbone...")

    evaluator = SareeEvaluator(model=model, device=device)
    results = evaluator.evaluate(
        gallery_loader=gallery_loader,
        query_loader=query_loader,
        top_k=cfg["evaluation"]["top_k"],
        use_reranking=args.reranking or cfg["evaluation"]["re_ranking"],
    )

    print("\n" + "=" * 60)
    print("        DEEPLURE EVALUATION REPORT (IDENTIFICATION & VERIFICATION)")
    print("=" * 60)
    print("\n--- 1. IDENTIFICATION (1:N Retrieval) ---")
    print(f"  • mean Average Precision (mAP) : {results['identification']['mAP']:.2f}%")
    for k in cfg["evaluation"]["top_k"]:
        key = f"Rank-{k}"
        if key in results["identification"]:
            print(f"  • Cumulative Match {key:<8} : {results['identification'][key]:.2f}%")

    print("\n--- 2. VERIFICATION (1:1 Pair Matching) ---")
    print(f"  • ROC-AUC                    : {results['verification']['ROC-AUC']:.2f}%")
    print(f"  • Equal Error Rate (EER)     : {results['verification']['EER']:.2f}% (Threshold: {results['verification']['EER_Threshold']:.4f})")
    print(f"  • TAR @ FAR = 1.0%           : {results['verification']['TAR@FAR=1%']:.2f}%")
    print(f"  • TAR @ FAR = 0.1%           : {results['verification']['TAR@FAR=0.1%']:.2f}%")
    print(f"  • Max F1 Score               : {results['verification']['Max_F1']:.2f}% (Optimal Thresh: {results['verification']['Optimal_Threshold']:.4f})")

    print("\n--- 3. DATASET SPLIT DETAILS ---")
    print(f"  • Query Images Evaluated     : {results['counts']['num_query']}")
    print(f"  • Gallery Images Indexed     : {results['counts']['num_gallery']}")
    print(f"  • Total Pair Comparisons     : {results['counts']['num_pairs_evaluated']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
