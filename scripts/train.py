"""
Master Training Script for Color-Invariant Saree Recognition.
Usage:
    python scripts/train.py --config configs/config.yaml
"""

import os
import sys
import argparse
import random
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.augmentations import get_train_transforms, get_val_transforms
from data.dataset import build_datasets_from_folder
from data.sampler import PKBalancedSampler
from models.saree_model import ColorInvariantSareeNet
from loss.combined_loss import CombinedSareeLoss
from engine.trainer import SareeTrainer
from scripts.prepare_data import generate_synthetic_saree_dataset


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description="Train Color-Invariant Saree Recognition Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--data_dir", type=str, default=None, help="Override dataset directory")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    # 1. Load Configuration
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.data_dir:
        cfg["data"]["data_dir"] = args.data_dir
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size

    set_seed(cfg["project"]["seed"])
    device = cfg["project"]["device"] if torch.cuda.is_available() else "cpu"
    print(f"[*] Training on device: {device.upper()}")

    # 2. Check Data
    data_dir = cfg["data"]["data_dir"]
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        print(f"[!] Data directory '{data_dir}' not found or empty.")
        print("[*] Generating synthetic multi-colorway dataset for demonstration...")
        generate_synthetic_saree_dataset(output_dir=data_dir, num_designs=25, colorways_per_design=6)

    # 3. Build Datasets & Transforms
    img_size = tuple(cfg["data"]["image_size"])
    train_transform = get_train_transforms(
        image_size=img_size,
        color_jitter_params=cfg["data"]["augmentation"]["color_jitter"],
        p_gray=cfg["data"]["augmentation"]["random_grayscale"]["p"],
    )
    val_transform = get_val_transforms(image_size=img_size)

    use_adv = cfg["model"]["adversarial_color_disentanglement"]["enabled"]
    num_color_clusters = cfg["model"]["adversarial_color_disentanglement"]["num_color_clusters"]

    train_ds, gallery_ds, query_ds = build_datasets_from_folder(
        data_dir=data_dir,
        train_transform=train_transform,
        val_transform=val_transform,
        val_ratio=cfg["data"]["split"]["val_ratio"],
        query_ratio=cfg["data"]["split"]["query_ratio"],
        seed=cfg["project"]["seed"],
        return_color_label=use_adv,
        num_color_clusters=num_color_clusters,
    )

    print(f"[+] Dataset Split Summary:")
    print(f"    - Training Samples: {len(train_ds)} (Classes: {train_ds.num_classes})")
    print(f"    - Gallery Samples:  {len(gallery_ds)}")
    print(f"    - Query Samples:    {len(query_ds)}")

    # 4. DataLoaders & Samplers
    if cfg["train"]["sampler"]["use_pk_sampler"]:
        p = min(cfg["train"]["sampler"]["p"], train_ds.num_classes)
        k = cfg["train"]["sampler"]["k"]
        batch_sampler = PKBalancedSampler(train_ds, p=p, k=k, length=max(20, len(train_ds) // (p * k)))
        train_loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=cfg["train"]["num_workers"] if os.name != "nt" else 0,
            pin_memory=(device.startswith("cuda")),
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["train"]["batch_size"],
            shuffle=True,
            num_workers=cfg["train"]["num_workers"] if os.name != "nt" else 0,
            pin_memory=(device.startswith("cuda")),
            drop_last=True,
        )

    gallery_loader = DataLoader(
        gallery_ds,
        batch_size=cfg["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"] if os.name != "nt" else 0,
    )
    query_loader = DataLoader(
        query_ds,
        batch_size=cfg["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"] if os.name != "nt" else 0,
    )

    # 5. Build Model
    model = ColorInvariantSareeNet(
        backbone_name=cfg["model"]["backbone"],
        num_classes=train_ds.num_classes,
        embedding_dim=cfg["model"]["embedding_dim"],
        pretrained=cfg["model"]["pretrained"],
        pooling=cfg["model"]["pooling"],
        gem_p=cfg["model"]["gem_p"],
        gem_trainable=cfg["model"]["gem_trainable"],
        dropout=cfg["model"]["dropout"],
        loss_type=cfg["loss"]["type"],
        margin_s=cfg["loss"]["margin_s"],
        margin_m=cfg["loss"]["margin_m"],
        sub_centers=cfg["loss"]["sub_centers"],
        use_color_disentanglement=use_adv,
        num_color_clusters=num_color_clusters,
        grl_alpha=cfg["model"]["adversarial_color_disentanglement"]["grl_alpha"],
    )

    # 6. Loss & Optimizer
    criterion = CombinedSareeLoss(
        label_smoothing=cfg["loss"]["label_smoothing"],
        use_supcon=True,
        supcon_weight=0.5,
        use_adversarial=use_adv,
        adversarial_weight=cfg["loss"]["adversarial_weight"],
    )

    # Differential Learning Rates (lower for backbone, higher for neck & heads)
    backbone_lr = cfg["train"]["optimizer"]["lr"] * cfg["train"]["optimizer"]["backbone_lr_ratio"]
    head_lr = cfg["train"]["optimizer"]["lr"]

    params = [
        {"params": model.backbone.parameters(), "lr": backbone_lr},
        {"params": model.neck.parameters(), "lr": head_lr},
        {"params": model.head.parameters(), "lr": head_lr},
    ]
    if model.color_head is not None:
        params.append({"params": model.color_head.parameters(), "lr": head_lr})

    optimizer = torch.optim.AdamW(
        params,
        lr=head_lr,
        weight_decay=cfg["train"]["optimizer"]["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["train"]["epochs"],
        eta_min=cfg["train"]["scheduler"]["min_lr"],
    )

    # 7. Start Training
    trainer = SareeTrainer(
        model=model,
        train_loader=train_loader,
        gallery_loader=gallery_loader,
        query_loader=query_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_amp=cfg["train"]["mixed_precision"],
        output_dir=cfg["project"]["output_dir"],
        checkpoint_dir=cfg["project"]["checkpoint_dir"],
    )

    print(f"\n[*] Starting training for {cfg['train']['epochs']} epochs...")
    history = trainer.fit(epochs=cfg["train"]["epochs"], eval_interval=2)
    print("\n[+] Training complete! Best Checkpoint saved.")


if __name__ == "__main__":
    main()
