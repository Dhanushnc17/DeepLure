"""
Dataset implementations for Saree Design Identification and Verification.
"""

import os
from typing import List, Tuple, Dict, Optional, Callable
from pathlib import Path
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class SareeDataset(Dataset):
    """
    Standard Metric Learning Dataset for Saree Designs.
    Each sample belongs to a design class (`design_id`).
    Can extract an empirical color cluster label for adversarial color disentanglement.
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[Callable] = None,
        return_color_label: bool = False,
        num_color_clusters: int = 8,
    ):
        assert len(image_paths) == len(labels), "Paths and labels must have equal length"
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.return_color_label = return_color_label
        self.num_color_clusters = num_color_clusters

        # Map labels to contiguous indices
        self.unique_labels = sorted(list(set(labels)))
        self.label_to_idx = {lbl: idx for idx, lbl in enumerate(self.unique_labels)}
        self.num_classes = len(self.unique_labels)
        self.indexed_labels = [self.label_to_idx[lbl] for lbl in labels]

        # Group indices by class for fast balanced sampling
        self.class_to_indices: Dict[int, List[int]] = {}
        for idx, lbl in enumerate(self.indexed_labels):
            self.class_to_indices.setdefault(lbl, []).append(idx)

    def __len__(self) -> int:
        return len(self.image_paths)

    def _extract_color_cluster(self, img: Image.Image) -> int:
        """
        Quantizes the image's dominant HSV hue/saturation into one of N discrete bins.
        Used as pseudo-label for the adversarial color discriminator head.
        """
        small_img = img.resize((16, 16)).convert("HSV")
        arr = np.array(small_img)
        # Average Hue (0..255) and Saturation (0..255)
        avg_hue = np.mean(arr[:, :, 0])
        avg_sat = np.mean(arr[:, :, 1])
        if avg_sat < 35:  # Low saturation / near monochrome
            return 0
        bin_idx = 1 + int((avg_hue / 256.0) * (self.num_color_clusters - 1))
        return min(bin_idx, self.num_color_clusters - 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, ...]:
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        
        color_label = self._extract_color_cluster(img) if self.return_color_label else -1
        label = self.indexed_labels[idx]

        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        if self.return_color_label:
            return img_tensor, label, color_label, path
        return img_tensor, label, path


class SareeVerificationPairDataset(Dataset):
    """
    Dataset for Pairwise Saree Verification (Deliverable 2 & 3).
    Yields (Image_A, Image_B, is_same_design: 0 or 1).
    Constructs hard negative pairs (e.g. same colorway, different design)
    and positive pairs (same design, different colorways).
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str, int]],
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            pairs: List of tuples (path_a, path_b, label_same [1 or 0])
            transform: Transformations for validation/test images
        """
        self.pairs = pairs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str, str]:
        path_a, path_b, is_same = self.pairs[idx]
        img_a = Image.open(path_a).convert("RGB")
        img_b = Image.open(path_b).convert("RGB")

        if self.transform is not None:
            tensor_a = self.transform(img_a)
            tensor_b = self.transform(img_b)
        else:
            tensor_a = torch.from_numpy(np.array(img_a)).permute(2, 0, 1).float() / 255.0
            tensor_b = torch.from_numpy(np.array(img_b)).permute(2, 0, 1).float() / 255.0

        return tensor_a, tensor_b, is_same, path_a, path_b


def build_datasets_from_folder(
    data_dir: str,
    train_transform: Callable,
    val_transform: Callable,
    val_ratio: float = 0.2,
    query_ratio: float = 0.3,
    seed: int = 42,
    return_color_label: bool = False,
    num_color_clusters: int = 8,
) -> Tuple[SareeDataset, SareeDataset, SareeDataset]:
    """
    Scans a folder structured as:
      root_dir/
        design_001/
          colorway_red.jpg
          colorway_blue.jpg
        design_002/
          ...
    Splits into:
      - Train Dataset (80% of data / designs)
      - Gallery Dataset (reference database for evaluation)
      - Query Dataset (query images to search in gallery)
    """
    root_path = Path(data_dir)
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    
    design_dirs = [d for d in root_path.iterdir() if d.is_dir()]
    design_dirs.sort()

    rng = random.Random(seed)
    
    train_paths, train_labels = [], []
    gallery_paths, gallery_labels = [], []
    query_paths, query_labels = [], []

    for label_idx, design_dir in enumerate(design_dirs):
        imgs = [str(f) for f in design_dir.iterdir() if f.suffix.lower() in image_extensions]
        if len(imgs) == 0:
            continue
        
        rng.shuffle(imgs)
        
        if len(imgs) == 1:
            # If only 1 image exists, place in train
            train_paths.append(imgs[0])
            train_labels.append(label_idx)
        else:
            # Split instances between train and eval (or gallery and query)
            num_eval = max(1, int(len(imgs) * val_ratio))
            num_query = max(1, int(num_eval * query_ratio))
            num_gallery = num_eval - num_query
            if num_gallery == 0:
                num_gallery = 1
                num_query = max(1, len(imgs) - num_gallery - 1)

            q_imgs = imgs[:num_query]
            g_imgs = imgs[num_query : num_query + num_gallery]
            t_imgs = imgs[num_query + num_gallery :]

            if len(t_imgs) == 0:
                # Fallback if too few samples
                t_imgs = g_imgs

            for p in t_imgs:
                train_paths.append(p)
                train_labels.append(label_idx)
            for p in g_imgs:
                gallery_paths.append(p)
                gallery_labels.append(label_idx)
            for p in q_imgs:
                query_paths.append(p)
                query_labels.append(label_idx)

    train_dataset = SareeDataset(
        train_paths,
        train_labels,
        transform=train_transform,
        return_color_label=return_color_label,
        num_color_clusters=num_color_clusters,
    )
    gallery_dataset = SareeDataset(gallery_paths, gallery_labels, transform=val_transform)
    query_dataset = SareeDataset(query_paths, query_labels, transform=val_transform)

    return train_dataset, gallery_dataset, query_dataset
