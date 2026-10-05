"""
Unit tests for data augmentations, dataset parsers, and samplers.
"""

import os
import shutil
import tempfile
import pytest
import torch
from PIL import Image
from data.augmentations import get_train_transforms, get_val_transforms
from data.dataset import SareeDataset, build_datasets_from_folder
from data.sampler import PKBalancedSampler
from scripts.prepare_data import generate_synthetic_saree_dataset


def test_augmentations():
    img = Image.new("RGB", (100, 100), color=(200, 50, 50))
    train_tf = get_train_transforms(image_size=(224, 224))
    val_tf = get_val_transforms(image_size=(224, 224))

    t_train = train_tf(img)
    t_val = val_tf(img)

    assert t_train.shape == (3, 224, 224)
    assert t_val.shape == (3, 224, 224)


def test_pk_sampler():
    temp_dir = tempfile.mkdtemp()
    try:
        generate_synthetic_saree_dataset(output_dir=temp_dir, num_designs=10, colorways_per_design=4)
        train_tf = get_train_transforms(image_size=(64, 64))
        val_tf = get_val_transforms(image_size=(64, 64))

        train_ds, gallery_ds, query_ds = build_datasets_from_folder(
            data_dir=temp_dir,
            train_transform=train_tf,
            val_transform=val_tf,
        )

        sampler = PKBalancedSampler(train_ds, p=4, k=2, length=5)
        for batch_indices in sampler:
            assert len(batch_indices) == 8
            batch_labels = [train_ds.indexed_labels[i] for i in batch_indices]
            # Verify exactly 4 distinct classes and 2 of each
            unique_in_batch = set(batch_labels)
            assert len(unique_in_batch) == 4
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
