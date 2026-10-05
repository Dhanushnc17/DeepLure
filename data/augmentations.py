"""
Color-Invariant Augmentation Pipeline for Saree Motif & Textile Recognition.

To ensure the model learns geometric structure, repeat patterns, weaves, borders,
and pallu motifs rather than memorizing color histograms, we apply an aggressive
suite of color-destructive and texture-preserving transformations.
"""

from typing import Tuple, List, Optional
import torch
import torchvision.transforms as T
from PIL import Image, ImageOps, ImageFilter
import random
import numpy as np


class RandomGrayscaleOrInvert(torch.nn.Module):
    """Randomly converts to grayscale or inverts image to break color cues."""
    def __init__(self, p_gray: float = 0.3, p_invert: float = 0.1):
        super().__init__()
        self.p_gray = p_gray
        self.p_invert = p_invert

    def forward(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p_invert:
            img = ImageOps.invert(img.convert("RGB"))
        elif random.random() < self.p_gray:
            img = ImageOps.grayscale(img).convert("RGB")
        return img


class RandomPosterizeSolarize(torch.nn.Module):
    """Random posterization and solarization to distort continuous color gradients."""
    def __init__(self, p_posterize: float = 0.2, p_solarize: float = 0.2):
        super().__init__()
        self.p_posterize = p_posterize
        self.p_solarize = p_solarize

    def forward(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p_posterize:
            bits = random.randint(2, 4)
            img = ImageOps.posterize(img, bits)
        if random.random() < self.p_solarize:
            threshold = random.randint(64, 192)
            img = ImageOps.solarize(img, threshold)
        return img


class RandomChannelShuffle(torch.nn.Module):
    """Permutes RGB channels to completely alter colorway while preserving geometry."""
    def __init__(self, p: float = 0.2):
        super().__init__()
        self.p = p

    def forward(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            channels = list(img.split())
            if len(channels) == 3:
                random.shuffle(channels)
                img = Image.merge("RGB", channels)
        return img


def get_train_transforms(
    image_size: Tuple[int, int] = (256, 256),
    color_jitter_params: Optional[dict] = None,
    p_gray: float = 0.35,
    p_invert: float = 0.1,
    p_posterize: float = 0.2,
    p_solarize: float = 0.2,
    p_channel_shuffle: float = 0.25,
) -> T.Compose:
    """
    Constructs the color-invariant training transform pipeline.
    """
    if color_jitter_params is None:
        color_jitter_params = {
            "brightness": 0.4,
            "contrast": 0.4,
            "saturation": 0.8,
            "hue": 0.5,
        }

    transforms_list = [
        # Spatial geometry & motif scale invariance
        T.RandomResizedCrop(
            size=image_size,
            scale=(0.7, 1.0),
            ratio=(0.8, 1.25),
            interpolation=T.InterpolationMode.BILINEAR,
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.1),
        T.RandomRotation(degrees=15),
        T.RandomPerspective(distortion_scale=0.25, p=0.3),
        
        # Colorway destruction & invariance enforcement
        RandomChannelShuffle(p=p_channel_shuffle),
        T.ColorJitter(
            brightness=color_jitter_params.get("brightness", 0.4),
            contrast=color_jitter_params.get("contrast", 0.4),
            saturation=color_jitter_params.get("saturation", 0.8),
            hue=color_jitter_params.get("hue", 0.5),
        ),
        RandomPosterizeSolarize(p_posterize=p_posterize, p_solarize=p_solarize),
        RandomGrayscaleOrInvert(p_gray=p_gray, p_invert=p_invert),
        
        # Tensor conversion & ImageNet standard normalization
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
    ]

    return T.Compose(transforms_list)


def get_val_transforms(image_size: Tuple[int, int] = (256, 256)) -> T.Compose:
    """
    Constructs the deterministic validation / inference transform pipeline.
    """
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
