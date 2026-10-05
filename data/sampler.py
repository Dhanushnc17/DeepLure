"""
Balanced P-K Batch Sampler for Metric Learning in Textile Motif Retrieval.

Each batch samples P distinct design classes, and K samples (different colorways / views)
per design class. This ensures every mini-batch contains both hard positive pairs
(same motif, different colorways) and hard negative pairs (different motifs).
"""

from typing import Iterator, List
import random
import torch
from torch.utils.data import Sampler
from data.dataset import SareeDataset


class PKBalancedSampler(Sampler[List[int]]):
    """
    Randomly samples P classes, and for each class samples K instances.
    Batch size = P * K.
    """

    def __init__(self, dataset: SareeDataset, p: int = 8, k: int = 4, length: int = 1000):
        super().__init__(dataset)
        self.dataset = dataset
        self.p = min(p, dataset.num_classes)
        self.k = k
        self.batch_size = self.p * self.k
        self.length = length  # Number of batches per epoch

        # Filter classes that have at least 1 instance
        self.available_classes = [
            cls_idx for cls_idx, idxs in dataset.class_to_indices.items() if len(idxs) > 0
        ]

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.length):
            batch = []
            selected_classes = random.sample(self.available_classes, self.p)

            for cls in selected_classes:
                indices = self.dataset.class_to_indices[cls]
                if len(indices) >= self.k:
                    sampled_indices = random.sample(indices, self.k)
                else:
                    # Oversample with replacement if class has fewer than K instances
                    sampled_indices = [random.choice(indices) for _ in range(self.k)]
                batch.extend(sampled_indices)

            random.shuffle(batch)
            yield batch
