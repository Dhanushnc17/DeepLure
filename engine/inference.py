"""
Production Inference & Retrieval Engine for Textile Motifs.
Supports single-query search against indexed gallery and 1:1 verification.
"""

from typing import List, Dict, Tuple, Union, Optional
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from models.saree_model import ColorInvariantSareeNet
from data.augmentations import get_val_transforms


class SareeSearchEngine:
    """
    High-performance inference engine for color-invariant saree identification & verification.
    """

    def __init__(
        self,
        model: ColorInvariantSareeNet,
        image_size: Tuple[int, int] = (256, 256),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verification_threshold: float = 0.65,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.transform = get_val_transforms(image_size=image_size)
        self.verification_threshold = verification_threshold

        # In-memory Gallery Index
        self.gallery_embeddings: Optional[torch.Tensor] = None
        self.gallery_paths: List[str] = []
        self.gallery_labels: List[Union[str, int]] = []

    def _preprocess(self, image_input: Union[str, Image.Image]) -> torch.Tensor:
        """Loads and pre-processes an image into a normalized torch tensor [1, 3, H, W]."""
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
        else:
            img = image_input.convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        return tensor

    @torch.no_grad()
    def get_embedding(self, image_input: Union[str, Image.Image, torch.Tensor]) -> torch.Tensor:
        """Extracts 1D normalized embedding [D] on CPU."""
        if not isinstance(image_input, torch.Tensor):
            tensor = self._preprocess(image_input)
        else:
            tensor = image_input.to(self.device)
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)

        emb = self.model.extract_features(tensor)
        return emb.cpu().squeeze(0)

    @torch.no_grad()
    def index_gallery(
        self,
        image_paths: List[str],
        labels: Optional[List[Union[str, int]]] = None,
        batch_size: int = 32,
    ):
        """
        Extracts and indexes embeddings for the reference saree gallery.
        """
        self.gallery_paths = list(image_paths)
        self.gallery_labels = list(labels) if labels is not None else [i for i in range(len(image_paths))]

        all_embs = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            tensors = torch.cat([self._preprocess(p) for p in batch_paths], dim=0)
            embs = self.model.extract_features(tensors).cpu()
            all_embs.append(embs)

        self.gallery_embeddings = torch.cat(all_embs, dim=0)
        print(f"[+] Successfully indexed {len(self.gallery_paths)} gallery items into search index.")

    @torch.no_grad()
    def search(
        self,
        query_image: Union[str, Image.Image],
        top_k: int = 5,
    ) -> List[Dict[str, Union[int, str, float]]]:
        """
        Searches the gallery for the best motif matches to the query image.
        Returns top-k ranked matches with cosine similarity scores.
        """
        assert self.gallery_embeddings is not None, "Gallery has not been indexed yet! Call index_gallery() first."

        query_emb = self.get_embedding(query_image).unsqueeze(0)  # [1, D]
        sims = torch.mm(query_emb, self.gallery_embeddings.t()).squeeze(0).numpy()

        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append({
                "rank": rank,
                "gallery_index": int(idx),
                "path": self.gallery_paths[idx],
                "design_id": self.gallery_labels[idx],
                "cosine_similarity": float(sims[idx]),
                "confidence_pct": float(max(0.0, sims[idx]) * 100.0),
            })
        return results

    @torch.no_grad()
    def verify(
        self,
        image_a: Union[str, Image.Image],
        image_b: Union[str, Image.Image],
        threshold: Optional[float] = None,
    ) -> Dict[str, Union[bool, float]]:
        """
        Verification task: Decides whether two images carry the SAME design / motif,
        regardless of colorway palette differences.
        """
        thresh = threshold if threshold is not None else self.verification_threshold
        emb_a = self.get_embedding(image_a)
        emb_b = self.get_embedding(image_b)

        # Cosine similarity between L2-normalized vectors is dot product
        similarity = float(torch.dot(emb_a, emb_b).item())
        is_same = bool(similarity >= thresh)

        return {
            "is_same_design": is_same,
            "cosine_similarity": similarity,
            "threshold_used": thresh,
            "confidence_margin": float(similarity - thresh),
        }
