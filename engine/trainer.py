"""
Training Engine for Color-Invariant Saree Recognition.
Supports AMP (fp16/bf16), Cosine Warmup Scheduling, Metric Validation, and Model Checkpointing.
"""

import os
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from models.saree_model import ColorInvariantSareeNet
from loss.combined_loss import CombinedSareeLoss
from engine.evaluator import SareeEvaluator


class SareeTrainer:
    """
    Manages model training, optimization, validation, and checkpointing.
    """

    def __init__(
        self,
        model: ColorInvariantSareeNet,
        train_loader: DataLoader,
        gallery_loader: DataLoader,
        query_loader: DataLoader,
        criterion: CombinedSareeLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_amp: bool = True,
        output_dir: str = "./outputs",
        checkpoint_dir: str = "./checkpoints",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.gallery_loader = gallery_loader
        self.query_loader = query_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp and (device.startswith("cuda"))
        self.scaler = GradScaler(enabled=self.use_amp)

        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.evaluator = SareeEvaluator(model=self.model, device=self.device)
        self.best_map = 0.0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Runs a single training epoch."""
        self.model.train()
        running_losses: Dict[str, float] = {}
        total_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        for batch in pbar:
            if len(batch) == 4:
                images, labels, color_labels, _ = batch
                color_labels = color_labels.to(self.device)
            else:
                images, labels, _ = batch
                color_labels = None

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                outputs = self.model(images, labels=labels)
                loss, loss_dict = self.criterion(
                    model_outputs=outputs,
                    motif_labels=labels,
                    color_labels=color_labels,
                )

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

            # Track metrics
            for k, v in loss_dict.items():
                running_losses[k] = running_losses.get(k, 0.0) + v

            pbar.set_postfix({k: f"{v / (pbar.n + 1):.4f}" for k, v in running_losses.items()})

        if self.scheduler is not None:
            self.scheduler.step()

        epoch_metrics = {k: v / total_batches for k, v in running_losses.items()}
        return epoch_metrics

    def fit(self, epochs: int = 30, eval_interval: int = 2) -> Dict[str, any]:
        """Runs full training loop with periodic evaluation."""
        history = {"train_loss": [], "val_map": [], "val_rank1": [], "val_auc": []}

        for epoch in range(epochs):
            train_metrics = self.train_epoch(epoch)
            history["train_loss"].append(train_metrics.get("loss_total", 0.0))

            print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_metrics.get('loss_total', 0.0):.4f}")

            # Validation step
            if (epoch + 1) % eval_interval == 0 or (epoch + 1) == epochs:
                eval_results = self.evaluator.evaluate(
                    gallery_loader=self.gallery_loader,
                    query_loader=self.query_loader,
                )
                m_ap = eval_results["identification"]["mAP"]
                r1 = eval_results["identification"].get("Rank-1", 0.0)
                auc_score = eval_results["verification"]["ROC-AUC"]

                history["val_map"].append(m_ap)
                history["val_rank1"].append(r1)
                history["val_auc"].append(auc_score)

                print(f"       >>> Val Identification: mAP = {m_ap:.2f}% | Rank-1 = {r1:.2f}%")
                print(f"       >>> Val Verification:   ROC-AUC = {auc_score:.2f}% | EER = {eval_results['verification']['EER']:.2f}%")

                # Save best checkpoint
                if m_ap > self.best_map:
                    self.best_map = m_ap
                    save_path = os.path.join(self.checkpoint_dir, "best_model.pth")
                    torch.save({
                        "epoch": epoch + 1,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_map": self.best_map,
                        "eval_results": eval_results,
                    }, save_path)
                    print(f"       [+] Saved new best model checkpoint to {save_path}")

        # Save final model
        final_path = os.path.join(self.checkpoint_dir, "latest_model.pth")
        torch.save(self.model.state_dict(), final_path)
        return history
