"""
World Model Training — Layer 2a.

Trains the EnsembleDynamicsNetwork using MSE loss on (s, a, s_next) tuples.
Supports:
- Per-ensemble-member independent optimization (diversity preservation)
- Per-dimension validation RMSE logging
- Best-checkpoint saving

Usage:
    trainer = WorldModelTrainer(world_model, device="cpu")
    trainer.train(dataset, n_epochs=100, batch_size=512)
"""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing import Dict, Optional
from loguru import logger

from world_model.world_model import WorldModel
from world_model.dataset_builder import TransitionDataset


# State dimension names for logging (matches OBS_DIM=8 from constellation_env)
STATE_DIM_NAMES = [
    "SoC", "SoH", "temperature", "P_solar",
    "orbital_phase", "eclipse_flag", "P_consumed", "comm_delay",
]


class WorldModelTrainer:
    """
    Training orchestrator for the WorldModel ensemble.

    Training strategy:
        Each ensemble member is trained with the same data but a different
        mini-batch ordering per epoch, which promotes diversity.

    Args:
        world_model  (WorldModel): the model to train
        device       (str)       : "cpu" or "cuda"
        lr           (float)     : learning rate for Adam. Default 1e-3.
        checkpoint_dir (str)     : directory to save checkpoints.
    """

    def __init__(
        self,
        world_model:    WorldModel,
        device:         str   = "cpu",
        lr:             float = 1e-3,
        checkpoint_dir: str   = "checkpoints",
    ) -> None:
        self.world_model    = world_model
        self.device         = torch.device(device)
        self.lr             = lr
        self.checkpoint_dir = checkpoint_dir
        self.loss_fn        = nn.MSELoss()

        # Separate Adam optimizer per ensemble member
        self.optimizers = [
            Adam(member.parameters(), lr=lr)
            for member in world_model.network.members
        ]

        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss":   [],
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        dataset:       TransitionDataset,
        n_epochs:      int   = 100,
        batch_size:    int   = 512,
        val_fraction:  float = 0.2,
        save_best:     bool  = True,
        log_every:     int   = 10,
    ) -> Dict[str, list]:
        """
        Train all ensemble members and periodically evaluate on validation set.

        Args:
            dataset      : full TransitionDataset
            n_epochs     : number of training epochs
            batch_size   : mini-batch size
            val_fraction : fraction of data held out for validation
            save_best    : save checkpoint when val loss improves
            log_every    : log frequency (epochs)

        Returns:
            history dict with "train_loss" and "val_loss" lists
        """
        # Fit normalizer on the training data
        states_np = dataset.states.numpy()
        self.world_model.fit_normalizer(states_np)
        logger.info("Normalizer fitted on training data.")

        train_ds, val_ds = dataset.split(val_fraction=val_fraction)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

        best_val_loss = float("inf")

        for epoch in range(1, n_epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_loss, per_dim_rmse = self._validate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if epoch % log_every == 0:
                dim_str = " | ".join(
                    f"{name}={rmse:.4f}"
                    for name, rmse in zip(STATE_DIM_NAMES, per_dim_rmse)
                )
                logger.info(
                    f"Epoch {epoch:4d}/{n_epochs} | "
                    f"train={train_loss:.5f} | val={val_loss:.5f} | "
                    f"[{dim_str}]"
                )

            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss)

        logger.info(f"Training complete. Best val loss: {best_val_loss:.5f}")
        return self.history

    def validate(self, dataset: TransitionDataset, batch_size: int = 512) -> Dict[str, float]:
        """
        Evaluate world model predictions on a held-out dataset.

        Returns:
            dict with "val_loss" and per-dim RMSE keys.
        """
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        val_loss, per_dim_rmse = self._validate(loader)
        result = {"val_loss": val_loss}
        for name, rmse in zip(STATE_DIM_NAMES, per_dim_rmse):
            result[f"rmse_{name}"] = float(rmse)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader) -> float:
        """One training epoch across all ensemble members."""
        self.world_model.train_mode()
        total_loss = 0.0
        n_batches  = 0

        for s, a, s_next in loader:
            s      = s.to(self.device)
            a      = a.to(self.device)
            s_next = s_next.to(self.device)

            # Normalize states
            s_norm      = self.world_model.state_normalizer.normalize(s)
            s_next_norm = self.world_model.state_normalizer.normalize(s_next)

            batch_loss = 0.0
            for idx, opt in enumerate(self.optimizers):
                opt.zero_grad()
                s_pred = self.world_model.network.forward_single(s_norm, a, member_idx=idx)
                # Convert s_pred back to unnormalized space for residual check
                loss = self.loss_fn(s_pred, s_next_norm)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.world_model.network.members[idx].parameters(), max_norm=1.0
                )
                opt.step()
                batch_loss += loss.item()

            total_loss += batch_loss / len(self.optimizers)
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    def _validate(self, loader: DataLoader):
        """Validate and return (mean_loss, per_dim_rmse)."""
        self.world_model.eval_mode()
        total_loss  = 0.0
        n_batches   = 0
        sq_errors   = None

        with torch.no_grad():
            for s, a, s_next in loader:
                s      = s.to(self.device)
                a      = a.to(self.device)
                s_next = s_next.to(self.device)

                s_norm      = self.world_model.state_normalizer.normalize(s)

                # Use ensemble mean for validation
                mean_pred, _ = self.world_model.network(s_norm, a)
                # Denormalize for interpretable RMSE
                mean_pred_denorm = self.world_model.state_normalizer.denormalize(mean_pred)

                loss = self.loss_fn(mean_pred_denorm, s_next)
                total_loss += loss.item()

                # Per-dim squared error (N, state_dim)
                se = (mean_pred_denorm - s_next) ** 2
                if sq_errors is None:
                    sq_errors = se.sum(dim=0).cpu().numpy()
                else:
                    sq_errors += se.sum(dim=0).cpu().numpy()

                n_batches += 1

        n_total = n_batches * loader.batch_size if loader.batch_size else 1
        per_dim_rmse = np.sqrt(sq_errors / max(n_total, 1))
        return total_loss / max(n_batches, 1), per_dim_rmse

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, "world_model_best.pt")
        self.world_model.save(path)
        logger.info(f"[Epoch {epoch}] New best model saved to {path} (val_loss={val_loss:.5f})")
