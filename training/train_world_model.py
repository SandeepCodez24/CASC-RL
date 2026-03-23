"""
train_world_model.py — Phase 1: World Model Pre-training.

Collects transition data from ConstellationEnv using a random policy, then
trains the EnsembleDynamicsNetwork (5-member) using MSE loss with per-member
Adam optimizers.

Usage (from project root):
    python training/train_world_model.py
    python training/train_world_model.py --n_transitions 200000 --n_epochs 150
    python training/train_world_model.py --n_satellites 6 --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys

# ── Make project root importable regardless of where script is run from ─────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from loguru import logger

from environment.constellation_env import ConstellationEnv
from world_model.world_model import WorldModel
from world_model.dataset_builder import DatasetBuilder, TransitionDataset
from world_model.training import WorldModelTrainer


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CASC-RL World Model")
    p.add_argument("--n_satellites",   type=int,   default=3,
                   help="Number of satellites in training constellation (default: 3)")
    p.add_argument("--n_transitions",  type=int,   default=100_000,
                   help="Transitions to collect for training dataset (default: 100000)")
    p.add_argument("--n_epochs",       type=int,   default=100,
                   help="Training epochs (default: 100)")
    p.add_argument("--batch_size",     type=int,   default=512,
                   help="Mini-batch size (default: 512)")
    p.add_argument("--lr",             type=float, default=1e-3,
                   help="Learning rate for Adam (default: 1e-3)")
    p.add_argument("--n_ensemble",     type=int,   default=5,
                   help="Number of ensemble members (default: 5)")
    p.add_argument("--hidden_dim",     type=int,   default=256,
                   help="Hidden layer width per ensemble member (default: 256)")
    p.add_argument("--n_layers",       type=int,   default=3,
                   help="Number of hidden layers per member (default: 3)")
    p.add_argument("--device",         type=str,   default="auto",
                   choices=["cpu", "cuda", "auto"],
                   help="Compute device (default: auto)")
    p.add_argument("--data_path",      type=str,   default="data/transitions.npz",
                   help="Path to save/load transition dataset (default: data/transitions.npz)")
    p.add_argument("--checkpoint_dir", type=str,   default="checkpoints",
                   help="Directory to save model checkpoints (default: checkpoints/)")
    p.add_argument("--load_data",      action="store_true",
                   help="Load existing dataset from --data_path instead of collecting new data")
    p.add_argument("--seed",           type=int,   default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--enable_eclipse", action="store_true", default=True,
                   help="Enable eclipse simulation during data collection")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main training pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Device resolution ────────────────────────────────────────────────────
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Using device: {device}")

    # ── Step 1: Build or load the transition dataset ─────────────────────────
    if args.load_data and os.path.exists(args.data_path):
        logger.info(f"Loading existing dataset from {args.data_path} ...")
        dataset = TransitionDataset.from_npz(args.data_path)
        logger.info(f"Loaded {len(dataset)} transitions.")
    else:
        logger.info(
            f"Creating ConstellationEnv with {args.n_satellites} satellites "
            f"(eclipse={'ON' if args.enable_eclipse else 'OFF'}) ..."
        )
        env = ConstellationEnv(
            n_satellites=args.n_satellites,
            enable_eclipse=args.enable_eclipse,
        )
        builder = DatasetBuilder(env, seed=args.seed)

        logger.info(f"Collecting {args.n_transitions:,} transitions with random policy ...")
        dataset = builder.collect(
            n_transitions=args.n_transitions,
            save_path=args.data_path,
        )
        env.close()

    # ── Step 2: Instantiate World Model ──────────────────────────────────────
    logger.info(
        f"Building WorldModel: ensemble={args.n_ensemble}, "
        f"hidden={args.hidden_dim}, layers={args.n_layers}"
    )
    world_model = WorldModel(
        state_dim=8,
        action_dim=5,
        n_ensemble=args.n_ensemble,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        device=device,
    )

    # ── Step 3: Train ────────────────────────────────────────────────────────
    trainer = WorldModelTrainer(
        world_model=world_model,
        device=device,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
    )
    logger.info(f"Starting world model training for {args.n_epochs} epochs ...")
    history = trainer.train(
        dataset=dataset,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        val_fraction=0.15,
        save_best=True,
        log_every=10,
    )

    # ── Step 4: Save final model ─────────────────────────────────────────────
    final_path = os.path.join(args.checkpoint_dir, "world_model_final.pt")
    world_model.save(final_path)
    logger.info(f"Final world model saved to: {final_path}")
    logger.success(
        f"World model training complete! "
        f"Best val loss: {min(history['val_loss']):.5f}"
    )


if __name__ == "__main__":
    main()
