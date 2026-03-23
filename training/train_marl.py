"""
train_marl.py — Phase 2: MAPPO Multi-Agent RL Training.

Trains all satellite agents jointly using Multi-Agent Proximal Policy
Optimization (MAPPO) with a centralized critic and decentralized actors.
Requires a pre-trained world model checkpoint from train_world_model.py.

Usage (from project root):
    python training/train_marl.py
    python training/train_marl.py --n_satellites 3 --n_episodes 5000
    python training/train_marl.py --n_satellites 6 --world_model_path checkpoints/world_model_best.pt --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from loguru import logger

from environment.constellation_env import ConstellationEnv
from world_model.world_model import WorldModel
from agents.satellite_agent import SatelliteAgent
from agents.policy_network import ActorNetwork
from agents.critic_network import CentralizedCriticNetwork
from marl.mappo_trainer import MAPPOTrainer
from marl.cooperative_rewards import CooperativeRewardShaper


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CASC-RL MAPPO Agents")
    p.add_argument("--n_satellites",        type=int,   default=3)
    p.add_argument("--n_episodes",          type=int,   default=5000)
    p.add_argument("--episode_length",      type=int,   default=1000,
                   help="Simulation steps per episode (default: 1000 = ~2.8 orbits)")
    p.add_argument("--rollout_length",      type=int,   default=200,
                   help="Steps to collect before each PPO update (default: 200)")
    p.add_argument("--n_epochs",            type=int,   default=10,
                   help="PPO update epochs per rollout (default: 10)")
    p.add_argument("--lr_actor",            type=float, default=3e-4)
    p.add_argument("--lr_critic",           type=float, default=1e-3)
    p.add_argument("--gamma",               type=float, default=0.99)
    p.add_argument("--lam",                 type=float, default=0.95)
    p.add_argument("--clip_epsilon",        type=float, default=0.2)
    p.add_argument("--entropy_coef",        type=float, default=0.01)
    p.add_argument("--batch_size",          type=int,   default=512)
    p.add_argument("--device",              type=str,   default="auto",
                   choices=["cpu", "cuda", "auto"])
    p.add_argument("--world_model_path",    type=str,
                   default="checkpoints/world_model_best.pt",
                   help="Path to pre-trained world model checkpoint")
    p.add_argument("--checkpoint_dir",      type=str,   default="checkpoints")
    p.add_argument("--save_every",          type=int,   default=500,
                   help="Save checkpoint every N episodes (default: 500)")
    p.add_argument("--enable_eclipse",      action="store_true", default=True)
    p.add_argument("--enable_degradation",  action="store_true", default=True)
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--log_every",           type=int,   default=50,
                   help="Log summary every N episodes (default: 50)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Device
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else \
             args.device if args.device != "auto" else "cpu"
    logger.info(f"Device: {device} | Satellites: {args.n_satellites} | Episodes: {args.n_episodes}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Step 1: Build environment ─────────────────────────────────────────────
    env = ConstellationEnv(
        n_satellites=args.n_satellites,
        enable_eclipse=args.enable_eclipse,
    )
    obs_dim    = 8                           # OBS_DIM per satellite
    action_dim = 5                           # N_ACTIONS
    global_obs_dim = obs_dim * args.n_satellites  # centralized critic input

    # ── Step 2: Load pre-trained world model ──────────────────────────────────
    world_model = WorldModel(
        state_dim=obs_dim,
        action_dim=action_dim,
        device=device,
    )
    if os.path.exists(args.world_model_path):
        world_model.load(args.world_model_path)
        logger.info(f"Loaded world model from {args.world_model_path}")
    else:
        logger.warning(
            f"World model checkpoint not found at {args.world_model_path}. "
            "Starting with untrained world model (predictions will be poor initially)."
        )

    # ── Step 3: Build agents ──────────────────────────────────────────────────
    actors = [
        ActorNetwork(obs_dim=obs_dim, future_dim=obs_dim, action_dim=action_dim)
        for _ in range(args.n_satellites)
    ]
    centralized_critic = CentralizedCriticNetwork(
        n_satellites=args.n_satellites,
        state_dim=obs_dim,
    )

    agents = [
        SatelliteAgent(
            agent_id=i,
            world_model=world_model,
            actor=actors[i],
            obs_dim=obs_dim,
        )
        for i in range(args.n_satellites)
    ]

    # ── Step 4: Build reward shaper ───────────────────────────────────────────
    reward_shaper = CooperativeRewardShaper(
        n_agents=args.n_satellites,
        alpha=0.5,
        beta=0.5,
        gamma=0.2,
    )

    # ── Step 5: Build MAPPO trainer ───────────────────────────────────────────
    trainer = MAPPOTrainer(
        env=env,
        actors=actors,
        critic=centralized_critic,
        reward_shaper=reward_shaper,
        n_satellites=args.n_satellites,
        device=device,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        lam=args.lam,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        rollout_length=args.rollout_length,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
    )

    # ── Step 6: Training loop ─────────────────────────────────────────────────
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_reward = -float("inf")
    episode_rewards = []

    logger.info("Starting MARL training ...")
    t_start = time.time()

    for episode in range(1, args.n_episodes + 1):
        ep_reward = trainer.run_episode(episode_length=args.episode_length)
        episode_rewards.append(ep_reward)

        if episode % args.log_every == 0:
            recent = episode_rewards[-args.log_every:]
            avg_r = float(np.mean(recent))
            elapsed = time.time() - t_start
            logger.info(
                f"Episode {episode:5d}/{args.n_episodes} | "
                f"avg_reward={avg_r:.3f} | "
                f"elapsed={elapsed/60:.1f}min"
            )

        # Save checkpoint
        if episode % args.save_every == 0:
            path = os.path.join(args.checkpoint_dir, f"mappo_ep{episode}.pt")
            trainer.save(path)
            logger.info(f"Checkpoint saved: {path}")

        # Track best
        if ep_reward > best_reward:
            best_reward = ep_reward
            trainer.save(os.path.join(args.checkpoint_dir, "mappo_best.pt"))

    # Final save
    trainer.save(os.path.join(args.checkpoint_dir, "mappo_final.pt"))
    logger.success(
        f"MARL training complete! Best episode reward: {best_reward:.3f} | "
        f"Total time: {(time.time()-t_start)/60:.1f} min"
    )
    env.close()


if __name__ == "__main__":
    main()
