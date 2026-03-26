"""
curriculum_training.py — Phase 3: Full Curriculum Training Pipeline.

Runs staged training as defined in config/training.yaml:
  Stage 1: single_nominal    (1 sat, no eclipse,  1000 episodes)
  Stage 2: single_eclipse    (1 sat, eclipse,     2000 episodes)
  Stage 3: three_nominal     (3 sats, eclipse,    2000 episodes)
  Stage 4: three_degradation (3 sats, degradation,2000 episodes)
  Stage 5: six_stress        (6 sats, full model, 3000 episodes)
  Stage 6: twelve_adversarial(12 sats, anomalies, 5000 episodes)

Each stage loads the checkpoint from the previous stage.

Usage (from project root):
    python training/curriculum_training.py
    python training/curriculum_training.py --start_stage 3
    python training/curriculum_training.py --device cuda --checkpoint_dir checkpoints/curriculum
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from loguru import logger
from dataclasses import dataclass
from typing import List

from environment.constellation_env import ConstellationEnv
from world_model.world_model import WorldModel
from agents.policy_network import ActorNetwork
from agents.critic_network import CentralizedCriticNetwork
from marl.mappo_trainer import MAPPOTrainer
from marl.cooperative_rewards import CooperativeRewardShaper


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum stage definitions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CurriculumStage:
    name:               str
    n_satellites:       int
    n_episodes:         int
    enable_eclipse:     bool = True
    enable_degradation: bool = False
    adversarial:        bool = False


CURRICULUM_STAGES: List[CurriculumStage] = [
    CurriculumStage("single_nominal",     n_satellites=1,  n_episodes=500,
                    enable_eclipse=False, enable_degradation=False),
    CurriculumStage("single_eclipse",     n_satellites=1,  n_episodes=1000,
                    enable_eclipse=True,  enable_degradation=False),
    CurriculumStage("three_nominal",      n_satellites=3,  n_episodes=1000,
                    enable_eclipse=True,  enable_degradation=False),
    CurriculumStage("three_degradation",  n_satellites=3,  n_episodes=1000,
                    enable_eclipse=True,  enable_degradation=True),
    CurriculumStage("six_stress",         n_satellites=6,  n_episodes=1500,
                    enable_eclipse=True,  enable_degradation=True,  adversarial=False),
    CurriculumStage("twelve_adversarial", n_satellites=12, n_episodes=2500,
                    enable_eclipse=True,  enable_degradation=True,  adversarial=True),
]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CASC-RL Curriculum Training")
    p.add_argument("--start_stage",        type=int,   default=1,
                   help="Stage to start from (1-6). Loads checkpoint from previous stage.")
    p.add_argument("--end_stage",          type=int,   default=6,
                   help="Last stage to run (1-6, default: 6)")
    p.add_argument("--world_model_path",   type=str,
                   default="checkpoints/world_model_best.pt")
    p.add_argument("--checkpoint_dir",     type=str,
                   default="checkpoints/curriculum")
    p.add_argument("--episode_length",     type=int,   default=1000)
    p.add_argument("--rollout_length",     type=int,   default=200)
    p.add_argument("--device",             type=str,   default="auto",
                   choices=["cpu", "cuda", "auto"])
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--log_every",          type=int,   default=100)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Train one curriculum stage
# ─────────────────────────────────────────────────────────────────────────────

def train_stage(
    stage:          CurriculumStage,
    stage_num:      int,
    world_model:    WorldModel,
    prev_checkpoint: str,
    device:         str,
    args:           argparse.Namespace,
) -> str:
    """
    Train one curriculum stage. Returns path to saved checkpoint.
    """
    logger.info(
        f"\n{'='*60}\n"
        f"  STAGE {stage_num}: {stage.name.upper()}\n"
        f"  Satellites: {stage.n_satellites} | Episodes: {stage.n_episodes}\n"
        f"  Eclipse: {stage.enable_eclipse} | Degradation: {stage.enable_degradation}\n"
        f"{'='*60}"
    )

    obs_dim    = 8
    action_dim = 5
    n_sat      = stage.n_satellites

    env = ConstellationEnv(
        n_satellites=n_sat,
        enable_eclipse=stage.enable_eclipse,
    )

    actors = [
        ActorNetwork(state_dim=obs_dim, n_actions=action_dim, predict_k=5)
        for _ in range(n_sat)
    ]
    critic = CentralizedCriticNetwork(n_satellites=n_sat, state_dim=obs_dim)
    
    # Custom reward shaper weights
    from marl.cooperative_rewards import CoopRewardWeights
    reward_shaper = CooperativeRewardShaper(
        n_agents=n_sat,
        weights=CoopRewardWeights(alpha=0.5, beta=0.5, gamma=0.2)
    )

    trainer = MAPPOTrainer(
        n_agents=n_sat,
        actors=actors,
        critic=critic,
        env=env,
        world_model=world_model,
        device=device,
        episode_length=args.episode_length,
        checkpoint_dir=args.checkpoint_dir,
    )
    # Inject the shaper
    trainer.shaper = reward_shaper

    # Load previous stage checkpoint if available
    if prev_checkpoint and os.path.exists(prev_checkpoint):
        logger.info(f"Loading previous stage weights from: {prev_checkpoint}")
        trainer.load_checkpoint(prev_checkpoint)

    # Run episodes
    episode_rewards = []
    best_reward     = -float("inf")
    stage_suffix    = f"stage{stage_num}_{stage.name}"

    for episode in range(1, stage.n_episodes + 1):
        # MAPPOTrainer uses train_episode() which returns (reward, al, cl, ent)
        ep_reward, _, _, _ = trainer.train_episode()
        episode_rewards.append(ep_reward)

        if episode % args.log_every == 0:
            avg_r = float(np.mean(episode_rewards[-args.log_every:]))
            logger.info(f"  [Stage {stage_num} | Ep {episode:5d}/{stage.n_episodes}] avg_reward={avg_r:.3f}")

        if ep_reward > best_reward:
            best_reward = ep_reward
            trainer.save_checkpoint(suffix=stage_suffix)

    stage_ckpt_path = os.path.join(args.checkpoint_dir, f"mappo_{stage_suffix}.pt")
    logger.success(
        f"Stage {stage_num} ({stage.name}) complete. "
        f"Best reward: {best_reward:.3f} | Checkpoint: {stage_ckpt_path}"
    )
    env.close()
    return stage_ckpt_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else \
             args.device if args.device != "auto" else "cpu"
    logger.info(f"Curriculum training | device={device} | stages {args.start_stage}-{args.end_stage}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Load shared world model
    world_model = WorldModel(state_dim=8, action_dim=5, device=device)
    if os.path.exists(args.world_model_path):
        world_model.load(args.world_model_path)
        logger.info(f"World model loaded from {args.world_model_path}")
    else:
        logger.warning("No world model checkpoint found — training with untrained world model.")

    prev_checkpoint = None

    # Check for existing stage checkpoints to resume from
    if args.start_stage > 1:
        prev_stage_num  = args.start_stage - 1
        prev_stage      = CURRICULUM_STAGES[prev_stage_num - 1]
        prev_checkpoint = os.path.join(
            args.checkpoint_dir,
            f"stage{prev_stage_num}_{prev_stage.name}.pt"
        )
        logger.info(f"Starting from stage {args.start_stage}, loading: {prev_checkpoint}")

    # Run stages
    for stage_num in range(args.start_stage, args.end_stage + 1):
        stage = CURRICULUM_STAGES[stage_num - 1]
        prev_checkpoint = train_stage(
            stage, stage_num, world_model, prev_checkpoint, device, args
        )

    logger.success("Full curriculum training complete!")


if __name__ == "__main__":
    main()
