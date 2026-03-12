"""
MAPPO Trainer — Layer 3: Cooperative Multi-Agent RL.

Implements Multi-Agent PPO (MAPPO) with:
    - Centralized training: shared CentralizedCriticNetwork observes global state
    - Decentralized execution: each actor uses only its local observation
    - Per-agent PPO clip updates with shared gradient accumulation
    - Entropy regularization for exploration
    - Gradient norm clipping for stability
    - Learning rate scheduling support

Architecture:
    actors   : [ActorNetwork] * n_agents   (individual, updated independently)
    critic   : CentralizedCriticNetwork    (shared, trained on global state)
    buffer   : RolloutBuffer               (holds one episode of transitions)
    gae      : GAEEstimator                (computes advantages + returns)
    comm     : CommunicationProtocol       (builds global state per step)
    shaper   : CooperativeRewardShaper     (blends local + global rewards)
"""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from typing import Dict, List, Optional, Tuple
from loguru import logger

from agents.policy_network      import ActorNetwork
from agents.critic_network      import CentralizedCriticNetwork
from agents.satellite_agent     import SatelliteAgent
from environment.constellation_env import ConstellationEnv
from marl.buffer                import RolloutBuffer
from marl.advantage_estimator   import GAEEstimator
from marl.cooperative_rewards   import CooperativeRewardShaper
from marl.communication_protocol import CommunicationProtocol


OBS_DIM   = 8
N_ACTIONS = 5


class MAPPOTrainer:
    """
    Multi-Agent PPO trainer for the CASC-RL constellation.

    Supports 3, 6, and 12 satellite configurations. Each satellite has its
    own actor network (decentralized execution). The centralized critic takes
    the concatenated global state from all satellites.

    Args:
        n_agents         (int)   : number of satellites
        actors           (list)  : list of ActorNetwork, one per agent
        critic           (CentralizedCriticNetwork): shared centralized critic
        world_model             : WorldModel instance (for agent forward passes)
        env              (ConstellationEnv): training environment
        gamma            (float) : discount factor.            Default 0.99
        lam              (float) : GAE lambda.                 Default 0.95
        clip_epsilon     (float) : PPO clip ratio.             Default 0.2
        entropy_coef     (float) : entropy regularization.     Default 0.01
        value_coef       (float) : critic loss coefficient.    Default 0.5
        max_grad_norm    (float) : gradient norm clip.         Default 0.5
        lr_actor         (float) : actor learning rate.        Default 3e-4
        lr_critic        (float) : critic learning rate.       Default 1e-3
        n_epochs         (int)   : PPO update epochs per rollout. Default 10
        batch_size       (int)   : mini-batch size.            Default 512
        episode_length   (int)   : max env steps per episode.  Default 1000
        checkpoint_dir   (str)   : directory for model saves.
        device           (str)   : "cpu" or "cuda"
    """

    def __init__(
        self,
        n_agents:       int,
        actors:         List[ActorNetwork],
        critic:         CentralizedCriticNetwork,
        env:            ConstellationEnv,
        world_model     = None,
        gamma:          float = 0.99,
        lam:            float = 0.95,
        clip_epsilon:   float = 0.2,
        entropy_coef:   float = 0.01,
        value_coef:     float = 0.5,
        max_grad_norm:  float = 0.5,
        lr_actor:       float = 3e-4,
        lr_critic:      float = 1e-3,
        n_epochs:       int   = 10,
        batch_size:     int   = 512,
        episode_length: int   = 1000,
        checkpoint_dir: str   = "checkpoints",
        device:         str   = "cpu",
    ) -> None:
        self.n_agents       = n_agents
        self.actors         = actors
        self.critic         = critic
        self.env            = env
        self.world_model    = world_model
        self.gamma          = gamma
        self.lam            = lam
        self.clip_epsilon   = clip_epsilon
        self.entropy_coef   = entropy_coef
        self.value_coef     = value_coef
        self.max_grad_norm  = max_grad_norm
        self.n_epochs       = n_epochs
        self.batch_size     = batch_size
        self.episode_length = episode_length
        self.checkpoint_dir = checkpoint_dir
        self.device         = torch.device(device)

        # Move networks to device
        for actor in self.actors:
            actor.to(self.device)
        self.critic.to(self.device)

        # Optimizers
        self.actor_optimizers = [
            Adam(actor.parameters(), lr=lr_actor) for actor in self.actors
        ]
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic)

        # Sub-modules
        self.buffer  = RolloutBuffer(n_agents, obs_dim=OBS_DIM, episode_len=episode_length)
        self.gae     = GAEEstimator(gamma=gamma, lam=lam)
        self.shaper  = CooperativeRewardShaper(n_agents=n_agents)
        self.comm    = CommunicationProtocol(n_agents=n_agents)

        # Training history
        self.history: Dict[str, List[float]] = {
            "episode_reward": [],
            "actor_loss":     [],
            "critic_loss":    [],
            "entropy":        [],
        }
        self._episode = 0
        self._best_reward = -float("inf")

    # ------------------------------------------------------------------
    # Factory classmethod
    # ------------------------------------------------------------------

    @classmethod
    def make(
        cls,
        n_agents:    int,
        env:         ConstellationEnv,
        predict_k:   int = 5,
        device:      str = "cpu",
        **kwargs,
    ) -> "MAPPOTrainer":
        """
        Factory: build actors + centralized critic and create a MAPPOTrainer.

        Args:
            n_agents  : number of satellites
            env       : ConstellationEnv instance
            predict_k : world model horizon for actor input
            device    : "cpu" or "cuda"
            **kwargs  : forwarded to MAPPOTrainer.__init__

        Returns:
            MAPPOTrainer
        """
        actors = [
            ActorNetwork(
                state_dim=OBS_DIM,
                n_actions=N_ACTIONS,
                predict_k=predict_k,
                hidden_dims=[256, 256, 128],
            )
            for _ in range(n_agents)
        ]
        critic = CentralizedCriticNetwork(
            n_satellites=n_agents,
            state_dim=OBS_DIM,
            hidden_dims=[256, 256],
        )
        return cls(n_agents=n_agents, actors=actors, critic=critic, env=env,
                   device=device, **kwargs)

    # ------------------------------------------------------------------
    # Core training loop
    # ------------------------------------------------------------------

    def train(
        self,
        n_episodes:   int,
        log_every:    int = 10,
        save_best:    bool = True,
    ) -> Dict[str, List[float]]:
        """
        Run the full MAPPO training for n_episodes.

        Args:
            n_episodes: total number of episodes to train
            log_every : log frequency in episodes
            save_best : save checkpoint when best episode reward is achieved

        Returns:
            history dict
        """
        logger.info(
            f"Starting MAPPO training: {n_episodes} episodes, "
            f"{self.n_agents} agents, device={self.device}"
        )

        for ep in range(1, n_episodes + 1):
            self._episode = ep
            ep_reward, actor_loss, critic_loss, entropy = self.train_episode()

            self.history["episode_reward"].append(ep_reward)
            self.history["actor_loss"].append(actor_loss)
            self.history["critic_loss"].append(critic_loss)
            self.history["entropy"].append(entropy)

            if ep % log_every == 0:
                logger.info(
                    f"[Episode {ep:5d}/{n_episodes}] "
                    f"reward={ep_reward:.3f} | "
                    f"actor_loss={actor_loss:.4f} | "
                    f"critic_loss={critic_loss:.4f} | "
                    f"entropy={entropy:.4f}"
                )

            if save_best and ep_reward > self._best_reward:
                self._best_reward = ep_reward
                self.save_checkpoint(suffix="best")

        logger.info(f"Training complete. Best episode reward: {self._best_reward:.3f}")
        return self.history

    def train_episode(self) -> Tuple[float, float, float, float]:
        """
        Run one full episode rollout and perform PPO updates.

        Returns:
            (episode_reward, mean_actor_loss, mean_critic_loss, mean_entropy)
        """
        self.buffer.clear()
        obs, _ = self.env.reset()              # (n_agents, OBS_DIM)

        episode_rewards: List[float] = []

        # ----- Rollout phase -----
        for _ in range(self.episode_length):
            actions, log_probs, value = self._collect_step(obs)

            # Step environment
            next_obs, env_rewards, terminated, truncated, _ = self.env.step(actions)
            env_rewards_arr = np.array([env_rewards] * self.n_agents, dtype=np.float32) \
                if np.isscalar(env_rewards) else np.asarray(env_rewards, dtype=np.float32)

            # Cooperative reward shaping
            shaped_rewards = self.shaper.shape(
                local_rewards=env_rewards_arr,
                actions=actions,
                obs=obs,
                mission_complete=bool(actions[0] == 0),  # payload_ON proxy
            )
            episode_rewards.append(shaped_rewards.mean())
            done = bool(terminated or truncated)

            # Global state for critic
            s_global = self.comm.get_global_state(obs)

            self.buffer.add(
                s_locals=obs,
                s_global=s_global,
                actions=actions,
                log_probs=log_probs,
                rewards=shaped_rewards,
                value=value,
                done=done,
            )

            obs = next_obs
            if done:
                break

        # Bootstrap value at end of episode
        with torch.no_grad():
            last_global = torch.tensor(
                self.comm.get_global_state(obs), dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            last_value = float(self.critic.value(last_global).item())

        # ----- Compute GAE -----
        tensors     = self.buffer.get_tensors(self.device)
        advantages, returns = self.gae.compute(
            rewards=tensors["rewards"],
            values=tensors["values"],
            dones=tensors["dones"],
            last_value=last_value,
            n_agents=self.n_agents,
        )

        # ----- PPO update epochs -----
        actor_losses:  List[float] = []
        critic_losses: List[float] = []
        entropies:     List[float] = []

        for _ in range(self.n_epochs):
            for batch in self.buffer.mini_batches(
                self.batch_size, self.device, advantages, returns
            ):
                al, cl, ent = self._update(batch)
                actor_losses.append(al)
                critic_losses.append(cl)
                entropies.append(ent)

        ep_reward    = float(np.mean(episode_rewards))
        mean_al      = float(np.mean(actor_losses))
        mean_cl      = float(np.mean(critic_losses))
        mean_entropy = float(np.mean(entropies))

        return ep_reward, mean_al, mean_cl, mean_entropy

    # ------------------------------------------------------------------
    # Step collection
    # ------------------------------------------------------------------

    def _collect_step(
        self,
        obs: np.ndarray,   # (n_agents, OBS_DIM)
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run all actors forward to collect actions, log-probs, and value.

        Args:
            obs: current observation matrix

        Returns:
            actions   : (n_agents,) int
            log_probs : (n_agents,) float
            value     : scalar float (centralized critic)
        """
        for actor in self.actors:
            actor.eval()
        self.critic.eval()

        actions   = np.zeros(self.n_agents, dtype=np.int64)
        log_probs = np.zeros(self.n_agents, dtype=np.float32)

        with torch.no_grad():
            for i, actor in enumerate(self.actors):
                s_t = torch.tensor(obs[i], dtype=torch.float32, device=self.device).unsqueeze(0)
                # World model future prediction (zeros if model not available)
                if self.world_model is not None:
                    try:
                        predict_k = actor.predict_k
                        futures_np = np.stack(
                            self.world_model.predict_k_steps(obs[i], [1] * predict_k, k=predict_k)
                        )
                    except Exception:
                        predict_k  = actor.predict_k
                        futures_np = np.zeros((predict_k, OBS_DIM), dtype=np.float32)
                else:
                    predict_k  = actor.predict_k
                    futures_np = np.zeros((predict_k, OBS_DIM), dtype=np.float32)

                s_future = torch.tensor(futures_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                action, log_prob, _ = actor.act(s_t, s_future)
                actions[i]   = int(action.item())
                log_probs[i] = float(log_prob.item())

            # Centralized critic on global state
            s_global = torch.tensor(
                self.comm.get_global_state(obs), dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            value = float(self.critic.value(s_global).item())

        return actions, log_probs, value

    # ------------------------------------------------------------------
    # PPO update step
    # ------------------------------------------------------------------

    def _update(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, float, float]:
        """
        Perform one mini-batch PPO update for all actors and the critic.

        Args:
            batch: dict from RolloutBuffer.mini_batches()

        Returns:
            (actor_loss, critic_loss, entropy)
        """
        for actor in self.actors:
            actor.train()
        self.critic.train()

        s_locals   = batch["s_locals"]     # (B, n_agents, OBS_DIM)
        s_globals  = batch["s_globals"]    # (B, n_agents * OBS_DIM)
        actions    = batch["actions"]       # (B, n_agents)
        old_lp     = batch["log_probs"]    # (B, n_agents)
        advantages = batch["advantages"]   # (B, n_agents)
        returns    = batch["returns"]      # (B, n_agents)

        B = s_locals.shape[0]

        # ----- Actor update (one per agent, PPO clip) -----
        total_actor_loss = 0.0
        total_entropy    = 0.0

        for i, (actor, opt) in enumerate(zip(self.actors, self.actor_optimizers)):
            s_t       = s_locals[:, i, :]             # (B, OBS_DIM)
            predict_k = actor.predict_k
            s_future  = torch.zeros(B, predict_k, OBS_DIM, device=self.device)  # (B, k, OBS_DIM)
            a_i       = actions[:, i]                  # (B,)
            old_lp_i  = old_lp[:, i]                  # (B,)
            adv_i     = advantages[:, i]               # (B,)

            new_lp, entropy = actor.evaluate_actions(s_t, s_future.view(B, -1), a_i)

            ratio   = torch.exp(new_lp - old_lp_i)
            surr1   = ratio * adv_i
            surr2   = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv_i
            a_loss  = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

            opt.zero_grad()
            a_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
            opt.step()

            total_actor_loss += a_loss.item()
            total_entropy    += entropy.mean().item()

        mean_actor_loss = total_actor_loss / self.n_agents
        mean_entropy    = total_entropy    / self.n_agents

        # ----- Critic update (centralized, MSE on returns) -----
        v_pred  = self.critic.value(s_globals)                        # (B,)
        v_target = returns.mean(dim=-1)                               # (B,) mean over agents
        c_loss  = nn.functional.mse_loss(v_pred, v_target.detach()) * self.value_coef

        self.critic_optimizer.zero_grad()
        c_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        return mean_actor_loss, c_loss.item(), mean_entropy

    # ------------------------------------------------------------------
    # Rollout (inference-mode, used for evaluation)
    # ------------------------------------------------------------------

    def rollout(
        self,
        n_steps:      int = 200,
        deterministic: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Run a full evaluation rollout without gradient updates.

        Args:
            n_steps      : steps to simulate
            deterministic: use argmax actions

        Returns:
            dict of trajectory arrays
        """
        for actor in self.actors:
            actor.eval()
        self.critic.eval()

        obs, _ = self.env.reset()
        rewards:  List[float] = []
        soc_hist: List[np.ndarray] = []

        with torch.no_grad():
            for _ in range(n_steps):
                actions, _, _ = self._collect_step(obs)
                obs, reward, terminated, truncated, info = self.env.step(actions)
                rewards.append(float(reward))
                if "soc" in info:
                    soc_hist.append(np.array(info["soc"]))
                if terminated or truncated:
                    break

        return {
            "rewards": np.array(rewards),
            "soc":     np.stack(soc_hist) if soc_hist else np.array([]),
        }

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_checkpoint(self, suffix: str = "") -> None:
        """Save all actor and critic weights."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        tag  = f"_{suffix}" if suffix else ""
        path = os.path.join(self.checkpoint_dir, f"mappo{tag}.pt")
        torch.save({
            "episode":  self._episode,
            "actors":   [a.state_dict() for a in self.actors],
            "critic":   self.critic.state_dict(),
            "history":  self.history,
        }, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load actors and critic from a checkpoint file."""
        ckpt = torch.load(path, map_location=self.device)
        for actor, sd in zip(self.actors, ckpt["actors"]):
            actor.load_state_dict(sd)
        self.critic.load_state_dict(ckpt["critic"])
        self.history   = ckpt.get("history", self.history)
        self._episode  = ckpt.get("episode", 0)
        logger.info(f"Checkpoint loaded from {path} (episode {self._episode})")
