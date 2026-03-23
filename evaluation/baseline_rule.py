"""
baseline_rule.py — Rule-Based Heuristic Scheduler Baseline.

Implements a deterministic rule-based policy for satellite power management.
This is Baseline B from the paper comparison table (§8 Phase 8).

Strategy:
  1. If SoC < 0.25 and eclipse:     → hibernate (action=2)
  2. If SoC < 0.20:                 → charge_priority (action=4)
  3. If SoC > 0.80 and sunlit:      → payload_ON (action=0)
  4. If another sat is relay-capable → payload_OFF + let other relay (action=1)
  5. Default:                        → relay_mode (action=3)

This represents a well-tuned expert heuristic — a strong but non-learning baseline.
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Any, Optional

from evaluation.metrics import EpisodeResult


# Observation vector indices (must match constellation_env.py)
IDX_SOC   = 0
IDX_SOH   = 1
IDX_TEMP  = 2
IDX_SOLAR = 3
IDX_PHASE = 4
IDX_ECLIPSE = 5
IDX_PCONS   = 6
IDX_DELAY   = 7

# Action codes (must match constellation_env.py)
ACTION_PAYLOAD_ON      = 0
ACTION_PAYLOAD_OFF     = 1
ACTION_HIBERNATE       = 2
ACTION_RELAY_MODE      = 3
ACTION_CHARGE_PRIORITY = 4


class RuleBasedBaseline:
    """
    Rule-based heuristic satellite power management policy.

    Each satellite independently decides its action based on local observations.
    No communication or cooperation — pure local rule application.

    Args:
        soc_critical  : SoC threshold below which charge_priority is forced (default 0.20)
        soc_warning   : SoC threshold below which payload is suspended (default 0.30)
        soc_good      : SoC threshold above which scientific payload is allowed (default 0.70)
        eclipse_margin: Extra SoC penalty to add during eclipse window (default 0.10)
    """

    def __init__(
        self,
        soc_critical:   float = 0.20,
        soc_warning:    float = 0.30,
        soc_good:       float = 0.70,
        eclipse_margin: float = 0.10,
        name:           str   = "RuleBasedScheduler",
    ):
        self.soc_critical   = soc_critical
        self.soc_warning    = soc_warning
        self.soc_good       = soc_good
        self.eclipse_margin = eclipse_margin
        self.name           = name

    def select_action(self, obs: np.ndarray, sat_id: int = 0, global_obs: np.ndarray = None) -> int:
        """
        Select action for one satellite based on its local observation.

        Args:
            obs        : shape (8,) — local satellite observation
            sat_id     : satellite index (for relay coordination)
            global_obs : shape (N, 8) — all satellites (enables basic cooperation)

        Returns:
            int: action code [0-4]
        """
        soc     = float(obs[IDX_SOC])
        eclipse = bool(obs[IDX_ECLIPSE] > 0.5)
        solar   = float(obs[IDX_SOLAR])

        # ── Priority 1: Critical SoC → charge immediately ─────────────────────
        if soc < self.soc_critical:
            return ACTION_CHARGE_PRIORITY

        # ── Priority 2: Eclipse + low SoC → hibernate ────────────────────────
        if eclipse and soc < (self.soc_warning + self.eclipse_margin):
            return ACTION_HIBERNATE

        # ── Priority 3: Warning SoC → stop payload, relay only ───────────────
        if soc < self.soc_warning:
            return ACTION_RELAY_MODE

        # ── Priority 4: High SoC + sunlit → run payload ───────────────────────
        if soc > self.soc_good and not eclipse:
            # Basic cooperation: if another sat is already on payload, relay here
            if global_obs is not None and global_obs.shape[0] > 1:
                others_on_payload = sum(
                    1 for i, o in enumerate(global_obs)
                    if i != sat_id and float(o[IDX_PCONS]) > 0.4  # high consumption = payload ON
                )
                if others_on_payload >= global_obs.shape[0] // 2:
                    return ACTION_RELAY_MODE
            return ACTION_PAYLOAD_ON

        # ── Default: relay mode (moderate power, keeps comms up) ──────────────
        return ACTION_RELAY_MODE

    def select_actions(self, obs: np.ndarray) -> np.ndarray:
        """
        Select actions for all satellites. obs shape: (n_sat, 8).
        Returns np.ndarray of shape (n_sat,).
        """
        n_sat   = obs.shape[0]
        actions = np.array([
            self.select_action(obs[i], sat_id=i, global_obs=obs)
            for i in range(n_sat)
        ], dtype=np.int64)
        return actions


# ─────────────────────────────────────────────────────────────────────────────
# Episode Runner for Rule-Based Baseline
# ─────────────────────────────────────────────────────────────────────────────

def run_rule_based_episodes(
    n_episodes:    int   = 20,
    n_satellites:  int   = 3,
    episode_length: int  = 1000,
    enable_eclipse: bool = True,
    seeds:         List[int] = None,
    soc_critical_thr: float  = 0.15,
    thermal_norm_thr: float  = 0.60,
    verbose:       bool = False,
) -> List[EpisodeResult]:
    """
    Run N evaluation episodes with the rule-based baseline.
    Returns a list of EpisodeResult for MetricComputer.

    Args:
        n_episodes    : number of evaluation episodes
        n_satellites  : constellation size
        episode_length: max steps per episode
        enable_eclipse: enable eclipse simulation
        seeds         : list of seeds (length must equal n_episodes)
        soc_critical_thr: SoC below this = critical event
        thermal_norm_thr: normalized temp above this = thermal violation
    """
    from environment.constellation_env import ConstellationEnv

    if seeds is None:
        seeds = list(range(n_episodes))

    policy  = RuleBasedBaseline()
    results = []

    for ep_idx, seed in enumerate(seeds[:n_episodes]):
        env = ConstellationEnv(n_satellites=n_satellites, enable_eclipse=enable_eclipse)
        obs, _ = env.reset(seed=seed)

        result = EpisodeResult(
            episode_id=ep_idx, n_satellites=n_satellites,
            seed=seed, total_reward=0.0, algorithm="RuleBasedScheduler",
        )

        for step in range(episode_length):
            actions = policy.select_actions(obs)
            next_obs, rewards, terminated, truncated, info = env.step(actions)

            step_reward = float(sum(rewards))
            result.total_reward      += step_reward
            result.reward_trajectory.append(step_reward)
            result.soc_trajectory.append(next_obs[:, IDX_SOC].copy())
            result.soh_trajectory.append(next_obs[:, IDX_SOH].copy())
            result.temp_trajectory.append(next_obs[:, IDX_TEMP].copy())
            result.action_trajectory.append(actions.copy())

            # Safety event detection
            if (next_obs[:, IDX_SOC] < soc_critical_thr).any():
                result.soc_critical_steps.append(step)
            if (next_obs[:, IDX_TEMP] > thermal_norm_thr).any():
                result.thermal_viol_steps.append(step)

            result.episode_length = step + 1
            obs = next_obs
            if terminated or truncated:
                break

        if verbose:
            print(f"[RuleBase] Ep {ep_idx+1}/{n_episodes}: reward={result.total_reward:.3f} | "
                  f"SoC_min={min(s.min() for s in result.soc_trajectory):.3f} | "
                  f"thermal_viols={len(result.thermal_viol_steps)}")

        results.append(result)
        env.close()

    return results
