"""
baseline_pid.py — PID Battery Controller Baseline.

Implements a classical PID controller for satellite power management.
This is Baseline A from the paper comparison table (§8 Phase 8).

The PID controller regulates SoC to a setpoint (default 0.75) by modulating
power consumption mode. Unlike the rule-based baseline, this uses continuous
error feedback with integral and derivative terms.

Controller:
    error     = SoC_setpoint - SoC_current
    P term    = Kp  × error
    I term    = Ki  × integral(error, dt)
    D term    = Kd  × d(error)/dt
    output    = P + I + D                    → mapped to discrete action

Action mapping (thresholded output):
    output > 0.5     → payload_ON       (more consumption allowed)
    0.2 < output ≤ 0.5 → relay_mode    (moderate consumption)
    -0.2 < output ≤ 0.2 → payload_OFF  (reduce consumption)
    output ≤ -0.2    → charge_priority  (emergency charging)

During eclipse, the setpoint is raised by `eclipse_boost` to prevent discharge.
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Any, Optional

from evaluation.metrics import EpisodeResult


# Observation vector indices
IDX_SOC     = 0
IDX_SOH     = 1
IDX_TEMP    = 2
IDX_SOLAR   = 3
IDX_PHASE   = 4
IDX_ECLIPSE = 5
IDX_PCONS   = 6
IDX_DELAY   = 7

# Action codes
ACTION_PAYLOAD_ON      = 0
ACTION_PAYLOAD_OFF     = 1
ACTION_HIBERNATE       = 2
ACTION_RELAY_MODE      = 3
ACTION_CHARGE_PRIORITY = 4


class PIDController:
    """
    Single-axis PID controller for SoC regulation.

    Args:
        setpoint    : target SoC (default 0.75 = 75%)
        Kp, Ki, Kd  : PID gains
        dt          : time step in seconds (default 10s = env dt)
        windup_limit: anti-windup clamp on integral term
        eclipse_boost: raise setpoint by this amount during eclipse
    """

    def __init__(
        self,
        setpoint:     float = 0.75,
        Kp:           float = 2.0,
        Ki:           float = 0.05,
        Kd:           float = 0.5,
        dt:           float = 10.0,
        windup_limit: float = 2.0,
        eclipse_boost: float = 0.10,
    ):
        self.setpoint     = setpoint
        self.Kp           = Kp
        self.Ki           = Ki
        self.Kd           = Kd
        self.dt           = dt
        self.windup_limit = windup_limit
        self.eclipse_boost = eclipse_boost

        # State
        self._integral    = 0.0
        self._prev_error  = 0.0

    def reset(self) -> None:
        self._integral   = 0.0
        self._prev_error = 0.0

    def compute(self, soc: float, in_eclipse: bool) -> float:
        """
        Compute PID output given current SoC.
        Returns a scalar in approximately [-2, 2].
        """
        sp    = self.setpoint + (self.eclipse_boost if in_eclipse else 0.0)
        error = sp - soc

        # Proportional
        P = self.Kp * error

        # Integral with anti-windup
        self._integral = float(np.clip(
            self._integral + error * self.dt,
            -self.windup_limit, self.windup_limit
        ))
        I = self.Ki * self._integral

        # Derivative (backward difference)
        D = self.Kd * (error - self._prev_error) / self.dt
        self._prev_error = error

        return P + I + D


class PIDBaseline:
    """
    Per-satellite PID controller baseline.

    Each satellite maintains its own independent PID controller (no cooperation).
    Maps continuous PID output to discrete action space {0,1,2,3,4}.

    Action thresholds (output → action):
        output > 0.6          → payload_ON   (SoC high, allow more consumption)
        0.2 < output ≤ 0.6   → relay_mode   (moderate)
        -0.2 ≤ output ≤ 0.2  → payload_OFF  (neutral)
        -0.6 < output < -0.2 → hibernate    (SoC dropping)
        output ≤ -0.6         → charge_priority (SoC low, charge urgently)

    Additionally: if SoC < 0.15 (hard safety), always charge_priority.
    """

    def __init__(
        self,
        n_satellites:      int   = 3,
        setpoint:          float = 0.75,
        Kp:                float = 2.0,
        Ki:                float = 0.05,
        Kd:                float = 0.5,
        dt:                float = 10.0,
        name:              str   = "PID_Controller",
    ):
        self.n_satellites = n_satellites
        self.name         = name
        self.controllers  = [
            PIDController(setpoint=setpoint, Kp=Kp, Ki=Ki, Kd=Kd, dt=dt)
            for _ in range(n_satellites)
        ]

    def _output_to_action(self, output: float, soc: float) -> int:
        """Map scalar PID output → discrete action code."""
        # Hard safety override
        if soc < 0.15:
            return ACTION_CHARGE_PRIORITY

        if output > 0.6:
            return ACTION_PAYLOAD_ON
        elif output > 0.2:
            return ACTION_RELAY_MODE
        elif output > -0.2:
            return ACTION_PAYLOAD_OFF
        elif output > -0.6:
            return ACTION_HIBERNATE
        else:
            return ACTION_CHARGE_PRIORITY

    def select_action(self, obs: np.ndarray, sat_id: int) -> int:
        """Select action for satellite `sat_id`."""
        soc     = float(obs[IDX_SOC])
        eclipse = bool(obs[IDX_ECLIPSE] > 0.5)
        pid_out = self.controllers[sat_id].compute(soc, in_eclipse=eclipse)
        return self._output_to_action(pid_out, soc)

    def select_actions(self, obs: np.ndarray) -> np.ndarray:
        """Select actions for all satellites. obs: (n_sat, 8)."""
        n_sat   = obs.shape[0]
        actions = np.array([self.select_action(obs[i], i) for i in range(n_sat)], dtype=np.int64)
        return actions

    def reset(self) -> None:
        for ctrl in self.controllers:
            ctrl.reset()


# ─────────────────────────────────────────────────────────────────────────────
# Episode Runner for PID Baseline
# ─────────────────────────────────────────────────────────────────────────────

def run_pid_episodes(
    n_episodes:      int   = 20,
    n_satellites:    int   = 3,
    episode_length:  int   = 1000,
    enable_eclipse:  bool  = True,
    seeds:           List[int] = None,
    soc_critical_thr: float = 0.15,
    thermal_norm_thr: float = 0.60,
    verbose:         bool  = False,
) -> List[EpisodeResult]:
    """
    Run N evaluation episodes with the PID controller baseline.

    Args:
        n_episodes    : number of evaluation seeds / episodes
        n_satellites  : constellation size
        episode_length: max steps per episode
        enable_eclipse: toggle eclipse simulation
        seeds         : list of seeds (length >= n_episodes)
        soc_critical_thr: threshold below which a step is counted as SoC-critical
        thermal_norm_thr: normalized temp above which a step is counted thermal violation
        verbose       : print per-episode summary

    Returns:
        List[EpisodeResult] — directly consumable by MetricComputer
    """
    from environment.constellation_env import ConstellationEnv

    if seeds is None:
        seeds = list(range(n_episodes))

    policy  = PIDBaseline(n_satellites=n_satellites)
    results = []

    for ep_idx, seed in enumerate(seeds[:n_episodes]):
        env = ConstellationEnv(n_satellites=n_satellites, enable_eclipse=enable_eclipse)
        obs, _ = env.reset(seed=seed)
        policy.reset()

        result = EpisodeResult(
            episode_id=ep_idx, n_satellites=n_satellites,
            seed=seed, total_reward=0.0, algorithm="PID_Controller",
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

            if (next_obs[:, IDX_SOC] < soc_critical_thr).any():
                result.soc_critical_steps.append(step)
            if (next_obs[:, IDX_TEMP] > thermal_norm_thr).any():
                result.thermal_viol_steps.append(step)

            result.episode_length = step + 1
            obs = next_obs
            if terminated or truncated:
                break

        if verbose:
            print(f"[PID] Ep {ep_idx+1}/{n_episodes}: reward={result.total_reward:.3f} | "
                  f"SoC_min={min(s.min() for s in result.soc_trajectory):.3f} | "
                  f"thermal_viols={len(result.thermal_viol_steps)}")

        results.append(result)
        env.close()

    return results
