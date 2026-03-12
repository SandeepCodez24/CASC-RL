"""
Action Selector — Layer 2b: Safety-Constrained Action Selection.

Implements a hard safety gate that overrides the RL policy's action choice
when physical constraints are violated. This prevents the learned policy from
commanding dangerous states (deep discharge, thermal overload, etc.)

Safety logic hierarchy (highest priority first):
    1. CRITICAL: SoC < SoC_CRITICAL  -> force CHARGE_PRIORITY
    2. WARNING:  SoC < SoC_MIN and action == PAYLOAD_ON -> redirect to PAYLOAD_OFF
    3. THERMAL:  Temp > T_MAX        -> force HIBERNATE
    4. NOMINAL:  pass through policy action unchanged

The selector also records when and why overrides occurred for logging / analysis.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional
from loguru import logger

# Actions (must match constellation_env.py)
ACTION_PAYLOAD_ON      = 0
ACTION_PAYLOAD_OFF     = 1
ACTION_HIBERNATE       = 2
ACTION_RELAY_MODE      = 3
ACTION_CHARGE_PRIORITY = 4

# State vector indices (must match OBS_DIM layout in constellation_env.py)
IDX_SOC   = 0    # State of Charge
IDX_SOH   = 1    # State of Health
IDX_TEMP  = 2    # Temperature (normalized: raw_T / 100.0)
IDX_PSOL  = 3    # Solar power  (normalized: raw_P / 200.0)


@dataclass
class SafetyConstraints:
    """
    Hard safety limits applied by the ActionSelector.

    Temperature values are in the normalized space used by constellation_env
    (divide raw Celsius by 100.0), so T_MAX=60°C -> 0.60 normalized.
    """
    soc_critical:    float = 0.10   # Force charge-priority below this SoC
    soc_min:         float = 0.15   # Prevent payload_ON below this SoC
    temp_max_norm:   float = 0.60   # Normalized: 60°C / 100 -> 0.60
    temp_warn_norm:  float = 0.50   # Normalized: 50°C / 100 -> 0.50


@dataclass
class OverrideRecord:
    """Record of a single safety override event."""
    step:           int
    agent_id:       int
    original_action: int
    safe_action:    int
    reason:         str


class ActionSelector:
    """
    Safety-constrained action selection gate.

    Wraps the raw RL policy action with physical safety checks.
    All checks operate on the normalized observation vector from the env.

    Args:
        constraints (SafetyConstraints): safety threshold configuration
        agent_id    (int)              : identifier for logging
    """

    def __init__(
        self,
        constraints: Optional[SafetyConstraints] = None,
        agent_id:    int = 0,
    ) -> None:
        self.constraints = constraints or SafetyConstraints()
        self.agent_id    = agent_id
        self._step       = 0
        self.override_log: list[OverrideRecord] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def select(
        self,
        policy_action: int,
        obs:           np.ndarray,
        verbose:       bool = False,
    ) -> Tuple[int, bool, str]:
        """
        Apply safety constraints to the policy's proposed action.

        Args:
            policy_action: integer action from the RL actor (0-4)
            obs:           observation vector from ConstellationEnv,
                           shape (OBS_DIM,) — normalized
            verbose:       if True, log every override to logger

        Returns:
            safe_action     (int) : the final action to execute
            was_overridden  (bool): True if policy was overridden
            reason          (str) : human-readable cause of override
        """
        self._step += 1
        soc  = float(obs[IDX_SOC])
        temp = float(obs[IDX_TEMP])   # already normalized in env
        c    = self.constraints

        # Priority 1: deep discharge — must charge immediately
        if soc < c.soc_critical:
            return self._override(policy_action, ACTION_CHARGE_PRIORITY, "SoC CRITICAL", verbose)

        # Priority 2: low SoC — prevent power-hungry payload
        if soc < c.soc_min and policy_action == ACTION_PAYLOAD_ON:
            return self._override(policy_action, ACTION_PAYLOAD_OFF, "SoC below min, payload denied", verbose)

        # Priority 3: thermal overload — hibernate immediately
        if temp > c.temp_max_norm:
            return self._override(policy_action, ACTION_HIBERNATE, "Temperature CRITICAL", verbose)

        # Priority 4: thermal warning — disable payload if on
        if temp > c.temp_warn_norm and policy_action == ACTION_PAYLOAD_ON:
            return self._override(policy_action, ACTION_PAYLOAD_OFF, "Temperature WARNING, payload denied", verbose)

        # Nominal: policy action is safe
        return policy_action, False, "nominal"

    def reset(self) -> None:
        """Reset step counter and override log (call at episode start)."""
        self._step = 0
        self.override_log.clear()

    def override_rate(self) -> float:
        """Fraction of steps where the policy was overridden."""
        if self._step == 0:
            return 0.0
        return len(self.override_log) / self._step

    def summary(self) -> str:
        """Return a summary of recent safety activity."""
        n = len(self.override_log)
        rate = self.override_rate()
        return (
            f"[SAT-{self.agent_id}] Safety overrides: {n}/{self._step} "
            f"({rate:.1%}) over episode"
        )

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _override(
        self,
        original: int,
        safe:     int,
        reason:   str,
        verbose:  bool,
    ) -> Tuple[int, bool, str]:
        record = OverrideRecord(
            step=self._step,
            agent_id=self.agent_id,
            original_action=original,
            safe_action=safe,
            reason=reason,
        )
        self.override_log.append(record)
        if verbose:
            logger.warning(
                f"[SAT-{self.agent_id}] Safety override at step {self._step}: "
                f"action {original} -> {safe} | reason: {reason}"
            )
        return safe, True, reason
